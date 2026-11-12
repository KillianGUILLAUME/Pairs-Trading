"""
neural_nets/signature_torch.py

Signature tronquée différentiable en pur PyTorch (CPU / CUDA / MPS).

Remplace `signatory` (non maintenu, incompatible torch >= 2) et `iisignature`
(CPU, non différentiable) pour tout ce qui touche à la loss MMD de la Neural SDE.

Principe : pour un chemin linéaire par morceaux x_0, ..., x_{L-1} dans R^d,
    Sig(x) = exp(Δ_1) ⊗ exp(Δ_2) ⊗ ... ⊗ exp(Δ_{L-1}),   Δ_i = x_i - x_{i-1},
avec exp(Δ) = (1, Δ, Δ⊗Δ/2!, Δ⊗Δ⊗Δ/3!, ...) (identité de Chen).
Le produit dans l'algèbre tensorielle tronquée est associatif : on réduit les
L-1 segments par un arbre binaire (log2(L) étapes vectorisées sur le batch).

Conventions identiques à iisignature / signatory :
    - sortie (B, d + d² + ... + d^N), niveaux concaténés, indices multiples en
      ordre lexicographique (row-major) ;
    - le niveau 0 (toujours égal à 1) n'est pas renvoyé.

Augmentations :
    - `time_augment` : ajoute un canal temps t ∈ [0, 1] en première coordonnée.
      SANS ce canal, la signature est invariante par reparamétrisation du temps
      (deux chemins de même géométrie parcourus à des vitesses différentes ont la
      même signature) : la loss ne verrait ni la demi-vie ni le clustering de
      volatilité. AVEC ce canal, elle les distingue.
    - `lead_lag` : transformation lead-lag (doubling de la dimension) qui rend
      la volatilité quadratique visible dans les termes de niveau 2.
"""

from __future__ import annotations

import torch


def sig_channels(dim: int, depth: int) -> int:
    """Nombre de coefficients d'une signature tronquée (niveau 0 exclu)."""
    return sum(dim ** k for k in range(1, depth + 1))


def augment_time(path: torch.Tensor) -> torch.Tensor:
    """(B, L, d) -> (B, L, d + 1) avec t = linspace(0, 1, L) en première coordonnée."""
    B, L, _ = path.shape
    t = torch.linspace(0.0, 1.0, L, device=path.device, dtype=path.dtype)
    t = t.view(1, L, 1).expand(B, L, 1)
    return torch.cat([t, path], dim=-1)


def lead_lag(path: torch.Tensor) -> torch.Tensor:
    """
    Transformation lead-lag : (B, L, d) -> (B, 2L - 1, 2d).
    Points : (x_i, x_i), (x_{i+1}, x_i), (x_{i+1}, x_{i+1}), ...
    """
    B, L, d = path.shape
    lead = path[:, 1:]            # x_{i+1}
    lag = path[:, :-1]            # x_i
    # interleave : (x_i, x_i) puis (x_{i+1}, x_i)
    diag = torch.cat([lag, lag], dim=-1)                 # (B, L-1, 2d)
    off = torch.cat([lead, lag], dim=-1)                 # (B, L-1, 2d)
    inter = torch.stack([diag, off], dim=2).reshape(B, 2 * (L - 1), 2 * d)
    last = torch.cat([path[:, -1:], path[:, -1:]], dim=-1)  # (x_{L-1}, x_{L-1})
    return torch.cat([inter, last], dim=1)


def _segment_exponentials(inc: torch.Tensor, depth: int) -> list[torch.Tensor]:
    """
    exp(Δ) tronqué pour chaque segment.
    inc : (B, S, d)  ->  [Δ, Δ⊗Δ/2!, ..., Δ^{⊗N}/N!] de formes (B, S, d^k).
    """
    levels = [inc]
    cur = inc
    for k in range(2, depth + 1):
        cur = (cur.unsqueeze(-1) * inc.unsqueeze(-2)).reshape(*inc.shape[:-1], -1) / k
        levels.append(cur)
    return levels


def _tensor_product(A: list[torch.Tensor], B: list[torch.Tensor]) -> list[torch.Tensor]:
    """
    Produit dans l'algèbre tensorielle tronquée, niveau 0 implicite égal à 1 :
        (A ⊗ B)_k = A_k + B_k + Σ_{i+j=k, i,j>=1} A_i ⊗ B_j
    A, B : listes [niveau 1, ..., niveau N] de formes (B, S, d^k).
    """
    depth = len(A)
    out = []
    for k in range(1, depth + 1):
        acc = A[k - 1] + B[k - 1]
        for i in range(1, k):
            j = k - i
            prod = A[i - 1].unsqueeze(-1) * B[j - 1].unsqueeze(-2)   # (B, S, d^i, d^j)
            acc = acc + prod.reshape(*prod.shape[:-2], -1)
        out.append(acc)
    return out


def signature(
    path: torch.Tensor,
    depth: int,
    time_augment: bool = False,
    lead_lag_transform: bool = False,
) -> torch.Tensor:
    """
    Signature tronquée d'un batch de chemins.

    Args:
        path  : (B, L, d) ou (L, d).
        depth : profondeur de troncature N >= 1.
    Returns:
        (B, sig_channels(d', N)) avec d' la dimension après augmentations.
    """
    if path.dim() == 2:
        path = path.unsqueeze(0)
    if depth < 1:
        raise ValueError("depth doit être >= 1")
    if lead_lag_transform:
        path = lead_lag(path)
    if time_augment:
        path = augment_time(path)

    inc = path[:, 1:] - path[:, :-1]                     # (B, S, d)
    if inc.shape[1] == 0:
        B, _, d = path.shape
        return torch.zeros(B, sig_channels(d, depth), device=path.device, dtype=path.dtype)

    levels = _segment_exponentials(inc, depth)

    # Réduction en arbre binaire (associativité du produit tensoriel).
    while levels[0].shape[1] > 1:
        S = levels[0].shape[1]
        if S % 2 == 1:
            # Padding par l'élément neutre (1, 0, 0, ...) : zéros à tous les niveaux >= 1.
            levels = [torch.cat([lv, torch.zeros_like(lv[:, :1])], dim=1) for lv in levels]
        A = [lv[:, 0::2] for lv in levels]
        Bv = [lv[:, 1::2] for lv in levels]
        levels = _tensor_product(A, Bv)

    return torch.cat([lv[:, 0] for lv in levels], dim=-1)


class TruncatedSignature(torch.nn.Module):
    """Module sans paramètre : `TruncatedSignature(depth, time_augment)(paths)`."""

    def __init__(self, depth: int, time_augment: bool = True, lead_lag_transform: bool = False):
        super().__init__()
        self.depth = depth
        self.time_augment = time_augment
        self.lead_lag_transform = lead_lag_transform

    def out_channels(self, data_dim: int) -> int:
        d = data_dim * 2 if self.lead_lag_transform else data_dim
        d = d + 1 if self.time_augment else d
        return sig_channels(d, self.depth)

    def forward(self, path: torch.Tensor) -> torch.Tensor:
        return signature(path, self.depth, self.time_augment, self.lead_lag_transform)

    def extra_repr(self) -> str:
        return f"depth={self.depth}, time_augment={self.time_augment}, lead_lag={self.lead_lag_transform}"


@torch.no_grad()
def signature_chunked(
    paths: torch.Tensor,
    depth: int,
    time_augment: bool = True,
    lead_lag_transform: bool = False,
    chunk_size: int = 4096,
    device: str | torch.device | None = None,
) -> torch.Tensor:
    """Signature par morceaux (pré-calcul de dataset) ; renvoie un tenseur CPU float32."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    out = []
    for i in range(0, len(paths), chunk_size):
        chunk = paths[i:i + chunk_size].to(device=device, dtype=torch.float32)
        out.append(signature(chunk, depth, time_augment, lead_lag_transform).cpu())
    return torch.cat(out, dim=0)


def sig_np(path, depth: int, time_augment: bool = False):
    """
    Remplaçant numpy de `iisignature.sig(path, depth)` : path (L, d) -> (C,) ou (B, L, d) -> (B, C),
    float64, CPU. Même ordre des coefficients.
    """
    import numpy as np
    arr = np.asarray(path, dtype=np.float64)
    squeeze = arr.ndim == 2
    out = signature(torch.from_numpy(arr), depth, time_augment=time_augment).numpy()
    return out[0] if squeeze else out
