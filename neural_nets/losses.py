"""
neural_nets/losses.py

Losses et boucles d'entraînement :
    - Conditional Flow Matching / Bridge Matching / régularisation entropique
      (approche autorégressive précédente, dormante, base du futur DSBM) ;
    - Kernel Signature MMD + `train_sde` pour la Neural SDE (chaîne active).

Dépendances : `neural_nets.signature_torch` (signature tronquée différentiable,
CPU/CUDA/MPS). Plus de dépendance à `signatory`.
"""

from __future__ import annotations

import math
import os
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F

try:
    from neural_nets.signature_torch import TruncatedSignature
except ImportError:  # exécution depuis le dossier neural_nets/
    from signature_torch import TruncatedSignature


# ============================================
# Loss pour entrainer le velocity field (CFM)
# ============================================

class ConditionalFlowMatchingLoss:
    """
    Loss pour entraîner le velocity field

    Formulation :
        À chaque pas de temps physique t, on transporte un bruit source
        Δ₀ ~ N(0, I) vers l'incrément réel Δ₁ = X_{t+1} - X_t.
        Le velocity field est conditionné par la signature du passé Sig_{t-w,t},
        ce qui garantit la causalité : aucune information future n'est utilisée.

        Le temps de diffusion τ ∈ [0, 1] est distinct du temps physique t.
        L'interpolant est : Δ_τ = (1 - τ) Δ₀ + τ Δ₁
        La target velocity est : v* = Δ₁ - Δ₀ = dΔ_τ/dτ
    """
    def __init__(self, velocity_field, interpolant):
        self.velocity_field = velocity_field
        self.interpolant = interpolant

    def __call__(self, x_0, x_1, precomputed_sig=None):
        batch_size = x_0.shape[0]
        device = x_0.device
        t = torch.rand(batch_size, 1, device=device)
        x_t, v_target = self.interpolant.calc_xt_ut(x_0, x_1, t)
        v_pred = self.velocity_field(x_t, t, precomputed_sig=precomputed_sig)
        return F.mse_loss(v_pred, v_target)


# ============================================
# Bridge Matching Loss
# ============================================

class BridgeMatchingLoss:
    """
    Bridge Matching (Shi, De Bortoli, Campbell, Doucet — NeurIPS 2023, §3.2).
        Δ_τ = (1 - τ) Δ₀ + τ Δ₁ + σ √(τ(1 - τ)) Z,   u*(Δ_τ, τ | Δ₁) = (Δ₁ - Δ_τ) / (1 - τ)
    NB : avec un couplage indépendant (Δ₀, Δ₁) c'est UNE demi-itération de DSBM ;
    le pont de Schrödinger complet demande la boucle IMF (forward / backward).
    """
    def __init__(self, drift_network, sigma: float = 0.5):
        self.drift_network = drift_network
        self.sigma = sigma

    def __call__(self, x0: torch.Tensor, x1: torch.Tensor) -> torch.Tensor:
        batch_size = x0.shape[0]
        device = x0.device
        eps = 1e-3
        t = torch.rand(batch_size, 1, device=device) * (1.0 - eps)
        mean_t = (1.0 - t) * x0 + t * x1
        std_t = self.sigma * torch.sqrt(t * (1.0 - t) + 1e-8)
        z = torch.randn_like(x0)
        x_t = mean_t + std_t * z
        denom = torch.clamp(1.0 - t, min=eps)
        u_target = (x1 - x_t) / denom
        u_pred = self.drift_network(x_t, t)
        return F.mse_loss(u_pred, u_target)


# ============================================
# Régularisation entropique (anti mode collapse)
# ============================================

class EntropicCFMLoss:
    """
    L = L_CFM + λ_H · max(0, ε_min - Var_B[v_θ])   (hinge sur la variance des prédictions).
    """

    def __init__(self, velocity_field, interpolant, lambda_entropy: float = 0.1, min_variance: float = 0.01):
        self.velocity_field = velocity_field
        self.interpolant = interpolant
        self.lambda_entropy = lambda_entropy
        self.min_variance = min_variance

    def __call__(self, x_0, x_1):
        batch_size = x_0.shape[0]
        device = x_0.device
        t = torch.rand(batch_size, 1, 1, device=device)
        x_t, v_target = self.interpolant.calc_xt_ut(x_0, x_1, t)
        v_pred = self.velocity_field(x_t, t)
        cfm_loss = F.mse_loss(v_pred, v_target)
        pred_var = v_pred.var(dim=0).mean()
        entropy_loss = F.relu(self.min_variance - pred_var)
        total_loss = cfm_loss + self.lambda_entropy * entropy_loss
        metrics = {
            "cfm_loss": cfm_loss.item(),
            "entropy_loss": entropy_loss.item(),
            "pred_variance": pred_var.item(),
        }
        return total_loss, metrics


def train_teacher(
    velocity_field,
    data_loader,
    interpolant,
    num_epochs: int = 1000,
    lr: float = 1e-3,
    device: str = 'cuda',
    lambda_entropy: float = 0.0,
    min_variance: float = 0.01,
    use_wandb: bool = False
):
    """Boucle d'entraînement CFM (dormante : chaîne autorégressive précédente)."""
    use_entropic = lambda_entropy > 0.0
    if use_entropic:
        loss_fn = EntropicCFMLoss(velocity_field, interpolant=interpolant,
                                  lambda_entropy=lambda_entropy, min_variance=min_variance)
    else:
        loss_fn = ConditionalFlowMatchingLoss(velocity_field, interpolant=interpolant)

    optimizer = torch.optim.AdamW(velocity_field.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=50)

    velocity_field.to(device)
    velocity_field.train()

    for epoch in range(num_epochs):
        epoch_loss = epoch_cfm = epoch_ent = 0.0
        num_batches = 0
        for batch in data_loader:
            x_1 = (batch[0] if isinstance(batch, (list, tuple)) else batch).to(device)
            optimizer.zero_grad()
            x_0 = interpolant.sample_x0(x_1.shape) if hasattr(interpolant, 'sample_x0') else torch.randn_like(x_1)
            x_1 = torch.nan_to_num(x_1, nan=0.0, posinf=1e4, neginf=-1e4)
            if use_entropic:
                loss, metrics = loss_fn(x_0, x_1)
                epoch_cfm += metrics["cfm_loss"]
                epoch_ent += metrics["entropy_loss"]
            else:
                loss = loss_fn(x_0, x_1)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(velocity_field.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()
            num_batches += 1
        avg = epoch_loss / max(num_batches, 1)
        scheduler.step(avg)
        if epoch % 10 == 0:
            print(f"Epoch {epoch:04d}/{num_epochs}, Total: {avg:.6f}")
            if use_wandb:
                import wandb
                wandb.log({"train/total_loss": avg, "epoch": epoch,
                           "train/learning_rate": optimizer.param_groups[0]['lr']})
    return velocity_field


# ============================================
# Neural SDE : Signature MMD Loss
# ============================================

class SignatureMMDLoss(torch.nn.Module):
    """Distance relative entre signatures moyennes (moment matching simple, diagnostic)."""

    def __init__(self, depth: int = 3, time_augment: bool = True, lead_lag: bool = False):
        super().__init__()
        self.depth = depth
        self.sig_fn = TruncatedSignature(depth, time_augment=time_augment, lead_lag_transform=lead_lag)

    def forward(self, real_paths: torch.Tensor, generated_paths: torch.Tensor) -> torch.Tensor:
        mean_sig_real = self.sig_fn(real_paths).mean(dim=0)
        mean_sig_gen = self.sig_fn(generated_paths).mean(dim=0)
        raw_dist = torch.norm(mean_sig_real - mean_sig_gen, p=2)
        real_norm = torch.norm(mean_sig_real, p=2).detach().clamp(min=1e-6)
        return raw_dist / real_norm


class KernelSignatureMMDLoss(torch.nn.Module):
    """
    MMD multi-échelle sur signatures tronquées (avec canal temps par défaut).

    Noyau : k(x, y) = 1 / (1 + γ ||x - y||²), γ = 1 / (2 σ²)  — noyau de Cauchy
    (quadratique rationnel, caractéristique), PAS un RBF gaussien malgré l'ancien
    nom. Les signatures sont standardisées coefficient par coefficient avec
    (sig_mean, sig_std) calculés sur le dataset réel.

    Canal temps : sans lui la signature est invariante par reparamétrisation et la
    loss ne contraint ni la demi-vie ni le clustering de volatilité (voir
    `signature_torch`).
    """

    def __init__(
        self,
        depth: int = 3,
        sigmas: list | None = None,
        sig_scaler_mean: torch.Tensor | None = None,
        sig_scaler_std: torch.Tensor | None = None,
        time_augment: bool = True,
        lead_lag: bool = False,
    ):
        super().__init__()
        self.depth = depth
        self.sig_fn = TruncatedSignature(depth, time_augment=time_augment, lead_lag_transform=lead_lag)

        _sigmas = torch.tensor(sigmas if sigmas is not None else [1.0, 5.0, 10.0, 20.0, 40.0], dtype=torch.float32)
        self.register_buffer("sigmas", _sigmas)

        self._use_scaler = sig_scaler_mean is not None and sig_scaler_std is not None
        if self._use_scaler:
            self.register_buffer("sig_mean", sig_scaler_mean.float())
            self.register_buffer("sig_std", sig_scaler_std.float().clamp(min=1e-8))
        else:
            self.register_buffer("sig_mean", torch.zeros(1))
            self.register_buffer("sig_std", torch.ones(1))

    def _normalize(self, sig_raw: torch.Tensor) -> torch.Tensor:
        if self._use_scaler:
            return (sig_raw - self.sig_mean) / self.sig_std
        return sig_raw

    def signatures(self, paths: torch.Tensor) -> torch.Tensor:
        return self._normalize(self.sig_fn(paths))

    def forward(self, real_paths, generated_paths, precomputed_sig_real=None):
        sig_real = precomputed_sig_real if precomputed_sig_real is not None else self.signatures(real_paths)
        sig_gen = self.signatures(generated_paths)

        dist_xx = self._pairwise_sq_dist(sig_real, sig_real)
        dist_yy = self._pairwise_sq_dist(sig_gen, sig_gen)
        dist_xy = self._pairwise_sq_dist(sig_real, sig_gen)

        gammas = (1.0 / (2.0 * self.sigmas.pow(2))).view(-1, 1, 1)
        k_xx = 1.0 / (1.0 + gammas * dist_xx)
        k_yy = 1.0 / (1.0 + gammas * dist_yy)
        k_xy = 1.0 / (1.0 + gammas * dist_xy)

        mmd_per_scale = k_xx.mean(dim=(1, 2)) + k_yy.mean(dim=(1, 2)) - 2.0 * k_xy.mean(dim=(1, 2))
        return torch.clamp(mmd_per_scale.sum(), min=0.0)

    @staticmethod
    def _pairwise_sq_dist(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = x.to(torch.float32)
        y = y.to(torch.float32)
        x_sq = (x * x).sum(dim=1, keepdim=True)
        y_sq = (y * y).sum(dim=1, keepdim=True).T
        xy = x @ y.T
        return (x_sq + y_sq - 2.0 * xy).clamp(min=0.0)


# ============================================
# Fonction d'entraînement (SDE)
# ============================================

def _spread_diagnostics(real_paths: torch.Tensor, generated_paths: torch.Tensor) -> dict:
    with torch.no_grad():
        real_spread = real_paths[:, :, 0] - real_paths[:, :, 1]
        gen_spread = generated_paths[:, :, 0] - generated_paths[:, :, 1]
        return {
            "real_spread_vol": real_spread.std(dim=1).mean().item(),
            "gen_spread_vol": gen_spread.std(dim=1).mean().item(),
            "real_drift": real_spread.diff(dim=1).mean().item(),
            "gen_drift": gen_spread.diff(dim=1).mean().item(),
            "gen_path_norm": generated_paths.norm(dim=-1).mean().item(),
            "n": real_paths.shape[0],
        }


def _save_checkpoint(path: str, model: torch.nn.Module, extras: dict | None, **meta) -> None:
    payload = {"model_state_dict": model.state_dict()}
    if extras:
        payload.update(extras)
    payload.update(meta)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(payload, path)


def train_sde(
    generator_sde,
    train_loader,
    val_loader,
    sig_mean: torch.Tensor,
    sig_std: torch.Tensor,
    num_epochs: int = 1000,
    lr: float = 1e-3,
    weight_decay: float = 0.01,
    device: str = 'cuda',
    sig_depth: int = 3,
    sig_time_augment: bool = True,
    sig_lead_lag: bool = False,
    use_wandb: bool = False,
    seed: int = 42,
    accum_iter: int = 4,
    ema_decay: float = 0.999,
    early_stop_patience: int = 70,
    val_every: int = 1,
    max_batches_per_epoch: int | None = None,
    save_dir: str = "data/models",
    run_name: str | None = None,
    checkpoint_extras: dict | None = None,
    plot_every: int = 5,
    path_clamp: float = 30.0,
):
    """
    Boucle d'entraînement de la Neural SDE par Kernel Signature MMD.

    Correctifs (audit 2026-11) :
        - scheduler cosinus sur le nombre total de PAS d'optimiseur (num_epochs × steps_per_epoch),
          et non sur num_epochs (le LR touchait eta_min après ~23 époques puis remontait) ;
        - AdamW(fused=...) seulement sur CUDA ;
        - meilleur modèle sauvegardé au format checkpoint complet (config + scalers +
          conditionnement), horodaté, dans `save_dir` ;
        - seed fixée, validation tous les `val_every` époques, diagnostics agrégés sur
          toute la validation.

    Retourne (modèle EMA, chemin du meilleur checkpoint, historique).
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device(device)
    is_cuda = device.type == "cuda"
    generator_sde.to(device)
    run_name = run_name or f"neural_sde_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    best_path = os.path.join(save_dir, f"{run_name}_best.pt")

    loss_fn = KernelSignatureMMDLoss(
        depth=sig_depth,
        sigmas=[1.0, 5.0, 10.0, 20.0, 40.0],
        sig_scaler_mean=sig_mean,
        sig_scaler_std=sig_std,
        time_augment=sig_time_augment,
        lead_lag=sig_lead_lag,
    ).to(device)

    optimizer = torch.optim.AdamW(generator_sde.parameters(), lr=lr, weight_decay=weight_decay, fused=is_cuda)

    from torch.optim.swa_utils import AveragedModel
    try:
        from torch.optim.swa_utils import get_ema_multi_avg_fn
        ema_sde = AveragedModel(generator_sde, multi_avg_fn=get_ema_multi_avg_fn(ema_decay))
    except ImportError:  # torch < 2.1
        ema_sde = AveragedModel(generator_sde, avg_fn=lambda avg, p, n: ema_decay * avg + (1.0 - ema_decay) * p)

    n_batches = len(train_loader) if max_batches_per_epoch is None else min(len(train_loader), max_batches_per_epoch)
    steps_per_epoch = max(1, math.ceil(n_batches / accum_iter))
    total_steps = max(1, num_epochs * steps_per_epoch)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-7)

    best_val_loss = float('inf')
    epochs_no_improve = 0
    history = []
    ts_cache = None

    if use_wandb:
        import wandb
    if use_wandb and plot_every:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

    def _unpack(batch):
        if isinstance(batch, (list, tuple)):
            real = batch[0].to(device, non_blocking=True)
            cond = batch[1].to(device, non_blocking=True) if len(batch) > 1 else None
            pre = batch[2].to(device, non_blocking=True) if len(batch) > 2 else None
            return real, cond, pre
        return batch.to(device, non_blocking=True), None, None

    for epoch in range(num_epochs):
        generator_sde.train()
        epoch_loss, num_batches = 0.0, 0
        optimizer.zero_grad(set_to_none=True)
        last_grad_norm = 0.0
        val_image = None

        for batch_idx, batch in enumerate(train_loader):
            if batch_idx >= n_batches:
                break
            real_paths, conditions, pre_sig = _unpack(batch)
            B, seq_len, _ = real_paths.shape
            if ts_cache is None or ts_cache.shape[0] != seq_len:
                ts_cache = torch.linspace(0., float(seq_len - 1), seq_len, device=device)

            y0 = real_paths[:, 0, :]
            generated_paths = generator_sde(y0, ts_cache, conditions)
            generated_paths = torch.clamp(generated_paths, min=-path_clamp, max=path_clamp)

            loss = loss_fn(real_paths, generated_paths, pre_sig)
            (loss / accum_iter).backward()

            if ((batch_idx + 1) % accum_iter == 0) or (batch_idx + 1 == n_batches):
                last_grad_norm = float(torch.nn.utils.clip_grad_norm_(generator_sde.parameters(), max_norm=1.0))
                optimizer.step()
                scheduler.step()
                ema_sde.update_parameters(generator_sde)
                optimizer.zero_grad(set_to_none=True)

            epoch_loss += loss.item()
            num_batches += 1

        avg_train_loss = epoch_loss / max(num_batches, 1)
        record = {"epoch": epoch, "train/mmd_loss": avg_train_loss,
                  "train/learning_rate": optimizer.param_groups[0]['lr'], "train/grad_norm": last_grad_norm}

        do_val = (epoch % val_every == 0) or (epoch == num_epochs - 1)
        if do_val:
            ema_sde.module.eval()
            val_loss, val_batches = 0.0, 0
            diag_acc, diag_n = {}, 0
            ts_val_cache = None
            real_sample = gen_sample = None
            with torch.no_grad():
                for batch in val_loader:
                    real_paths_val, conditions_val, pre_val = _unpack(batch)
                    _, seq_len_val, _ = real_paths_val.shape
                    if ts_val_cache is None or ts_val_cache.shape[0] != seq_len_val:
                        ts_val_cache = torch.linspace(0.0, float(seq_len_val - 1), seq_len_val, device=device)
                    generated_val = ema_sde.module(real_paths_val[:, 0, :], ts_val_cache, conditions_val)
                    val_loss += loss_fn(real_paths_val, generated_val, pre_val).item()
                    val_batches += 1
                    d = _spread_diagnostics(real_paths_val, generated_val)
                    for k, v in d.items():
                        if k != "n":
                            diag_acc[k] = diag_acc.get(k, 0.0) + v * d["n"]
                    diag_n += d["n"]
                    if real_sample is None:
                        real_sample, gen_sample = real_paths_val[0].cpu().numpy(), generated_val[0].cpu().numpy()

            avg_val_loss = val_loss / max(val_batches, 1)
            diag = {f"diagnostics/{k}": v / max(diag_n, 1) for k, v in diag_acc.items()}
            diag["diagnostics/vol_ratio"] = diag.get("diagnostics/gen_spread_vol", 0.0) / (diag.get("diagnostics/real_spread_vol", 0.0) + 1e-8)
            record.update({"val/mmd_loss": avg_val_loss, "val/overfit_gap": avg_val_loss - avg_train_loss, **diag})

            print(f"Epoch {epoch:04d}/{num_epochs} | Train MMD: {avg_train_loss:.4f} | Val MMD: {avg_val_loss:.4f} "
                  f"| vol ratio: {diag['diagnostics/vol_ratio']:.3f} | lr: {record['train/learning_rate']:.2e}")

            if use_wandb and plot_every and epoch % plot_every == 0 and real_sample is not None:
                fig, axes = plt.subplots(1, 2, figsize=(12, 5))
                axes[0].plot(real_sample[:, 0], label='Real Asset A'); axes[0].plot(real_sample[:, 1], label='Real Asset B')
                axes[0].set_title('Validation: Scaled Log-Prices (Real)'); axes[0].legend()
                axes[1].plot(gen_sample[:, 0], '--', label='Synth Asset A'); axes[1].plot(gen_sample[:, 1], '--', label='Synth Asset B')
                axes[1].set_title('Validation: SDE Generated Path'); axes[1].legend()
                plt.tight_layout()
                val_image = wandb.Image(fig)
                plt.close(fig)

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                epochs_no_improve = 0
                _save_checkpoint(best_path, ema_sde.module, checkpoint_extras, epoch=epoch, val_loss=avg_val_loss)
            else:
                epochs_no_improve += val_every

        record["train/epochs_no_improve"] = epochs_no_improve
        history.append(record)
        if use_wandb:
            log_dict = dict(record)
            if val_image is not None:
                log_dict["val/generated_paths"] = val_image
            wandb.log(log_dict)

        if epochs_no_improve >= early_stop_patience:
            print(f"🛑 Early stopping à l'epoch {epoch} (patience={early_stop_patience}).")
            break

    if os.path.exists(best_path):
        best = torch.load(best_path, map_location=device, weights_only=False)
        ema_sde.module.load_state_dict(best["model_state_dict"])
        print(f"✅ Meilleur modèle restauré : {best_path} (val MMD={best.get('val_loss', float('nan')):.4f}, epoch {best.get('epoch')})")

    return ema_sde.module, (best_path if os.path.exists(best_path) else None), history
