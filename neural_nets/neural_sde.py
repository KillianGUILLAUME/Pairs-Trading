import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
import torchsde


def _inv_softplus(x: float) -> float:
    return math.log(math.expm1(x))


class GeneratorSDE(nn.Module):
    """
    Neural SDE pour la génération de trajectoires de paires de trading (formulation Itô).

    État observé  x = (log A, log B) ancré à 0.  Dynamique de base :
        dx = f(x, Δs, c) dt + exp(v_t) G(x, Δs, c) dW_t,         Δs = (x_A - x_B) - (x_A - x_B)(0)
    avec G triangulaire inférieure (Cholesky 2x2, corrélation apprise).

    Extensions optionnelles (toutes désactivées par défaut = comportement historique, intégré par torchsde) :

    latent_dim = 1  : volatilité stochastique. État caché v (log-multiplicateur de volatilité),
                          dv = -κ v dt + ξ dW^v,    v_0 = m(c) + s_0 ε,
                      κ, ξ, s_0 > 0 appris (softplus), m(c) linéaire en les conditions. La volatilité de x est
                      multipliée par exp(v_t) : clustering de volatilité et queues épaisses marginales.
    noise = "student" : incréments du bruit de x en Student-t à ν degrés de liberté (ν appris, ν > 2.1),
                      normalisés en variance unité, à chaque sous-pas. Queues épaisses sans clustering.
    mean_reversion  : rappel explicite du spread, drift_x += (-θ_c (Δs - m_c)) * (1/2, -1/2), de sorte que
                      d(Δs) = -θ_c (Δs - m_c) dt + ... ; θ_c = softplus(w_θ·c + b_θ) (init 0,05 par barre),
                      m_c = w_m·c. Force le retour à la moyenne et rend le conditionnement utile.

    Intégration : dès qu'une extension est active, schéma d'Euler-Maruyama maison (torchsde n'accepte pas
    de bruit non gaussien) ; sinon torchsde (anciens checkpoints). `dt` en unités de barre.
    """

    def __init__(self, data_dim=2, hidden_dim=128, target_vol=None, min_vol=1e-4, condition_dim=0,
                 dt=0.25, solver="euler", latent_dim=0, noise="gaussian", student_nu=6.0,
                 mean_reversion=False):
        """
        Args:
            data_dim      : dimension de l'état observé (2).
            hidden_dim    : largeur des MLP drift / diffusion.
            target_vol    : OBSOLÈTE, ignoré.
            min_vol       : plancher des termes diagonaux de la diffusion.
            condition_dim : nombre de descripteurs de conditionnement (features causales + spread initial).
            dt            : pas d'intégration en unités de barre (0.5 = 2 sous-pas / barre).
            solver        : schéma torchsde pour le chemin historique ("euler").
            latent_dim    : 0 ou 1 (volatilité stochastique latente).
            noise         : "gaussian" ou "student".
            student_nu    : valeur initiale de ν pour noise="student".
            mean_reversion: rappel explicite du spread conditionné par c.
        """
        super().__init__()
        if latent_dim not in (0, 1):
            raise ValueError("latent_dim doit valoir 0 ou 1")
        if noise not in ("gaussian", "student"):
            raise ValueError("noise doit être 'gaussian' ou 'student'")
        if mean_reversion and data_dim != 2:
            raise ValueError("mean_reversion suppose data_dim == 2 (spread = x_A - x_B)")
        self.data_dim = data_dim
        self.noise_type = "general"      # exigé par torchsde (chemin historique)
        self.sde_type = "ito"
        self.min_vol = min_vol
        self.condition_dim = condition_dim
        self.dt = float(dt)
        self.solver = solver
        self.latent_dim = latent_dim
        self.noise = noise
        self.mean_reversion = mean_reversion
        self.custom = bool(latent_dim > 0 or noise != "gaussian" or mean_reversion)

        # Entrée des réseaux : [x, Δs, (v), c]
        input_dim = data_dim + 1 + latent_dim + condition_dim

        self.drift_net = nn.Sequential(
            spectral_norm(nn.Linear(input_dim, hidden_dim)),
            nn.SiLU(),
            spectral_norm(nn.Linear(hidden_dim, hidden_dim)),
            nn.SiLU(),
            nn.Linear(hidden_dim, data_dim)
        )

        # Diffusion : sortie Cholesky [L11, L21, L22]
        cholesky_dim = (data_dim * (data_dim + 1)) // 2
        self.diffusion_net = nn.Sequential(
            spectral_norm(nn.Linear(input_dim, hidden_dim)),
            nn.SiLU(),
            spectral_norm(nn.Linear(hidden_dim, hidden_dim)),
            nn.SiLU(),
            nn.Linear(hidden_dim, cholesky_dim),
        )

        for m in self.drift_net:
            if isinstance(m, nn.Linear):
                w = m.weight_orig if hasattr(m, "weight_orig") else m.weight
                w.data.normal_(mean=0.0, std=1e-3)
                m.bias.data.fill_(0.)
        self.drift_net[-1].weight.data.zero_()
        self.drift_net[-1].bias.data.zero_()

        for m in self.diffusion_net:
            if isinstance(m, nn.Linear):
                w = m.weight_orig if hasattr(m, "weight_orig") else m.weight
                w.data.normal_(mean=0.0, std=1e-3)
                m.bias.data.zero_()
        self.diffusion_net[-1].weight.data.zero_()
        self.diffusion_net[-1].bias.data.fill_(0.0)
        # softplus(-3) ~ 0.05 : ancre absolue de la diffusion initiale
        self.diffusion_net[-1].bias.data[0] = -3.0   # L11
        self.diffusion_net[-1].bias.data[2] = -3.0   # L22

        # --- Extensions ---------------------------------------------------------------------
        if latent_dim:
            self.kappa_raw = nn.Parameter(torch.tensor(_inv_softplus(0.02)))     # rappel de v (par barre)
            self.xi_raw = nn.Parameter(torch.tensor(_inv_softplus(0.05)))        # vol de vol (par racine de barre)
            self.v0_scale_raw = nn.Parameter(torch.tensor(_inv_softplus(0.25)))  # écart-type initial de v
            if condition_dim:
                self.v0_mean = nn.Linear(condition_dim, 1)
                nn.init.zeros_(self.v0_mean.weight)
                nn.init.zeros_(self.v0_mean.bias)
        if noise == "student":
            self.nu_raw = nn.Parameter(torch.tensor(_inv_softplus(max(student_nu - 2.1, 0.1))))
        if mean_reversion:
            self.theta_bias = nn.Parameter(torch.tensor(_inv_softplus(0.05)))
            self.mu_bias = nn.Parameter(torch.tensor(0.0))
            if condition_dim:
                self.theta_head = nn.Linear(condition_dim, 1, bias=False)
                self.mu_head = nn.Linear(condition_dim, 1, bias=False)
                nn.init.zeros_(self.theta_head.weight)
                nn.init.zeros_(self.mu_head.weight)
            self.register_buffer("_pair_sign", torch.tensor([0.5, -0.5]))

        # Variables de session
        self.current_y0 = None
        self.current_conditions = None
        self.cached_initial_spread = None
        self.last_latent = None

    # ------------------------------------------------------------------
    # Paramètres dérivés (positifs)
    # ------------------------------------------------------------------
    @property
    def kappa(self):
        return F.softplus(self.kappa_raw)

    @property
    def xi(self):
        return F.softplus(self.xi_raw)

    @property
    def nu(self):
        return 2.1 + F.softplus(self.nu_raw)

    def _theta(self, c):
        pre = self.theta_bias + (self.theta_head(c).squeeze(-1) if self.condition_dim else 0.0)
        return F.softplus(pre)

    def _mu(self, c):
        return self.mu_bias + (self.mu_head(c).squeeze(-1) if self.condition_dim else 0.0)

    # ------------------------------------------------------------------
    # Chemin historique (torchsde) : état = x uniquement
    # ------------------------------------------------------------------
    def build_spread_and_condition(self, y):
        current_spread = y[:, 0:1] - y[:, 1:2]
        spread_deviation = current_spread - self.cached_initial_spread
        return torch.cat([y, spread_deviation, self.current_conditions], dim=-1)

    def f(self, t, y):
        y_ext = self.build_spread_and_condition(y)
        raw_drift = self.drift_net(y_ext)
        path_norm = torch.norm(y, dim=-1, keepdim=True)
        adaptive_bound = torch.sigmoid(-torch.log(path_norm.detach() / 10.0 + 1e-6))
        return raw_drift * adaptive_bound

    def _cholesky(self, out):
        L11 = torch.clamp(F.softplus(out[:, 0]), min=self.min_vol, max=5.0)
        L22 = torch.clamp(F.softplus(out[:, 2]), min=self.min_vol, max=5.0)
        L21 = torch.clamp(out[:, 1], min=-5.0, max=5.0)
        G = torch.zeros(out.shape[0], 2, 2, device=out.device, dtype=out.dtype)
        G[:, 0, 0] = L11
        G[:, 1, 0] = L21
        G[:, 1, 1] = L22
        return G

    def g(self, t, y):
        y_ext = self.build_spread_and_condition(y)
        return self._cholesky(self.diffusion_net(y_ext))

    # ------------------------------------------------------------------
    # Chemin étendu : intégrateur d'Euler-Maruyama maison
    # ------------------------------------------------------------------
    def _terms(self, y):
        """Drift (B, D+L) et matrice de diffusion (B, D+L, D+L) pour l'état complet [x, v]."""
        D = self.data_dim
        x = y[:, :D]
        c = self.current_conditions
        s_dev = x[:, 0:1] - x[:, 1:2] - self.cached_initial_spread
        parts = [x, s_dev]
        if self.latent_dim:
            v = y[:, D:D + 1]
            parts.append(v)
        parts.append(c)
        inp = torch.cat(parts, dim=-1)

        raw_drift = self.drift_net(inp)
        path_norm = torch.norm(x, dim=-1, keepdim=True)
        drift_x = raw_drift * torch.sigmoid(-torch.log(path_norm.detach() / 10.0 + 1e-6))
        if self.mean_reversion:
            force = -self._theta(c).unsqueeze(-1) * (s_dev - self._mu(c).unsqueeze(-1))     # (B, 1)
            drift_x = drift_x + force * self._pair_sign
        G = self._cholesky(self.diffusion_net(inp))

        if not self.latent_dim:
            return drift_x, G

        scale = torch.exp(torch.clamp(v, -3.0, 3.0)).unsqueeze(-1)                          # (B, 1, 1)
        B = y.shape[0]
        Gfull = torch.zeros(B, D + 1, D + 1, device=y.device, dtype=y.dtype)
        Gfull[:, :D, :D] = G * scale
        Gfull[:, D, D] = self.xi
        drift = torch.cat([drift_x, -self.kappa * v], dim=-1)
        return drift, Gfull

    def _increments(self, B, M, device, dtype):
        """Bruit unitaire (B, M) : gaussien, ou Student-t (composantes de x) à variance unité."""
        z = torch.randn(B, M, device=device, dtype=dtype)
        if self.noise == "student":
            nu = self.nu
            D = self.data_dim
            chi2 = torch.distributions.Chi2(nu.expand(B, D)).rsample()
            z_x = torch.randn(B, D, device=device, dtype=dtype) / torch.sqrt(chi2 / nu) / torch.sqrt(nu / (nu - 2.0))
            z = torch.cat([z_x, z[:, D:]], dim=-1)
        return z

    def _integrate_custom(self, y0_full, n_bars):
        n_sub = max(1, int(round(1.0 / self.dt)))
        h = 1.0 / n_sub
        sh = math.sqrt(h)
        y = y0_full
        M = y.shape[1]
        out = [y]
        for _ in range(n_bars):
            for _ in range(n_sub):
                drift, G = self._terms(y)
                z = self._increments(y.shape[0], M, y.device, y.dtype)
                y = y + drift * h + torch.bmm(G, (sh * z).unsqueeze(-1)).squeeze(-1)
            out.append(y)
        return torch.stack(out, dim=1)

    def _initial_latent(self, B, conditions, device, dtype):
        m = self.v0_mean(conditions) if self.condition_dim else 0.0
        return m + F.softplus(self.v0_scale_raw) * torch.randn(B, 1, device=device, dtype=dtype)

    # ------------------------------------------------------------------
    def forward(self, y0, ts, conditions):
        """Intègre la SDE depuis y0 (B, data_dim) aux instants ts (barres) ; retourne (B, len(ts), data_dim)."""
        B = y0.shape[0]
        if conditions is None:
            conditions = y0.new_zeros(B, 0)
        self.current_y0 = y0
        self.current_conditions = conditions
        self.cached_initial_spread = y0[:, 0:1] - y0[:, 1:2]

        if not self.custom:
            paths = torchsde.sdeint(self, y0, ts, dt=self.dt, method=self.solver)
            return paths.transpose(0, 1)

        y0_full = y0
        if self.latent_dim:
            y0_full = torch.cat([y0, self._initial_latent(B, conditions, y0.device, y0.dtype)], dim=-1)
        path = self._integrate_custom(y0_full, int(ts.shape[0]) - 1)
        self.last_latent = path[:, :, self.data_dim:].detach() if self.latent_dim else None
        return path[:, :, :self.data_dim]

    def sample(self, y0, seq_len, conditions):
        ts = torch.linspace(0.0, float(seq_len - 1), seq_len, device=y0.device)
        return self.forward(y0.float(), ts, conditions.float())

    @classmethod
    def load_from_checkpoint(cls, checkpoint_path, device="cpu"):
        """Charge un modèle et ses métadonnées de scaling avec détection dynamique."""
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if not isinstance(ckpt, dict) or "model_state_dict" not in ckpt:
            raise ValueError(
                f"{checkpoint_path} n'est pas un checkpoint complet (state_dict brut sans config/scaler) ; "
                "voir data/models/legacy/."
            )
        state_dict = ckpt["model_state_dict"]
        cfg = ckpt.get("config", {})
        model_cfg = cfg.get("model", {})

        new_state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}

        first_layer_name = "drift_net.0.weight_orig" if "drift_net.0.weight_orig" in new_state_dict else "drift_net.0.weight"
        hidden_dim, input_dim = new_state_dict[first_layer_name].shape
        data_dim = model_cfg.get("data_dim", 2)
        latent_dim = int(model_cfg.get("latent_dim", 0))
        condition_dim = input_dim - data_dim - 1 - latent_dim

        if condition_dim < 0:
            raise ValueError(
                f"Checkpoint incompatible avec l'architecture courante (input_dim={input_dim} < data_dim + 1) : "
                "ancien modèle non conditionnel, voir data/models/legacy/."
            )

        model = cls(
            data_dim=data_dim,
            hidden_dim=hidden_dim,
            condition_dim=condition_dim,
            dt=float(model_cfg.get("dt_sde", 0.25)),
            solver=model_cfg.get("sde_solver", "euler"),
            latent_dim=latent_dim,
            noise=model_cfg.get("noise", "gaussian"),
            student_nu=float(model_cfg.get("student_nu", 6.0)),
            mean_reversion=bool(model_cfg.get("mean_reversion", False)),
        )

        try:
            model.load_state_dict(new_state_dict)
        except RuntimeError as e:
            raise ValueError(
                f"{checkpoint_path} : architecture incompatible avec GeneratorSDE courant "
                f"(ex. spectral_norm sur la dernière couche) ; voir data/models/legacy/. Détail : {str(e)[:200]}"
            ) from e
        model.to(device)
        model.eval()
        return model, ckpt
