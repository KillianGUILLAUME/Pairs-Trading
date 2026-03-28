import torch
import math

class Interpolant:
    """
    Classe de base pour les interpolants entre bruit source (Δ₀) et incrément cible (Δ₁).

    Dans le cadre autorégressif causal :
        - x0 = Δ₀ ~ N(0, I)                        (bruit source)
        - x1 = Δ₁ = log(P_{t+1}/P_t)  (prochain log-return, standardisé)
        - τ  ∈ [0, 1]                               (temps de diffusion, distinct du temps physique t)

    L'interpolant construit un chemin Δ_τ entre Δ₀ et Δ₁, ainsi que la
    velocity target u_τ que le réseau doit apprendre à prédire.
    """
    def __init__(self, device):
        self.device = device

    def calc_xt_ut(self, x0, x1, t):
        raise NotImplementedError
    

class LinearInterpolant(Interpolant):
    """
    Rectified Flow (interpolation linéaire déterministe).

    Chemin :   Δ_τ = (1 - τ) Δ₀ + τ Δ₁
    Velocity : u_τ = Δ₁ - Δ₀ = dΔ_τ/dτ   (constante en τ)

    Propriété : le transport optimal entre deux Gaussiennes est linéaire.
    C'est l'interpolant le plus simple et le plus stable numériquement.
    """
    def calc_xt_ut(self, x0, x1, t):
        # t est de taille (Batch, 1) — temps de diffusion τ
        # x0, x1 sont de taille (Batch, Dim) — Δ₀, Δ₁

        xt = (1 - t) * x0 + t * x1
        
        ut = x1 - x0  # dérivée par rapport à τ
        
        return xt, ut
    
class StochasticInterpolant(Interpolant):
    """
    Interpolation stochastique (ajout de bruit le long du chemin).

    Chemin :   Δ_τ = (1 - τ) Δ₀ + τ Δ₁ + σ √(τ(1 - τ)) Z,   Z ~ N(0, I)
    Velocity : u_τ = (Δ₁ - Δ₀) + σ · (1 - 2τ) / (2√(τ(1-τ))) · Z

    L'ajout de bruit régularise l'apprentissage et aide à explorer les
    modes de la distribution cible. Le paramètre σ contrôle l'intensité.
    """
    def __init__(self, device, sigma=0.1):
        super().__init__(device)
        self.sigma = sigma

    def calc_xt_ut(self, x0, x1, t):
        z = torch.randn_like(x0).to(self.device)
        
        # γ_τ = σ √(τ(1 - τ))
        term_bruit = torch.sqrt(t * (1 - t) + 1e-8)
        xt = (1 - t) * x0 + t * x1 + self.sigma * term_bruit * z
        
        # velocity : d/dτ [Δ_τ]
        # d/dτ √(τ - τ²) = (1 - 2τ) / (2√(...))
        d_gamma = (1 - 2*t) / (2 * term_bruit)
        ut = (x1 - x0) + self.sigma * d_gamma * z
        
        return xt, ut


class BrownianBridgeInterpolant(Interpolant):
    """
    Interpolant pour le Diffusion Schrödinger Bridge Matching (DSBM).

    Chemin :   Δ_τ = (1 - τ) Δ₀ + τ Δ₁ + σ √(τ(1 - τ)) Z

    Target :   u*(Δ_τ, τ | Δ₁) = (Δ₁ - Δ_τ) / (1 - τ)

    À la différence du StochasticInterpolant (target = dΔ_τ/dτ),
    ici la target est la DÉRIVE ANALYTIQUE du pont brownien conditionnel.
    C'est la formule du Bridge Matching (Proposition 2, Shi et al. NeurIPS 2023).

    Dans le cadre autorégressif :
        - Δ₀ = bruit source
        - Δ₁ = prochain incrément (log-return)
        - Le pont transporte Δ₀ → Δ₁ avec régularisation stochastique

    Ref: Shi, De Bortoli, Campbell, Doucet — NeurIPS 2023, Section 3.2.
    """

    def __init__(self, device, sigma: float = 0.5):
        super().__init__(device)
        self.sigma = sigma

    def calc_xt_ut(self, x0, x1, t):
        """
        Args:
            x0, x1 : (B, D)  — Δ₀ (bruit), Δ₁ = log(P_{t+1}/P_t) (log-return cible)
            t       : (B, 1)  — temps de diffusion τ ∈ [0, 1-ε]

        Returns:
            x_t : (B, D)  — point du pont brownien Δ_τ
            u_t : (B, D)  — dérive analytique cible (Δ₁ - Δ_τ) / (1 - τ)
        """
        z = torch.randn_like(x0).to(self.device)

        # Point du pont brownien
        std = self.sigma * torch.sqrt(t * (1.0 - t) + 1e-8)
        x_t = (1.0 - t) * x0 + t * x1 + std * z

        # Dérive analytique cible
        eps = 1e-2
        denom = torch.clamp(1.0 - t, min=eps)
        u_t = (x1 - x_t) / denom

        return x_t, u_t


class OUPriorInterpolant(Interpolant):
    """
    Interpolant Seq2Seq utilisant un processus d'Ornstein-Uhlenbeck comme prior (X_0).
    Au lieu de partir d'un bruit blanc Gaussien (Random Walk), la génération
    part d'un bruit qui a DÉJÀ des propriétés de réversion à la moyenne.
    """
    def __init__(self, device, theta=0.05, sigma=0.5, base_type="linear"):
        super().__init__(device)
        self.theta = theta
        self.sigma = sigma
        self.base_type = base_type # "linear" ou "brownian_bridge"

    def sample_x0(self, target_shape):
        """
        Génère la séquence X_0 (le prior OU) pour tout le batch.
        target_shape attendue : (Batch, Seq_len, Dim)
        """
        B, L, C = target_shape

        x0 = torch.randn(target_shape, device=self.device)
        
        # 1. Initialisation de l'état t=0 (Distribution stationnaire de l'OU)
        # Variance asymptotique d'un processus OU = sigma^2 / (2 * theta)
        asymptotic_std = self.sigma / math.sqrt(2 * self.theta) if self.theta > 0 else self.sigma

        x0[:, 0, :] *= asymptotic_std
        x0[:, 1:, :] *= self.sigma
        
        decay = 1.0 - self.theta
        for t in range(1, L):
            x0[:, t, :] += decay * x0[:, t-1, :]
            
        return x0

    def calc_xt_ut(self, x0, x1, t):
        """
        Applique le Flow Matching entre la trajectoire OU (x0) et la trajectoire Réelle (x1).
        x0, x1 : (Batch, Seq_len, Dim)
        t : (Batch, 1, 1) - Le temps de diffusion (Flow-time)
        """
        if self.base_type == "linear":
            # Transport Optimal Linéaire (Rectified Flow standard)
            xt = (1 - t) * x0 + t * x1
            ut = x1 - x0
            return xt, ut
            
        elif self.base_type == "brownian_bridge":
            # Diffusion Schrödinger Bridge Matching (DSBM)
            # Ajoute de la stochasticité PENDANT le transport
            z = torch.randn_like(x0).to(self.device)
            variance = torch.clamp(t * (1.0 - t), min=1e-8)
            std = self.sigma * torch.sqrt(variance)
            
            xt = (1.0 - t) * x0 + t * x1 + std * z
            
            eps = 1e-2
            denom = torch.clamp(1.0 - t, min=eps)
            ut = (x1 - xt) / denom
            
            return xt, ut
        else:
            raise ValueError("base_type doit être 'linear' ou 'brownian_bridge'")