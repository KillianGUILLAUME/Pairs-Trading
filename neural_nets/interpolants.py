import torch

class Interpolant:
    """
    Classe de base pour gérer la trajectoire x_t entre le bruit (x0) et la data (x1).
    """
    def __init__(self, device):
        self.device = device

    def calc_xt_ut(self, x0, x1, t):
        raise NotImplementedError
    

class LinearInterpolant(Interpolant):
    """
    Rectified Flow
    Coefficients : alpha = 1-t, beta = t, gamma = 0
    """
    def calc_xt_ut(self, x0, x1, t):
        # t est de taille (Batch, 1)
        # x0, x1 sont de taille (Batch, Dim)

        xt = (1 - t) * x0 + t * x1
        
        ut = x1 - x0 #derivatitve with respect to time
        
        return xt, ut
    
class StochasticInterpolant(Interpolant):
    """$
    Trajectoire : Pont Brownien.
    """
    def __init__(self, device, sigma=0.1):
        super().__init__(device)
        self.sigma = sigma

    def calc_xt_ut(self, x0, x1, t):
        z = torch.randn_like(x0).to(self.device)
        
        # Gamma_t = sigma * sqrt(t(1-t))
        term_bruit = torch.sqrt(t * (1 - t) + 1e-8)
        xt = (1 - t) * x0 + t * x1 + self.sigma * term_bruit * z
        
        # velocity (derivatitve with respect to time)
        # d/dt sqrt(t - t^2) = (1 - 2t) / (2 * sqrt(...))
        d_gamma = (1 - 2*t) / (2 * term_bruit)
        ut = (x1 - x0) + self.sigma * d_gamma * z
        
        return xt, ut


class BrownianBridgeInterpolant(Interpolant):
    """
    Interpolant pour le Diffusion Schrödinger Bridge Matching (DSBM).

    Processus de référence : Brownian Bridge entre (x₀, x₁).
        x_t = (1-t) x₀ + t x₁ + σ √(t(1-t)) z

    À la différence du StochasticInterpolant (dont la target est ẋ_t, dérivée de x_t),
    ici la target est la DÉRIVE ANALYTIQUE du pont brownien conditionnel :
        u*(x_t, t | x₁) = (x₁ - x_t) / (1 - t)

    C'est la formule utilisée dans la Bridge Matching Loss (Proposition 2, NeurIPS 2023).

    Référence : Shi, De Bortoli, Campbell, Doucet — NeurIPS 2023, Section 3.2.
    """

    def __init__(self, device, sigma: float = 0.5):
        super().__init__(device)
        self.sigma = sigma

    def calc_xt_ut(self, x0, x1, t):
        """
        Args:
            x0, x1 : (B, D)  — extrémités du pont
            t       : (B, 1)  — temps dans [0, 1-ε]

        Returns:
            x_t : (B, D)  — point du pont brownien conditionnel
            u_t : (B, D)  — dérive analytique cible (x₁ - x_t) / (1 - t)
        """
        z = torch.randn_like(x0).to(self.device)

        # Point du pont brownien conditionnel
        std = self.sigma * torch.sqrt(t * (1.0 - t) + 1e-8)
        x_t = (1.0 - t) * x0 + t * x1 + std * z

        # Dérive analytique cible (drift du pont conditionnel)
        eps = 1e-3
        denom = torch.clamp(1.0 - t, min=eps)
        u_t = (x1 - x_t) / denom

        return x_t, u_t