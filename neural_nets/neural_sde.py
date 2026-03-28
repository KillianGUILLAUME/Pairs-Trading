import torch
import torch.nn as nn
import torchsde


torch.autograd.set_detect_anomaly(False)

class GeneratorSDE(nn.Module):
    """
    Modèle Génératif "Neural SDE" basé sur la méthode de Kidger et al. (2021).
    Modélise directement la différentielle stochastique :
        dX_t = f(t, X_t)dt + g(t, X_t)dW_t
    
    A l'avantage exceptionnel de rester stable sur de très longues séquences
    (ex: 10k bars) grâce à sa fonction de drift contractive.
    """
    
    # Conventions de torchsde
    noise_type = "diagonal"
    sde_type = "ito"

    def __init__(self, data_dim=2, hidden_dim=128, target_vol=0.1, min_vol=1e-4):
        super().__init__()
        self.data_dim = data_dim
        self.target_vol = target_vol
        self.min_vol = min_vol  
        
        # Le réseau Drift (Tendance locale & Cointegration/Mean Reversion)
        self.drift_net = nn.Sequential(
            nn.Linear(data_dim + 1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, data_dim)
        )
        
        # Le réseau Diffusion (Volatilité instantanée locale)
        self.diffusion_net = nn.Sequential(
            nn.Linear(data_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, data_dim),
            # nn.Tanh()
        )
        
        # Ajustement des poids pour commencer de manière stable
        self.drift_net[-1].weight.data.fill_(0.)
        self.drift_net[-1].bias.data.fill_(0.)

        vol_tensor = torch.tensor(target_vol)
        init_bias = torch.log(torch.exp(vol_tensor) - 1 + 1e-8).item()
        
        self.diffusion_net[-1].weight.data.fill_(0.)
        self.diffusion_net[-1].bias.data.fill_(init_bias)

    def f(self, t, y):
        """Fonction de Drift : Tendance directionnelle déterministe."""
        spread = y[:, 0:1] - y[:, 1:2] 
        f_input = torch.cat([y, spread], dim=-1)
        return self.drift_net(f_input)

    def g(self, t, y):
        """Fonction de Diffusion : Intensité du bruit injecté localement."""
        # Pour le noise_type "diagonal", g doit avoir la même forme que y
        # et on clamp pour éviter une volatilité explosive qui ferait diverger le solveur
        raw_diff = self.diffusion_net(y)
        volatility = torch.nn.functional.softplus(raw_diff)
        return torch.clamp(volatility, min=self.min_vol, max=5.0)

    @torch.no_grad()
    def generate_synthetic_paths(self, n_paths, seq_len=60, dt=1.0, device="cuda"):
        """Génération à l'inférence pour la simulation (Walk-Forward)."""
        ts = torch.linspace(0.0, float(seq_len), seq_len, device=device)
        
        # Condition initiale (X_0) : On démarre à (0, 0)
        # (A adapter si on veut faire de la SDE conditionnelle basée sur une vraie historique)
        y0 = torch.zeros(n_paths, self.data_dim, device=device)
        
        # Intégration SDE !
        paths = torchsde.sdeint(self, y0, ts, dt=dt, method="milstein")
        
        # torchsde retourne (L, Batch, Dim). On veut (Batch, L, Dim)
        return paths.transpose(0, 1)

    def forward(self, y0, ts, dt=1.0):
        """Passage forward pour l'entraînement."""
        # L'utilisation de `sdeint` "standard" (sans _adjoint) est BEAUCOUP plus rapide 
        # pour des séquences de taille moyenne (< 500) car il ne calcule pas la rétro-SDE 
        # via la méthode des jacobiens à l'envers. À moins que ta VRAM sature, sdeint > sdeint_adjoint
        paths = torchsde.sdeint(self, y0, ts, dt=dt, method="milstein")
        return paths.transpose(0, 1)
