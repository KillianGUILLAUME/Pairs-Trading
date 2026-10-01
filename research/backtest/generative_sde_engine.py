import torch
import torchsde
import numpy as np
import pandas as pd
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from neural_nets.neural_sde import GeneratorSDE

class GenerativeSDEEngine:
    """
    Moteur de Backtest Génératif (Architecture Ancrée - V3).
    Génère des trajectoires cointégrées avec respect absolu de la volatilité et du spread.
    Gère automatiquement le scaling global via le checkpoint.
    """
    def __init__(self, model_path: str, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        print(f"Loading Neural SDE from {model_path}...")
        
        # Utilisation de la nouvelle méthode utilitaire de chargement
        self.model, self.checkpoint = GeneratorSDE.load_from_checkpoint(model_path, device=device)
        
        self.scale_factor = self.checkpoint.get("scale_factor", 1.0)
        self.global_std = self.checkpoint.get("global_std", 1.0)
        self.scaler = self.checkpoint.get("scaler", {"mean": [0,0], "std": 1.0})
        
        self.model.eval()
        print(f"✅ Model loaded. Global Scale Factor: {self.scale_factor:.4f}")
        
    @torch.no_grad()
    def simulate_markets(self, n_paths: int = 1, bars: int = 1000, 
                         p_a_init: float = 100.0, p_b_init: float = 100.0,
                         conditions: torch.Tensor = None) -> list[pd.DataFrame]:
        """
        🚀 Simule N marchés parallèles de longueur `bars`.
        
        Args:
            n_paths: Nombre de trajectoires à générer.
            bars: Longueur de chaque trajectoire.
            p_a_init, p_b_init: Prix initiaux réels pour l'amorçage.
            conditions: Tenseur (1, C) de metrics pour le conditionnement SDE.
        """
        # 1. Préparation des conditions (expansion pour n_paths)
        if conditions is None:
            conditions = torch.zeros(1, self.model.condition_dim, device=self.device)
        
        cond_batch = conditions.to(self.device).view(1, -1).repeat(n_paths, 1)
        
        # 2. Initial State (Latent space, ancré à 0)
        y0 = torch.zeros(n_paths, self.model.data_dim, device=self.device)
        
        # 3. Génération SDE
        # generated_paths: (Batch, Seq_Len, Dim)
        generated_paths = self.model.sample(y0, bars, cond_batch)
        
        # 4. Denormalisation (Latent -> Log Diff)
        # On divise par le scale_factor global utilisé lors de l'entraînement
        log_diffs = generated_paths / self.scale_factor
        
        # 5. Reconstruction des prix réels
        log_p_a_start = np.log(p_a_init)
        log_p_b_start = np.log(p_b_init)
        
        log_prices_a = log_diffs[:, :, 0] + log_p_a_start
        log_prices_b = log_diffs[:, :, 1] + log_p_b_start
        
        prices_a = torch.exp(log_prices_a).cpu().numpy()
        prices_b = torch.exp(log_prices_b).cpu().numpy()
        
        # Jump Diffusion (Optionnel, simulation de bruit de microstructure)
        jump_prob = 0.005 
        jump_vol = 0.002
        
        market_dfs = []
        for i in range(n_paths):
            pa = prices_a[i]
            pb = prices_b[i]
            
            # Ajout de petits sauts de poisson
            jumps_a = np.random.normal(0, jump_vol, bars) * (np.random.rand(bars) < jump_prob)
            jumps_b = np.random.normal(0, jump_vol, bars) * (np.random.rand(bars) < jump_prob)
            pa *= (1 + jumps_a)
            pb *= (1 + jumps_b)
            
            df = pd.DataFrame({
                "SYNTH_A": pa,
                "SYNTH_B": pb
            })
            df.index.name = "timestamp"
            market_dfs.append(df)
            
        return market_dfs