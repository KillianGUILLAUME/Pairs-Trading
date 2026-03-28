import torch
import torchsde
import numpy as np
import pandas as pd
import yaml
import sys
import os

# Inserts the neural_nets directory into sys.path to load NeuralSDE directly
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from neural_nets.neural_sde import GeneratorSDE


class GenerativeSDEEngine:
    """
    Moteur de Backtest Génératif.
    Charge une dynamique stochastique entraînée via Neural SDE et 
    Génère des trajectoires de marché de longueur infinie.
    """
    def __init__(self, model_path: str, device: str = "cpu"):
        self.device = device
        print(f"Loading Neural SDE from {model_path}...")
        
        # Load the checkpoint
        checkpoint = torch.load(model_path, map_location=device)
        self.config = checkpoint["config"]
        self.scaler = checkpoint["scaler"]
        
        # Init model
        model_cfg = self.config["model"]
        self.model = GeneratorSDE(
            data_dim=model_cfg.get("data_dim", 2),
            hidden_dim=model_cfg.get("hidden_dim", 128)
        ).to(device)
        
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()
        print("Model loaded successfully.")
        
    @torch.no_grad()
    def simulate_markets(self, n_paths: int = 1, bars: int = 10000, 
                         p_a_init: float = 100.0, p_b_init: float = 100.0) -> list[pd.DataFrame]:
        """
        Génère `n_paths` univers parallèles de longueur arbitraire (`bars`).
        
        L'avantage majeur de la SDE : on lui demande d'intégrer T=10000 directement.
        Les fonctions f(t,X) et g(t,X) maintiendront la cointegration et le risque.
        """
        print(f"🚀 Simulating {n_paths} parallel markets of length {bars} bars...")
        
        # Résolution continue via SDE : pas de "stitching" !
        generated_scaled_log_returns = self.model.generate_synthetic_paths(
            n_paths=n_paths, 
            seq_len=bars, 
            dt=1.0, 
            device=self.device
        )
        
        # Inverse Transform
        mean = torch.tensor(self.scaler["mean"], device=self.device, dtype=torch.float32)
        std = torch.tensor(self.scaler["std"], device=self.device, dtype=torch.float32)
        
        log_returns = generated_scaled_log_returns * std + mean
        
        # Cumsum pour obtenir les log-prices accumulés (Shape: Batch, L, 2)
        log_prices = torch.cumsum(log_returns, dim=1)
        
        # Exponentielle pour retrouver le prix nominal
        prices = torch.exp(log_prices).cpu().numpy()
        
        market_dfs = []
        for i in range(n_paths):
            # Application du point de départ
            price_a = prices[i, :, 0] * p_a_init
            price_b = prices[i, :, 1] * p_b_init
            
            df = pd.DataFrame({
                "timestamp": np.arange(bars),
                "SYNTH_A": price_a,
                "SYNTH_B": price_b
            }).set_index("timestamp")
            
            market_dfs.append(df)
            
        return market_dfs

# Exemple d'usage si run en standalone
if __name__ == "__main__":
    pass
