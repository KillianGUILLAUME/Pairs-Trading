# NEW: data/connectors/sde_connector.py
import numpy as np
import pandas as pd
import torch
from core.models.sde_generator import GenerativeSDEEngine


class SDEGenerator:
    """
    Moteur de génération de marchés synthétiques.
    Utilise un BaseConnector pour s'amorcer (seed) sur la réalité, 
    puis utilise l'IA pour générer les futurs possibles.
    """
    
    def __init__(self, model: torch.nn.Module, scaler: dict, data_source: BaseConnector, window_size: int = 60):
        self.model = model
        self.model.eval()
        self.scaler_mean = np.array(scaler["mean"])
        self.scaler_std = np.array(scaler["std"])
        self.data_source = data_source 
        self.window_size = window_size

    def generate_markets(self, symbol_a: str, symbol_b: str, timeframe: str, n_bars: int, n_paths: int = 1) -> list[pd.DataFrame]:
        """
        Génère N chemins futurs pour une paire.
        """
        logger.info(f"SDE: Amorçage avec les données réelles {symbol_a} et {symbol_b}...")
        
        # 1. On utilise le connecteur pour récupérer l'historique nécessaire à l'IA
        df_a = self.data_source.fetch_latest(symbol_a, timeframe, n_bars=self.window_size + 1)
        df_b = self.data_source.fetch_latest(symbol_b, timeframe, n_bars=self.window_size + 1)
        
        # Alignement strict des timestamps
        df_merged = pd.merge(
            df_a[['timestamp', 'close']].rename(columns={'close': symbol_a}),
            df_b[['timestamp', 'close']].rename(columns={'close': symbol_b}),
            on='timestamp', how='inner'
        )
        
        if len(df_merged) <= self.window_size:
            raise ValueError("Le connecteur n'a pas pu récupérer assez de données pour amorcer l'IA.")

        price_a = df_merged[symbol_a].values
        price_b = df_merged[symbol_b].values
        last_timestamp = df_merged['timestamp'].iloc[-1]
        
        # 2. Préparation pour PyTorch (Log-Rendements standardisés)
        log_a = np.log(price_a)
        log_b = np.log(price_b)
        
        scaled_returns = (np.stack([log_a, log_b], axis=1) - self.scaler_mean)
        
        device = next(self.model.parameters()).device
        initial_window = torch.tensor(scaled_returns, dtype=torch.float32).unsqueeze(0).to(device)
        initial_window = initial_window.repeat(n_paths, 1, 1) # Parallélisation GPU
        
        # 3. Inférence PyTorch (Génération des n_bars futurs)
        logger.info(f"SDE: Génération de {n_paths} chemins de {n_bars} barres sur GPU...")
        with torch.no_grad():
            generated_scaled = self.model.generate_trajectory(
                initial_window=initial_window,
                n_future_steps=n_bars,
                n_ode_steps=20,
                martingale=True # Projection PCFM activée
            )
            
        generated_numpy = generated_scaled.cpu().numpy()
        
        # 4. Reconstruction des prix synthétiques
        synthetic_markets = []
        time_delta = pd.Timedelta(hours=1) if timeframe == "1h" else pd.Timedelta(minutes=5) # À adapter
        synth_timestamps = [last_timestamp + time_delta * (h + 1) for h in range(n_bars)]
        
        for i in range(n_paths):
            path_returns = (generated_numpy[i] * self.scaler_std) + self.scaler_mean
            
            price_a_synth = np.exp(np.log(price_a[-1]) + np.cumsum(path_returns[:, 0]))
            price_b_synth = np.exp(np.log(price_b[-1]) + np.cumsum(path_returns[:, 1]))
            
            synthetic_markets.append(pd.DataFrame({
                "timestamp": synth_timestamps,
                symbol_a: price_a_synth,
                symbol_b: price_b_synth
            }))
            
        return synthetic_markets