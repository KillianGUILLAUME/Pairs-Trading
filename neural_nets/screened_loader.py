"""
DataLoader qui charge les fenêtres pré-screened depuis le fichier Parquet 
généré par research/pairs/screener.py. Chaque fenêtre a déjà ses métriques (ADF, Hurst, Half-Life)
et peut être filtrée en amont.
"""

import numpy as np
import torch
from torch.utils.data import Dataset
import pandas as pd


class ScreenedPairsDataset(Dataset):
    """
    Dataset pour l'entraînement de la Neural SDE à partir de fenêtres pré-screened.
    
    Args:
        parquet_path: Chemin vers le fichier .parquet généré par screener.py
        max_adf_pvalue: Filtre les fenêtres non-stationnaires (défaut: garder tout)
        max_hurst: Filtre les fenêtres non mean-reverting (défaut: garder tout)
        max_half_life: Filtre les fenêtres avec un retour trop lent (défaut: garder tout)
        mean: Moyenne de standardisation des prix (si None, calculée sur le dataset)
        std: Écart-type de standardisation des prix (si None, calculé sur le dataset)
        sig_depth: Profondeur de la signature (défaut: 3)
    """

    def __init__(
        self,
        parquet_path: str,
        max_adf_pvalue: float = 1.0,
        max_hurst: float = 1.0,
        max_half_life: float = float("inf"),
        mean: np.ndarray = None,
        std: np.ndarray = None,
        sig_depth: int = 3
    ):
        print(f"Chargement du dataset screened : {parquet_path}")
        import iisignature
        df = pd.read_parquet(parquet_path)
        
        n_total = len(df)
        
        # 1. Filtrage par qualité
        mask = (
            (df["adf_pvalue"] <= max_adf_pvalue) &
            (df["hurst"] <= max_hurst) &
            (df["half_life"] <= max_half_life)
        )
        df = df[mask].reset_index(drop=True)
        n_filtered = len(df)
        
        print(f"   {n_filtered}/{n_total} fenêtres retenues "
              f"(ADF≤{max_adf_pvalue}, H≤{max_hurst}, HL≤{max_half_life})")
        
        # 2. Extraction des log-prix
        self.log_prices_a = np.array(df["log_prices_a"].tolist(), dtype=np.float32) 
        self.log_prices_b = np.array(df["log_prices_b"].tolist(), dtype=np.float32) 
        self.seq_len = self.log_prices_a.shape[1]
        
        # 3. Extraction et SCALING des descripteurs de conditionnement
        ignore_cols = ["window_id", "timestamp", "timestamp_start", "timestamp_end", "log_prices_a", "log_prices_b"]
        numeric_cols = df.select_dtypes(include=["number"]).columns
        condition_cols = [c for c in numeric_cols if c not in ignore_cols]
        
        if "half_life" in df.columns:
            df["half_life"] = df["half_life"].clip(upper=500.0)
            
        df[condition_cols] = df[condition_cols].fillna(0.0)
        
        raw_metrics = df[condition_cols].to_numpy(dtype=np.float32)
        # Scaling Z-score pour les conditions (Vital pour éviter l'explosion du drift)
        self.metrics_mean = np.mean(raw_metrics, axis=0)
        self.metrics_std = np.std(raw_metrics, axis=0)
        safe_metrics_std = np.where(self.metrics_std == 0, 1e-8, self.metrics_std)
        scaled_metrics = (raw_metrics - self.metrics_mean) / safe_metrics_std
        self.metrics = torch.tensor(scaled_metrics, dtype=torch.float32)
        self.condition_dim = self.metrics.shape[1]
        
        # 4. Stack en features [log_pA, log_pB]
        self.raw_features = np.stack([self.log_prices_a, self.log_prices_b], axis=-1)
        
        # 5. Standardisation (Centrage Global)
        all_points = self.raw_features.reshape(-1, 2)
        if mean is not None:
            self.mean = mean
        else:
            self.mean = np.mean(all_points, axis=0)
        
        centered_features = self.raw_features - self.mean
        
        # ---------------------------------------------------------
        # ZERO-ANCHORING & SCALING GLOBAL (True Global Fix)
        # ---------------------------------------------------------
        # 1. Capture du point de départ
        initial_points = centered_features[:, 0:1, :]
        
        # 2. Application du Zero-Anchoring
        anchored_paths = centered_features - initial_points
        
        # 3. Calcul de l'échelle globale UNIQUE
        if std is not None:
            self.global_std = float(std)
        else:
            self.global_std = float(np.std(anchored_paths))
            
        self.scale_factor = 1.0 / (self.global_std + 1e-8)
        self.scaled_features = anchored_paths * self.scale_factor
        
        # 4. Injection du spread initial (SCALÉ !) dans les conditions
        # On utilise le point de départ DÉJÀ CENTRE mais PAS ENCORE ANCHORÉ
        # mais on doit le scaler avec le MEME facteur
        initial_spreads = (initial_points[:, 0, 0] - initial_points[:, 0, 1]) * self.scale_factor
        initial_spreads_tensor = torch.tensor(initial_spreads, dtype=torch.float32).unsqueeze(1)
        self.metrics = torch.cat([self.metrics, initial_spreads_tensor], dim=1)
        self.condition_dim = self.metrics.shape[1]
        
        self.std = self.global_std 
        print(f"   ⚖️ SCALING GLOBAL APPLIQUÉ : Std={self.global_std:.6f} | Factor={self.scale_factor:.2f}")
        
        # ---------------------------------------------------------
        # Pré-calcul statique des signatures
        print(f"🚀 Pré-calcul massif des {len(self.scaled_features)} signatures (Depth={sig_depth})...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        rf_tensor = torch.tensor(self.scaled_features, dtype=torch.float32).to(device)
        self.signatures = []
        
        chunk_sz = 8192
        with torch.no_grad():
            for i in range(0, len(rf_tensor), chunk_sz):
                chunk_np = rf_tensor[i:i+chunk_sz].cpu().numpy()
                s_np = iisignature.sig(chunk_np, sig_depth)
                self.signatures.append(torch.from_numpy(s_np.astype(np.float32)))
                
        self.signatures = torch.cat(self.signatures, dim=0)
        self.sig_mean = self.signatures.mean(dim=0)
        self.sig_std = self.signatures.std(dim=0).clamp(min=1e-8)
        self.signatures = (self.signatures - self.sig_mean) / self.sig_std
        print(f"✅ Signatures MMD pré-calculées avec succès : {self.signatures.shape}")

    def __len__(self):
        return len(self.scaled_features)

    def __getitem__(self, idx):
        path = torch.tensor(self.scaled_features[idx], dtype=torch.float32)
        return path, self.metrics[idx], self.signatures[idx]
