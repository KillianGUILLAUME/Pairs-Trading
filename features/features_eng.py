import numpy as np
import pandas as pd
import warnings
from statsmodels.tsa.stattools import adfuller


warnings.filterwarnings("ignore", message="divide by zero encountered in divide")

class FeatureEngineer:
    def __init__(self, rsi_period: int = 14, hurst_window: int = 300, adf_window: int = 300):
        self.rsi_period = rsi_period
        self.hurst_window = hurst_window
        self.adf_window = adf_window

    def _compute_rsi(self, series: pd.Series, period: int) -> pd.Series:
        """Calcule le RSI de base."""
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def _compute_hurst(self, ts: np.ndarray) -> float:
        """
        Calcule l'Exposant de Hurst (H).
        H < 0.5 : Retour à la moyenne 
        H = 0.5 : Marche aléatoire 
        H > 0.5 : Tendance 
        """
        lags = range(2, 20)
        # Calcul de la variance des différences pour plusieurs lags
        tau = [np.sqrt(np.std(np.subtract(ts[lag:], ts[:-lag]))) for lag in lags]
        # Régression linéaire sur l'échelle log-log
        poly = np.polyfit(np.log(lags), np.log(tau), 1)
        return poly[0] * 2.0

    def extract_features(
        self, 
        df_prices: pd.DataFrame, 
        sig,               # SignalGenerator (sig.spreads, sig.betas)
        entry_indices: np.ndarray, 
        directions: np.ndarray,
        symbol_a: str,
        symbol_b: str
    ) -> pd.DataFrame:
        """
        Prend une "photo" des conditions du marché à chaque point d'entrée.
        """
        features = []
        
        # 1. Calcul des Features "légères" vectorisées sur toute la série
        price_a = df_prices[symbol_a]
        price_b = df_prices[symbol_b]
        
        rsi_a = self._compute_rsi(price_a, self.rsi_period)
        rsi_b = self._compute_rsi(price_b, self.rsi_period)
        rsi_spread = rsi_a - rsi_b  # Différentiel de momentum
        
        spreads_series = pd.Series(sig.spreads)
        vol_24h = spreads_series.rolling(24).std()
        vol_72h = spreads_series.rolling(72).std()
        
        betas_series = pd.Series(sig.betas)
        beta_momentum = betas_series.diff(24) # Vitesse de changement du Beta sur 24h

        # 2. Boucle uniquement sur les points d'intervention (Optimisation CPU)
        for idx, direction in zip(entry_indices, directions):
            # Sécurité : pas assez d'historique pour calculer les features complexes
            if idx < max(self.hurst_window, self.adf_window):
                continue
                
            # Extraction de la fenêtre historique (Strictement AVANT l'entrée pour éviter le Data Leakage)
            hist_spread = sig.spreads[idx - self.hurst_window : idx]
            hist_adf    = sig.spreads[idx - self.adf_window : idx]
            
            # --- FEATURE 1 : Exposant de Hurst ---
            hurst_val = self._compute_hurst(hist_spread)
            
            hist_adf_clean = pd.Series(hist_adf).dropna().values
            
            # 2. On vérifie qu'on a assez de données et que la variance n'est pas nulle
            if len(hist_adf_clean) > 30 and np.std(hist_adf_clean) > 1e-8:
                try:
                    # autolag='AIC' laisse la fonction choisir le meilleur paramètre mathématique
                    _, adf_pvalue, _, _, _, _ = adfuller(hist_adf_clean, autolag='AIC')
                except Exception as e:
                    # Si ça plante quand même (matrice singulière, etc.), on met 1.0
                    adf_pvalue = 1.0 
            else:
                # Pas assez de données ou ligne plate = pas de co-intégration prouvable
                adf_pvalue = 1.0
                
            # --- CONSTITUTION DE LA LIGNE (X) ---
            row = {
                "entry_idx": idx,
                "direction": direction, # +1 (Long Spread) ou -1 (Short Spread)
                "hurst_exponent": hurst_val,
                "adf_pvalue": adf_pvalue,
                "rsi_differential": rsi_spread.iloc[idx],
                "spread_vol_24h": vol_24h.iloc[idx],
                "spread_vol_72h": vol_72h.iloc[idx],
                "vol_ratio": vol_24h.iloc[idx] / vol_72h.iloc[idx] if vol_72h.iloc[idx] > 0 else 1.0,
                "beta_momentum": beta_momentum.iloc[idx],
                "zscore_at_entry": sig.zscores[idx] # Est-on rentré violemment (-4) ou doucement (-2) ?
            }
            
            features.append(row)
            
        df_features = pd.DataFrame(features)
        
        # On supprime les éventuels NaN créés par les calculs roulants
        df_features = df_features.dropna().reset_index(drop=True)
        return df_features