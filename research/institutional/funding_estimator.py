import numpy as np
import pandas as pd
from loguru import logger

class FundingEstimator:
    """
    Simulateur de Funding Rates (Carry Costs) pour les actifs Crypto.
    Modélise le coût de maintien d'une position longue/courte.
    """
    def __init__(self, base_annual_rate: float = 0.10):
        self.base_rate_hourly = base_annual_rate / 8760
        
    def estimate_funding_series(self, price_series: np.ndarray, window: int = 24) -> np.ndarray:
        """
        Estime une série temporelle de funding rates.
        Logique : Le funding tend à être positif quand le marché est en "bull" (demande de long).
        On ajoute une composante de volatilité pour refléter la prime de risque.
        """
        returns = pd.Series(price_series).pct_change().fillna(0)
        
        # Composante 1 : Momentum (Bull/Bear bias)
        momentum = returns.rolling(window).mean()
        
        # Composante 2 : Volatilité (Risk Premium)
        vol = returns.rolling(window).std()
        
        # Modèle empirique : Base + Momentum_scaled + Vol_scaled
        # On scale pour arriver à des valeurs réalistes (ex: 0.01% à 0.03% par 8h)
        funding = self.base_rate_hourly + (momentum * 0.1) + (vol * 0.05)
        
        # Clip pour éviter les valeurs aberrantes (max 0.1% par heure)
        return np.clip(funding.values, -0.001, 0.001)

    def apply_carry_to_pnl(self, pnl_series: np.ndarray, position_series: np.ndarray, funding_a: np.ndarray, funding_b: np.ndarray) -> np.ndarray:
        """
        Applique le coût de financement au PnL.
        Si on est Long A / Short B :
           PnL_net = PnL_gross - (funding_a * Capital) + (funding_b * Capital)
        (On paye le funding sur le Long, on reçoit sur le Short)
        """
        # Note: position_series est 1 pour Long A/Short B, -1 pour Short A/Long B
        carry_a = -position_series * funding_a
        carry_b = position_series * funding_b
        
        total_carry = carry_a + carry_b
        return pnl_series + total_carry
