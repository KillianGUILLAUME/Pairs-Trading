import numpy as np
import pandas as pd
from loguru import logger
from typing import List, Dict
from dataclasses import dataclass
from research.sizing.kelly_fractionnaire import FractionalKelly

@dataclass
class PortfolioConfig:
    kelly_fraction: float = 0.5       # Fixed Kelly Fraction (Half-Kelly par défaut)
    max_gross_exposure: float = 2.0   # Plafond d'exposition globale (200%)
    max_pair_exposure: float = 0.5    # Exposition Max par paire (50%)

class PortfolioManager:
    """
    Tier-1 Institutional Portfolio Manager.
    - Agrège les signaux de N paires.
    - Vectorise l'allocation via Fractional Kelly.
    - Applique un pro-rata si Gross Exposure > max_gross_exposure.
    """
    def __init__(self, config: PortfolioConfig = PortfolioConfig()):
        self.config = config
        self.kelly = FractionalKelly(fraction=config.kelly_fraction, max_leverage=config.max_pair_exposure)
        
    def allocate_vectorized(self, pair_signals: dict, win_returns: dict, loss_returns: dict) -> pd.DataFrame:
        """
        Génère une matrice (Time x Pairs) contenant l'allocation en capital dynamique respectant le plafond.
        
        :param pair_signals: Dict[pair_name -> np.ndarray (Probabilités XGBoost ou 1.0 si Signal Pur)]
        :param win_returns: Dict[pair_name -> float] (Average Win %)
        :param loss_returns: Dict[pair_name -> float] (Average Loss %)
        :return: DataFrame des allocations finales
        """
        # 1. Calculer le Target Kelly indépendant pour chaque paire
        target_allocations = {}
        timestamps = None
        
        for pair, probs in pair_signals.items():
            w_ret = win_returns.get(pair, 0.02) # Fallback 2%
            l_ret = loss_returns.get(pair, 0.01) # Fallback 1%
            
            # Application de la formule mathématique Kelly
            allocs = self.kelly.size_array(probs, w_ret, l_ret)
            target_allocations[pair] = allocs
            
            if timestamps is None:
                timestamps = np.arange(len(probs))
                
        df_target = pd.DataFrame(target_allocations, index=timestamps).fillna(0.0)
        
        # 2. Calculer le Gross Exposure par timestep
        gross_exposure = df_target.sum(axis=1)
        
        # 3. Facteur de Scaling Pro-Rata (Clamp)
        # Si la somme des allocations dépasse le Max Gross Exposure, on réduit tout proportionnellement
        scaling_factor = np.minimum(1.0, self.config.max_gross_exposure / (gross_exposure + 1e-8))
        
        # 4. Allocation Finale
        df_final = df_target.multiply(scaling_factor, axis=0)
        
        # Logs Statistiques
        avg_gross = df_final.sum(axis=1).mean()
        max_gross = df_final.sum(axis=1).max()
        logger.info(f"📊 Portfolio Allocation Complete | Paires: {len(pair_signals)} | Avg Gross: {avg_gross:.2%} | Max Gross: {max_gross:.2%}")
        
        return df_final
