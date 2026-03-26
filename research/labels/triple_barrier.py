import numpy as np
import pandas as pd

class TripleBarrierLabeler:
    """
    Labels pour chaque point d'entrée :
      +1  : upper barrier touchée en premier (profit)
      -1  : lower barrier touchée en premier (stop loss)
       0  : time stop (half-life expirée)
    
    Adapté pairs trading : barrières sur le Z-score
    """

    def __init__(
        self,
        upper_z: float = 0.5,    # exit profit (retour vers 0)
        lower_z: float = 4.0,    # stop loss
        time_factor: float = 2.0, # max_hold = time_factor * half_life
    ):
        self.upper_z = upper_z #stop loss
        self.lower_z = lower_z #profit
        self.time_factor = time_factor #half life

    def label(
        self,
        zscores: np.ndarray,
        half_life: float,
        entry_indices: np.ndarray,
        directions: np.ndarray,   # +1 long spread, -1 short spread
    ) -> pd.DataFrame:
        """
        Returns DataFrame avec colonnes : [entry_idx, label, exit_idx, duration, pnl_type]
        """
        max_hold = max(int(self.time_factor * half_life), 10) #empirically, half_life 6.1
        results  = []

        for idx, direction in zip(entry_indices, directions):
            label     = 0
            exit_idx  = min(idx + max_hold, len(zscores) - 1)
            pnl_type  = "time_stop"

            for t in range(idx + 1, exit_idx + 1):
                abs_z = abs(zscores[t])
 
                if abs_z <= self.lower_z:
                    label    = 1
                    exit_idx = t
                    pnl_type = "profit"
                    break

                if abs_z >= self.upper_z:
                    label    = -1
                    exit_idx = t
                    pnl_type = "stop_loss"
                    break

            results.append({
                "entry_idx": idx,
                "exit_idx":  exit_idx,
                "duration":  exit_idx - idx,
                "label":     label,
                "pnl_type":  pnl_type,
            })

        return pd.DataFrame(results)
