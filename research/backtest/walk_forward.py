"""
Walk-Forward Validation
Train/Test split temporel + métriques comparatives
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional
from loguru import logger

from research.backtest.engine import BacktestEngine, BacktestConfig
from research.signals.signal_generator import SignalGenerator, SignalGeneratorConfig

# ============================================================
# CONFIG
# ============================================================

@dataclass
class WalkForwardConfig:
    train_ratio:      float = 0.70
    n_windows:        int   = 1
    min_bars:         int   = 500


# ============================================================
# WALK-FORWARD ENGINE
# ============================================================

class WalkForwardEngine:

    def __init__(
        self,
        wf_config:       Optional[WalkForwardConfig] = None,
        backtest_config: Optional[BacktestConfig]    = None,
        sig_config:      Optional[SignalGeneratorConfig] = None,
    ):
        self.wf_cfg  = wf_config      or WalkForwardConfig()
        self.bt_cfg  = backtest_config or BacktestConfig()
        self.sig_cfg = sig_config     or SignalGeneratorConfig()

    def run_pair(
        self,
        timestamps: np.ndarray,
        price_a:    np.ndarray,
        price_b:    np.ndarray,
        symbol_a:   str,
        symbol_b:   str,
    ) -> dict:

        n          = len(timestamps)
        split_idx  = int(n * self.wf_cfg.train_ratio)

        if split_idx < self.wf_cfg.min_bars:
            logger.warning(f"{symbol_a}×{symbol_b} : pas assez de bars train ({split_idx})")
            return {}

        label = f"{symbol_a}×{symbol_b}"

        gen = SignalGenerator(
            delta_beta        = self.sig_cfg.delta_beta,
            delta_intercept   = self.sig_cfg.delta_intercept,
            obs_noise         = self.sig_cfg.obs_noise,
            zscore_window     = self.sig_cfg.zscore_window,
            entry_threshold   = self.sig_cfg.entry_threshold,
            exit_threshold    = self.sig_cfg.exit_threshold,
            use_ewm           = self.sig_cfg.use_ewm,
            compute_signature = self.sig_cfg.compute_signature,
        )
        sig_full = gen.generate(
            timestamps, price_a, price_b, symbol_a, symbol_b
        )

        results = {}
        engine  = BacktestEngine(self.bt_cfg)

        for period, (start, end) in [
            ("train", (0,         split_idx)),
            ("test",  (split_idx, n        )),
            ("full",  (0,         n        )),
        ]:
            sig_slice = sig_full.slice(start, end)

            if len(sig_slice.timestamps) < self.wf_cfg.min_bars:
                logger.warning(f"{label} [{period}] : trop peu de bars")
                continue

            res = engine.run(sig_slice, label=f"{label} [{period}]")
            results[period] = res

        return results

    def run_all(
        self,
        prices_pd:   pd.DataFrame,
        valid_pairs: list,
    ) -> pd.DataFrame:

        rows = []

        for pair in valid_pairs:
            sa, sb = pair.symbol_a, pair.symbol_b
            key    = f"{sa}×{sb}"

            res = self.run_pair(
                timestamps = prices_pd.index.values,
                price_a    = prices_pd[sa].values,
                price_b    = prices_pd[sb].values,
                symbol_a   = sa,
                symbol_b   = sb,
            )

            if not res:
                continue

            row = {"pair": key}
            for period, data in res.items():
                m = data["metrics"]
                for metric, value in m.items():
                    row[f"{period}_{metric}"] = value
            rows.append(row)

        df = pd.DataFrame(rows).set_index("pair")
        return df

# ============================================================
# ANALYSE DES RÉSULTATS
# ============================================================

def analyze_walkforward(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcule les métriques de dégradation train → test
    """
    analysis = pd.DataFrame(index=df.index)

    for metric in ["sharpe", "total_return_pct", "max_dd", "calmar"]:
        train_col = f"train_{metric}"
        test_col  = f"test_{metric}"

        if train_col in df.columns and test_col in df.columns:
            analysis[f"train_{metric}"] = df[train_col]
            analysis[f"test_{metric}"]  = df[test_col]

            # Ratio dégradation (1.0 = pas de dégradation)
            if metric != "max_dd":
                analysis[f"retention_{metric}"] = (
                    df[test_col] / df[train_col].replace(0, np.nan)
                ).round(3)

    return analysis
