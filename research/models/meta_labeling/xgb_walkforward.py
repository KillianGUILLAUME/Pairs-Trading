import os
import sys
import pickle
import numpy as np
import pandas as pd
import xgboost as xgb
from loguru import logger
from typing import Optional

from research.backtest.walk_forward import WalkForwardEngine, WalkForwardConfig
from research.backtest.engine import BacktestEngine, BacktestConfig
from research.signals.signal_generator import SignalGenerator, SignalGeneratorConfig

from features.features_eng import get_screener_feature_dict
from research.models.hmm_regimes.inference import RegimeDetector

class XGBWalkForwardEngine(WalkForwardEngine):
    def __init__(
        self,
        hmm_path: str,
        wf_config:       Optional[WalkForwardConfig] = None,
        backtest_config: Optional[BacktestConfig]    = None,
        sig_config:      Optional[SignalGeneratorConfig] = None,
    ):
        super().__init__(wf_config, backtest_config, sig_config)
        
        try:
            self.hmm = RegimeDetector(hmm_path)
        except:
            self.hmm = None
            logger.warning("🔴 Régime HMM Oracle Introuvable. Inférence dégradée.")
            
        # Load Universal Meta-Label Model
        model_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 
            "data/models/meta_labeling/universal_xgb_oracle.pkl"
        )
        try:
            with open(model_path, 'rb') as f:
                data = pickle.load(f)
            self.model = data['model']
            self.feature_cols = data['features']
        except Exception as e:
            self.model = None
            logger.error(f"🔴 Oracle XGBoost non trouvé: {e}")

    def run_pair_with_xgb(
        self,
        timestamps: np.ndarray,
        price_a:    np.ndarray,
        price_b:    np.ndarray,
        price_btc:  np.ndarray,
        price_eth:  np.ndarray,
        symbol_a:   str,
        symbol_b:   str,
    ) -> dict:
        
        n          = len(timestamps)
        split_idx  = int(n * self.wf_cfg.train_ratio)

        if split_idx < self.wf_cfg.min_bars:
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
        sig_full = gen.generate(timestamps, price_a, price_b, symbol_a, symbol_b)

        engine = BacktestEngine(self.bt_cfg)
        sig_train = sig_full.slice(0, split_idx)
        res_train = engine.run(sig_train, label=f"{label} [train]")

        sig_test_pure = sig_full.slice(split_idx, n)
        res_test_pure = engine.run(sig_test_pure, label=f"{label} [test_pure]")

        sig_test_xgb = sig_full.slice(split_idx, n)
        
        if self.model is not None:
            e_long = np.copy(sig_test_xgb.entry_long)
            e_short = np.copy(sig_test_xgb.entry_short)
            
            signal_indices = np.where((e_long == 1) | (e_short == 1))[0]
            
            log_a = np.log(price_a.astype(float))
            log_b = np.log(price_b.astype(float))
            log_btc = np.log(price_btc.astype(float))
            log_eth = np.log(price_eth.astype(float))
            
            zscores = sig_full.zscores
            
            for idx in signal_indices:
                global_idx = split_idx + idx
                h_start = max(0, global_idx - 127)
                feat_dict = get_screener_feature_dict(
                    log_a[h_start:global_idx+1], 
                    log_b[h_start:global_idx+1], 
                    log_btc[h_start:global_idx+1], 
                    log_eth[h_start:global_idx+1]
                )
                
                feat_dict["entry_z"] = float(zscores[global_idx])
                
                if self.hmm is not None:
                    r_probs = self.hmm.predict_proba(log_btc[h_start:global_idx+1])
                    feat_dict["hmm_regime_0"] = r_probs[0]
                    feat_dict["hmm_regime_1"] = r_probs[1]
                    feat_dict["hmm_regime_2"] = r_probs[2]
                else:
                    feat_dict["hmm_regime_0"] = 0.0
                    feat_dict["hmm_regime_1"] = 0.0
                    feat_dict["hmm_regime_2"] = 0.0
                
                feat_vector = []
                for c in self.feature_cols:
                    val = feat_dict.get(c, 0.0)
                    if not np.isfinite(val): val = 0.0
                    feat_vector.append(val)
                
                x_arr = np.array(feat_vector, dtype=np.float32).reshape(1, -1)
                
                pred_prob = self.model.predict_proba(x_arr)[0][1]
                
                if pred_prob < 0.55: #TODO optimize threshold via optuna ?
                    e_long[idx] = 0
                    e_short[idx] = 0
                    
            sig_test_xgb.entry_long = e_long
            sig_test_xgb.entry_short = e_short
            
        res_test_xgb = engine.run(sig_test_xgb, label=f"{label} [test_xgb]")

        return {
            "train": res_train,
            "test_pure": res_test_pure,
            "test_xgb": res_test_xgb
        }
