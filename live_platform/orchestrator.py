import asyncio
import os
import sys
import pickle
import numpy as np
import pandas as pd
from loguru import logger

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from live_platform.config import LiveConfig as cfg

class LiveOrchestrator:
    """
    Chef d'orchestre de la plateforme live.
    Gère la boucle de décision et d'exécution.
    """
    def __init__(self, paper_trading: bool = True):
        # Oracle Fused (Ultimate V3)
        print("DEBUG: Loading Oracle...")
        oracle_path = cfg.ORACLE_MODEL_PATH
        if not os.path.exists(oracle_path):
            logger.error(f"❌ Oracle model not found at {oracle_path}")
            raise FileNotFoundError(oracle_path)
            
        with open(oracle_path, "rb") as f:
            oracle_data = pickle.load(f)
        self.oracle_model = oracle_data["model"]
        self.oracle_features = oracle_data["features"]
        print("DEBUG: Oracle Loaded.")

        # Delay imports to avoid library conflict
        from live_platform.exchange_bridge import ExchangeBridge
        from live_platform.order_manager import OrderManager
        from research.institutional.signal_engine import InstitutionalSignalEngine
        from research.institutional.feature_master import FeatureMaster
        from research.sizing.portfolio_manager import PortfolioManager, PortfolioConfig

        self.bridge = ExchangeBridge(cfg.API_KEY, cfg.API_SECRET, cfg.EXCHANGE_ID)
        self.om = OrderManager(self.bridge)
        self.paper_trading = paper_trading
        
        # Moteurs de décision
        print("DEBUG: Initializing engines...")
        self.sig_engine = InstitutionalSignalEngine(window=cfg.HJB_WINDOW, cost=0.001)
        self.fm = FeatureMaster(signature_level=2)
        
        pm_cfg = PortfolioConfig(
            kelly_fraction=cfg.KELLY_FRACTION, 
            max_gross_exposure=cfg.MAX_GROSS_EXPOSURE,
            max_pair_exposure=cfg.MAX_PAIR_EXPOSURE
        )
        print("DEBUG: Initializing Portfolio Manager...")
        self.pm = PortfolioManager(pm_cfg)
        print("DEBUG: Portfolio Manager Initialized.")

    async def run_cycle(self):
        """
        Un cycle complet de décision.
        """
        logger.info("🕒 Nouveau cycle de décision démarré...")
        
        if not self.bridge.is_connected:
            await self.bridge.connect()
            
        pair_signals = {}
        probs_dict = {}
        
        # 1. Pipeline de Signal pour chaque paire
        for (raw_a, raw_b) in cfg.get_pairs():
            # Conversion CCXT -> Filename
            sym_a = raw_a
            sym_b = raw_b
            pair_label = f"{sym_a}|{sym_b}"
            logger.info(f"🔍 Scan de la paire {pair_label}...")
            
            # Fetch OHLCV
            df_a = await self.bridge.fetch_ohlcv_dataframe(sym_a, limit=cfg.HJB_WINDOW + 20)
            df_b = await self.bridge.fetch_ohlcv_dataframe(sym_b, limit=cfg.HJB_WINDOW + 20)
            
            pa = df_a['close'].values.astype(float)
            pb = df_b['close'].values.astype(float)
            ts = df_a['timestamp'].values
            
            # Signal Engine
            signal = self.sig_engine.generate(ts, pa, pb)
            
            # Oracle Veto (uniquement sur le dernier bar)
            prob = 0.0
            if (signal.entry_long[-1] == 1) or (signal.entry_short[-1] == 1):
                log_a, log_b = np.log(pa), np.log(pb)
                f_dict = self.fm.get_feature_vector(log_a[-128:], log_b[-128:])
                f_dict["spread_at_entry"] = signal.spreads[-1]
                f_dict["dist_to_mu"] = np.abs(signal.spreads[-1] - np.mean(signal.spreads[-128:]))
                
                x_vec = [f_dict.get(fname, 0.0) for fname in self.oracle_features]
                prob = self.oracle_model.predict_proba(np.array([x_vec]))[0][1]
                
                if prob < cfg.ORACLE_THRESHOLD:
                    logger.warning(f"🚫 VETO ORACLE sur {pair_label} | Prob: {prob:.2f}")
                    signal.entry_long[-1] = 0
                    signal.entry_short[-1] = 0
                    prob = 0.0 # Veto total sur l'allocation
                else:
                    logger.success(f"🚀 SIGNAL VALIDÉ sur {pair_label} | Prob: {prob:.2f}")
            
            pair_signals[pair_label] = signal
            probs_dict[pair_label] = prob
            await asyncio.sleep(0.1) # Soft rate limit

        # 2. Portfolio Allocation
        alloc_probs = {k: np.array([v]) for k, v in probs_dict.items()}
        alloc_df = self.pm.allocate_vectorized(alloc_probs, {}, {})
        
        # 3. Exécution (si non paper trading)
        execution_log = []
        for pair_label, alloc in alloc_df.iloc[-1].items():
            if alloc > 0:
                sym_a, sym_b = pair_label.split('|')
                log_entry = f"📝 Simulation {pair_label} | Size: {alloc:.4f}"
                logger.info(log_entry)
                execution_log.append(log_entry)
                
        # 4. State Dump for Visualization
        state = {
            "timestamp": pd.Timestamp.now().isoformat(),
            "pairs_scanned": len(cfg.get_pairs()),
            "veto_count": sum(1 for p in probs_dict.values() if 0 < p < cfg.ORACLE_THRESHOLD),
            "valid_signals": sum(1 for p in probs_dict.values() if p >= cfg.ORACLE_THRESHOLD),
            "probs": {k: float(v) for k, v in probs_dict.items()},
            "allocations": alloc_df.iloc[-1].to_dict(),
            "execution_log": execution_log[-10:]
        }
        import json
        with open(os.path.join(project_root, "live_platform", "state.json"), "w") as f:
            json.dump(state, f)
        logger.info("💾 State dump updated for Live Monitor.")

    async def start(self):
        """
        Boucle infinie.
        """
        logger.info(f"🛡️ Plateforme Institutionnelle LIVE démarrée (Mode Paper: {self.paper_trading})")
        while True:
            try:
                await self.run_cycle()
                logger.info(f"😴 Veille terminée. Prochain scan dans {cfg.POLLING_INTERVAL_SEC}s")
                await asyncio.sleep(cfg.POLLING_INTERVAL_SEC)
            except Exception as e:
                logger.error(f"💥 Erreur Critique dans la boucle : {e}")
                await asyncio.sleep(60)

if __name__ == "__main__":
    orchestrator = LiveOrchestrator(paper_trading=True)
    asyncio.run(orchestrator.start())
