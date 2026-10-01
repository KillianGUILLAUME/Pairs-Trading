import os
import sys
import numpy as np
import pandas as pd
import pickle
import xgboost as xgb
from loguru import logger
import concurrent.futures

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

from research.institutional.signal_engine import InstitutionalSignalEngine
from research.institutional.feature_master import FeatureMaster
from research.institutional.funding_estimator import FundingEstimator
from research.sizing.portfolio_manager import PortfolioManager, PortfolioConfig
from research.backtest.engine import BacktestEngine, BacktestConfig, PerformanceMetrics

def process_single_pair(pair_a: str, pair_b: str, oracle_data: dict, window: int = 156):
    """
    Exécute la pipeline institutionnelle (HJB + Veto) sur une seule paire.
    """
    data_dir = os.path.join(project_root, "data", "storage", "parquet", "1h")
    df_a = pd.read_parquet(os.path.join(data_dir, f"{pair_a}.parquet"))
    df_b = pd.read_parquet(os.path.join(data_dir, f"{pair_b}.parquet"))
    
    df = pd.merge(df_a[['timestamp', 'close']], df_b[['timestamp', 'close']], on='timestamp', suffixes=('_a', '_b'))
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    ts = df['timestamp'].values
    pa = df['close_a'].values.astype(float)
    pb = df['close_b'].values.astype(float)
    
    # Signaux HJB
    engine = InstitutionalSignalEngine(window=window, solver_type="fpt", cost=0.001)
    signal = engine.generate(ts, pa, pb)
    
    # Oracle Veto & Probs
    oracle_model = oracle_data["model"]
    oracle_features = oracle_data["features"]
    fm = FeatureMaster(signature_level=2)
    log_a, log_b = np.log(pa), np.log(pb)
    
    probs_array = np.zeros(len(ts))
    
    entries = np.where((signal.entry_long == 1) | (signal.entry_short == 1))[0]
    for idx in entries:
        h_start = max(0, idx - 127)
        if idx - h_start < 64: continue
        
        f_dict = fm.get_feature_vector(log_a[h_start:idx+1], log_b[h_start:idx+1])
        f_dict["spread_at_entry"] = signal.spreads[idx]
        f_dict["dist_to_mu"] = np.abs(signal.spreads[idx] - np.mean(signal.spreads[h_start:idx+1]))
        
        x_vec = [f_dict.get(fname, 0.0) for fname in oracle_features]
        prob = oracle_model.predict_proba(np.array([x_vec]))[0][1]
        probs_array[idx] = prob
        
    # On renvoie tout ce qu'il faut pour le backtest
    return {
        "pair": f"{pair_a}x{pair_b}",
        "signal": signal,
        "probs": probs_array,
        "ts": ts,
        "pa": pa,
        "pb": pb
    }

def run_multi_asset_institutional():
    # Chargement dynamique des paires valides
    import pandas as pd
    data_dir = os.path.join(project_root, "data", "storage", "parquet", "1h")
    available_files = [f.replace(".parquet", "") for f in os.listdir(data_dir) if f.endswith(".parquet")]
    
    screened_path = os.path.join(project_root, "data", "storage", "screened/super_dataset_SDE_128.parquet")
    df_s = pd.read_parquet(screened_path)
    all_pairs = df_s[['pair_a', 'pair_b']].drop_duplicates().values.tolist()
    
    # On ne garde que celles qui ont des fichiers data
    pairs = [(p[0], p[1]) for p in all_pairs if p[0] in available_files and p[1] in available_files]
    
    logger.info(f"🌍 UNIVERSAL INSTITUTIONAL BACKTEST | {len(pairs)} PAIRES")
    
    oracle_path = os.path.join(project_root, "data", "models", "institutional", "oracle_v2_universal.pkl")
    with open(oracle_path, "rb") as f:
        oracle_data = pickle.load(f)
        
    results = []
    # Augmentation du parallélisme pour 153 paires
    with concurrent.futures.ProcessPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(process_single_pair, pa, pb, oracle_data) for pa, pb in pairs]
        for future in concurrent.futures.as_completed(futures):
            res = future.result()
            results.append(res)
            logger.info(f"✅ Signal Engine Terminé pour {res['pair']}")

    # 2. Portfolio Management
    pm_cfg = PortfolioConfig(kelly_fraction=0.25, max_gross_exposure=2.0, max_pair_exposure=0.5)
    pm = PortfolioManager(pm_cfg)
    
    probs_dict = {r["pair"]: r["probs"] for r in results}
    # On suppose des stats de trades similaires pour l'instant (2% win / 1% loss)
    win_rets = {r["pair"]: 0.02 for r in results}
    loss_rets = {r["pair"]: 0.015 for r in results}
    
    logger.info("💼 Calcul de l'Allocation Fractional Kelly Consolidée...")
    alloc_df = pm.allocate_vectorized(probs_dict, win_rets, loss_rets)
    
    # 3. Consolidated Backtest Loop
    bt_cfg = BacktestConfig(initial_capital=100000)
    bt_engine = BacktestEngine(bt_cfg)
    fe = FundingEstimator()
    
    portfolio_pnl = np.zeros(len(results[0]["ts"]))
    count = 0
    
    for r in results:
        pair = r["pair"]
        signal = r["signal"]
        alloc_arr = alloc_df[pair].values
        
        # Filtre Veto Final (Prob < 60%)
        signal.entry_long[r["probs"] < 0.60] = 0
        signal.entry_short[r["probs"] < 0.60] = 0
        
        # Exécution du backtest avec allocations dynamiques
        bt_res = bt_engine.run(signal, allocations=alloc_arr)
        df_pnl = bt_res["df"]
        
        # Carry Costs
        fund_a = fe.estimate_funding_series(r["pa"])
        fund_b = fe.estimate_funding_series(r["pb"])
        pos = df_pnl["position"].values
        # On utilise une série de capital propre pour ne pas propager de NaN
        cap_prev = df_pnl["capital"].shift(1).fillna(bt_cfg.initial_capital).values
        carry_pct = -pos * fund_a + pos * fund_b
        
        net_pair_pnl = df_pnl["net_pnl"].values + np.nan_to_num(carry_pct * cap_prev)
        portfolio_pnl += np.nan_to_num(net_pair_pnl)
        
        logger.info(f"   ► [{pair}] Net PnL = {np.nansum(net_pair_pnl):.2f}")

    total_cap = bt_cfg.initial_capital + np.cumsum(portfolio_pnl)
    total_cap = np.nan_to_num(total_cap, nan=bt_cfg.initial_capital)
    
    # Consolidation des métriques
    final_rets = pd.Series(total_cap).pct_change().fillna(0).values
    sharpe = (final_rets.mean() / (final_rets.std() + 1e-9)) * np.sqrt(8760)
    mdd = ((total_cap - np.maximum.accumulate(total_cap)) / np.maximum.accumulate(total_cap)).min()
    
    logger.info("="*60)
    logger.info("🌍 PERFORMANCE PORTFOLIO INSTITUTIONNEL V2")
    logger.info("="*60)
    logger.info(f"Sharpe Ratio  : {sharpe:6.3f}")
    logger.info(f"Max Drawdown  : {mdd*100:6.2f}%")
    logger.info(f"Profit Final  : ${total_cap[-1]-100000:.2f}")
    logger.info(f"Exposition Max: {alloc_df.sum(axis=1).max()*100:.1f}%")
    logger.info("="*60)

if __name__ == "__main__":
    run_multi_asset_institutional()
