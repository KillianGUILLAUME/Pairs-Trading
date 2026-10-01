import os
import sys
import pandas as pd

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(project_root)

from research.models.meta_labeling.xgb_walkforward import XGBWalkForwardEngine
from research.backtest.walk_forward import WalkForwardConfig

def test_run():
    hmm_path = os.path.join(project_root, "data/models/hmm/btc_regime_hmm.pkl")
    engine = XGBWalkForwardEngine(hmm_path=hmm_path, wf_config=WalkForwardConfig(train_ratio=0.7))
    
    path = os.path.join(project_root, "data/storage/parquet/1h")
    df_a = pd.read_parquet(os.path.join(path, "ZEC_USDT.parquet"))
    df_b = pd.read_parquet(os.path.join(path, "XRP_USDT.parquet"))
    df_btc = pd.read_parquet(os.path.join(path, "BTC_USDT.parquet"))
    df_eth = pd.read_parquet(os.path.join(path, "ETH_USDT.parquet"))
    
    df = pd.DataFrame({
        'ZEC': df_a.set_index('timestamp')['close'],
        'XRP': df_b.set_index('timestamp')['close'],
        'BTC': df_btc.set_index('timestamp')['close'],
        'ETH': df_eth.set_index('timestamp')['close'],
    }).dropna()
    
    ts = df.index.values
    a = df['ZEC'].values
    b = df['XRP'].values
    btc = df['BTC'].values
    eth = df['ETH'].values
    
    print("🚀 Démarage du Test XGBWalkForwardEngine (Train: 70%, Test: 30%) : ZEC/XRP")
    res = engine.run_pair_with_xgb(
        timestamps=ts, price_a=a, price_b=b, 
        price_btc=btc, price_eth=eth, 
        symbol_a="ZEC", symbol_b="XRP"
    )
    
    if not res:
        print("Crash du WalkForward.")
        return
        
    m_pure = res['test_pure']['metrics']
    m_xgb = res['test_xgb']['metrics']
    
    print("\n" + "=" * 50)
    print("📊 COMPARAISON DES METRIQUES (FENETRE DE TEST OUT-OF-SAMPLE)")
    print("=" * 50)
    print(f"Trades Éxécutés : Sans XGB = {m_pure['n_trades']} | Avec XGB = {m_xgb['n_trades']}")
    print(f"Total Return Pct: Sans XGB = {m_pure['total_return_pct']}% | Avec XGB = {m_xgb['total_return_pct']}%")
    print(f"Max Drawdown    : Sans XGB = {m_pure['max_dd']}% | Avec XGB = {m_xgb['max_dd']}%")
    print(f"Sharpe Ratio    : Sans XGB = {m_pure['sharpe']} | Avec XGB = {m_xgb['sharpe']}")
    print(f"Capital Final   : Sans XGB = {m_pure['final_capital']} | Avec XGB = {m_xgb['final_capital']}")
    print("=" * 50)

if __name__ == "__main__":
    test_run()
