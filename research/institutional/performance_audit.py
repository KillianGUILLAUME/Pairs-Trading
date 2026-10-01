import os
import sys
import numpy as np
import pandas as pd
import pickle
import xgboost as xgb
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, precision_recall_curve
from loguru import logger
import matplotlib.pyplot as plt

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

def run_performance_audit():
    logger.info("🕵️ AUDIT DE PERFORMANCE GLOBALE : Oracle V2 Universel")
    
    # 1. Chargement de l'Oracle
    oracle_path = os.path.join(project_root, "data", "models", "institutional", "oracle_v2_universal.pkl")
    if not os.path.exists(oracle_path):
        logger.error("Oracle non trouvé.")
        return
        
    with open(oracle_path, "rb") as f:
        data = pickle.load(f)
        model = data["model"]
        features = data["features"]

    # 2. Re-construction d'un échantillon d'audit (On prend quelques paires variées)
    from research.institutional.universal_oracle_trainer import build_single_pair_dataset
    
    audit_pairs = [("ADA_USDT", "AVAX_USDT"), ("LINK_USDT", "SOL_USDT"), ("ETH_USDT", "BTC_USDT"), ("BNB_USDT", "NEAR_USDT")]
    
    X_test, y_test = [], []
    for pa, pb in audit_pairs:
        X, y, _ = build_single_pair_dataset(pa, pb)
        if X is not None:
            # On prend les 20% de fin pour simuler un Out-of-Sample (même si ici c'est de l'audit)
            split = int(len(X) * 0.8)
            X_test.append(X[split:])
            y_test.append(y[split:])
            
    X_test = np.concatenate(X_test)
    y_test = np.concatenate(y_test)
    
    # 3. Métriques de Précision et Recouvrement
    y_prob = model.predict_proba(X_test)[:, 1]
    threshold = 0.55
    y_pred = (y_prob >= threshold).astype(int)
    
    # 4. Calculs demandés par l'utilisateur
    total_signals = len(y_test)
    raw_winners = np.sum(y_test == 1)
    raw_losers = np.sum(y_test == 0)
    
    # Après Filtrage (Oracle)
    filtered_signals = np.sum(y_pred == 1)
    filtered_winners = np.sum((y_pred == 1) & (y_test == 1))
    
    # Missed Opportunities (Stopped by Oracle but were good)
    missed_winners = np.sum((y_pred == 0) & (y_test == 1))
    
    print("\n" + "="*55)
    print("🏆 BILAN INSTITUTIONNEL (SEUIL 0.55)")
    print("="*55)
    print(f"1. QUALITÉ DU SIGNAL (KALMAN-HJB) :")
    print(f"   - Total Signaux Bruts :    {total_signals}")
    print(f"   - Bons Trades (Win) :       {raw_winners} ({raw_winners/total_signals:.2%})")
    print(f"   - Mauvais Trades (Loss) :   {raw_losers} ({raw_losers/total_signals:.2%})")
    
    print(f"\n2. PERFORMANCE APRÈS FILTRE (ORACLE) :")
    print(f"   - Signaux Validés :         {filtered_signals}")
    print(f"   - Probabilité de Réussite : {filtered_winners/filtered_signals:.2%}" if filtered_signals > 0 else "   - Probabilité de Réussite : N/A (0 trade)")
    
    print(f"\n3. COÛT D'OPPORTUNITÉ (VETO) :")
    print(f"   - Trades gagnants stoppés : {missed_winners} ({missed_winners/raw_winners:.2%})")
    print(f"   - Libellé : L'Oracle a arrêté {missed_winners/raw_winners:.2%} des opportunités pour garantir la sécurité.")

    # 5. Feature Importance (Diagnostic)
    feat_imp = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)
    print("\n" + "="*55)
    print("🧬 FACTEURS DE DÉCISION (FEATURE IMPORTANCE)")
    print("="*55)
    print(feat_imp.head(5))

if __name__ == "__main__":
    run_performance_audit()
