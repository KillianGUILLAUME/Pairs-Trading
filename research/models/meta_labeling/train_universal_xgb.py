import os
import sys
import numpy as np
import pandas as pd
import xgboost as xgb
import pickle
from loguru import logger
from sklearn.model_selection import train_test_split

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(project_root)

def main():
    dataset_path = os.path.join(project_root, "data", "models", "meta_labeling", "universal_dataset.parquet")
    model_path = os.path.join(project_root, "data", "models", "meta_labeling", "universal_xgb_oracle.pkl")
    
    logger.info(f"💾 Loading Universal Dataset at: {dataset_path}")
    if not os.path.exists(dataset_path):
        logger.error("❌ Dataset not found! Run build_universal_labels.py first.")
        return
        
    df = pd.read_parquet(dataset_path)
    
    # 1. Identifier Features vs Target
    feature_cols = [
        col for col in df.columns 
        if col not in ['target_y', 'window_id', 'timestamp_start', 'timestamp_end', 'log_prices_a', 'log_prices_b', 'pair_a', 'pair_b']
    ]
    
    logger.info(f"🔎 Features ({len(feature_cols)}): {feature_cols}")
    
    X = df[feature_cols].copy()
    y = df['target_y'].copy()
    
    # 2. Nettoyage des Inf / NaN
    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    X = X.fillna(X.median())
    
    # Casting explicite float32 vital pour éviter l'explosion DMatrix C++
    X_np = X.values.astype(np.float32)
    y_np = y.values.astype(int)
    
    logger.info(f"📊 Shape: X={X_np.shape}, y={y_np.shape}")
    
    pos_count = np.sum(y_np == 1)
    neg_count = np.sum(y_np == 0)
    logger.info(f"   Distribution: {pos_count} Wins (1) | {neg_count} Loss/Timeout (0)")
    
    scale_pos = neg_count / max(pos_count, 1)
    
    X_train, X_test, y_train, y_test = train_test_split(X_np, y_np, test_size=0.15)
    
    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos,
        eval_metric='auc',
        early_stopping_rounds=30,
        n_jobs=-1
    )
    
    logger.info("🧠 Entraînement de l'Oracle en cours...")
    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_test, y_test)],
        verbose=30
    )
    
    # Metrics de base
    y_pred = model.predict(X_test)
    acc = np.mean(y_pred == y_test)
    
    logger.info(f"✅ Accuracy Out-Of-Sample : {acc:.2%}")
    logger.info(f"✅ Baseline Label : {max(pos_count, neg_count)/len(y_np):.2%}")
    
    # 4. Feature Importance
    importance = model.feature_importances_
    ranks = np.argsort(importance)[::-1]
    logger.info("\n🏆 IMPORTANCE DES FEATURES :")
    for i in range(min(5, len(ranks))):
        logger.info(f"   {i+1}. {feature_cols[ranks[i]]:<15} ({importance[ranks[i]]:.1%})")
        
    # 5. Backup
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    with open(model_path, 'wb') as f:
        pickle.dump({'model': model, 'features': feature_cols}, f)
        
    logger.info(f"✨ ORACLE UNIVERSEL SAUVEGARDÉ : {model_path}")

if __name__ == "__main__":
    main()
