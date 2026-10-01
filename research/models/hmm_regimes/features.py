import numpy as np
import pandas as pd

def extract_hmm_features(log_prices: np.ndarray, window_vol=24, window_mom=4) -> tuple:
    """
    Extrait les variables stationnaires pour l'entraînement du HMM.
    Input:
        log_prices: (N,) tableau de prix logarithmiques.
    Output:
        X_scaled: (N-window_vol, 3) matrice de features HMM.
        valid_indices: les indices conservés du tableau initial (pour faire le lien avec le prix).
    """
    if len(log_prices) < window_vol + 1:
        raise ValueError(f"Historique trop court. Requis: {window_vol + 1}")
        
    returns_1h = np.diff(log_prices)
    
    # Construction d'un DataFrame temporaire pour les calculs glissants
    df = pd.DataFrame({'ret': returns_1h})
    
    df['ret_1h'] = df['ret']
    df[f'ret_{window_mom}h'] = df['ret'].rolling(window=window_mom).sum()
    df[f'vol_{window_vol}h'] = df['ret'].rolling(window=window_vol).std()
    
    # On garde une trace des index valides
    valid_mask = df[f'vol_{window_vol}h'].notna()
    df_valid = df[valid_mask].copy()
    
    X = df_valid[[f'ret_1h', f'ret_{window_mom}h', f'vol_{window_vol}h']].values
    
    # Standardisation Z-Score robuste (Median / MAD) pour accélérer l'Espérance-Maximisation
    median = np.median(X, axis=0)
    mad = np.median(np.abs(X - median), axis=0)
    mad[mad == 0] = 1e-8
    X_scaled = (X - median) / (1.4826 * mad)
    
    # Clamping des extrêmes (Black Swans)
    X_scaled = np.clip(X_scaled, -10.0, 10.0)
    
    # L'index du dataframe + 1 donne l'index exact dans log_prices du point temporel correspondant
    valid_indices = df_valid.index.values + 1
    
    return X_scaled, valid_indices
