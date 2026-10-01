import numpy as np
import pandas as pd
import iisignature
from loguru import logger

from research.signals.signal_generator import SignatureFeatures, SignalGenerator
from features_eng import FeatureEngineer
from research.labels.triple_barrier import TripleBarrierLabeler

def extract_hybrid_ml_dataset(markets_list, sig_cfg, label_cfg, window=50, depth=3):
    
    sig_gen = SignalGenerator(**sig_cfg)
    labeler = TripleBarrierLabeler(**label_cfg)
    
    # On instancie tes deux extracteurs de features !
    sig_extractor = SignatureFeatures(window=window, depth=depth)
    stat_engineer = FeatureEngineer(rsi_period=14, hurst_window=100, adf_window=100)
    
    X_global = []
    y_global = []
    
    for df in markets_list:
        # A. Génération du signal et des labels
        signal = sig_gen.generate(df.index.values, df["SYNTH_A"].values, df["SYNTH_B"].values, "A", "B")
        df_sig = signal.to_dataframe()
        
        trades_list = labeler.label_all(...)
        labels_df = labeler.to_dataframe(trades_list)
        
        if len(labels_df) == 0: continue
            
        # B. Extraction de tes Features Classiques (ADF, Hurst, etc.)
        entry_indices = labels_df['entry_idx'].values
        directions = labels_df['direction'].values
        
        # Ton FeatureEngineer renvoie un DataFrame parfait
        df_classic_features = stat_engineer.extract_features(
            df_prices=df, 
            sig=signal, 
            entry_indices=entry_indices, 
            directions=directions, 
            symbol_a="SYNTH_A", 
            symbol_b="SYNTH_B"
        )
        
        # C. Fusion ligne par ligne
        time_numeric = np.arange(len(df_sig), dtype=float)
        
        # On s'assure de boucler sur les features qui ont réussi à être calculées (pas de NaN)
        for _, classic_row in df_classic_features.iterrows():
            idx = int(classic_row['entry_idx'])
            
            # On retrouve le label correspondant
            match_label = labels_df[labels_df['entry_idx'] == idx]['label'].values[0]
            
            # Extraction géométrique (Path Signatures)
            t_win = time_numeric[idx - window + 1 : idx + 1]
            s_win = signal.spreads[idx - window + 1 : idx + 1]
            z_win = signal.zscores[idx - window + 1 : idx + 1]
            
            path = sig_extractor._build_path(t_win, s_win, z_win)
            signature = iisignature.sig(path, depth)
            
            # 🚀 LA FUSION MAGIQUE (On colle les stats à la suite de la signature)
            # On enlève 'entry_idx' et 'direction' des features statistiques pour ne garder que les maths
            classic_values = classic_row.drop(['entry_idx', 'direction']).values
            
            # Le vecteur X final : [sig1, sig2... sig39, hurst, adf_pvalue, rsi_diff, vol_24...]
            hybrid_x = np.concatenate([signature, classic_values])
            
            y_label = 1 if match_label == 1 else 0
            
            X_global.append(hybrid_x)
            y_global.append(y_label)

    return np.array(X_global), np.array(y_global)