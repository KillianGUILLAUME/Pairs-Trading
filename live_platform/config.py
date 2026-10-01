import os

class LiveConfig:
    # --- Exchange Settings ---
    EXCHANGE_ID = 'binance'
    API_KEY = os.getenv('BINANCE_API_KEY', '')
    API_SECRET = os.getenv('BINANCE_API_SECRET', '')
    
    # --- Portfolio Settings ---
    TARGET_PAIRS = []  # Si vide, on charge tout depuis le screener
    
    @classmethod
    def get_pairs(cls):
        if cls.TARGET_PAIRS:
            return cls.TARGET_PAIRS
        
        # Chargement dynamique depuis le dataset screened
        import pandas as pd
        path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data/storage/screened/super_dataset_SDE_128.parquet")
        if os.path.exists(path):
            df = pd.read_parquet(path)
            pairs = df[['pair_a', 'pair_b']].drop_duplicates().values.tolist()
            # On reformate pour CCXT (Slash)
            return [(p[0].replace('_', '/'), p[1].replace('_', '/')) for p in pairs]
        return [('ADA/USDT', 'AVAX/USDT')] # Fallback
    
    BASE_CURRENCY = 'USDT'
    INITIAL_CASH = 1000.0 # Pour le calcul du sizing Kelly
    
    # --- Strategy Settings ---
    HJB_WINDOW = 220
    HJB_RECALIB_FREQ = 8
    # 4. Filtre (Oracle)
    ORACLE_MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "models", "institutional", "oracle_v3_universal.pkl")
    ORACLE_THRESHOLD = 0.60
    
    # --- Risk Settings ---
    KELLY_FRACTION = 0.25
    MAX_GROSS_EXPOSURE = 1.0 # Réduit pour le début du live
    MAX_PAIR_EXPOSURE = 0.2
    
    # --- Loop Settings ---
    POLLING_INTERVAL_SEC = 3600 # 1h polling for the 1h blueprint
