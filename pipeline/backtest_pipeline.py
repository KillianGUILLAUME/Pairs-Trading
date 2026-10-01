# pipeline/backtest_pipeline.py

class BacktestPipeline:
    """
    Orchestrates: Data → Features → RF Filter → Position → PnL
    Mode: runs over historical data (real or synthetic)
    """
    
    def __init__(self, config):
        self.data_source = ...     # BaseConnector (real or SDE)
        self.signal_gen = SignalGenerator(**config.signal)
        self.feat_eng = FeatureEngineer(**config.features)
        self.rf_filter = RFSignalFilter.load(config.rf_model_path)
        self.kelly = FractionalKelly(**config.sizing)
        self.bt_engine = BacktestEngine(config.backtest)
    
    def run(self, price_a, price_b, timestamps):
        # 1. Signal generation (z-score, Kalman, etc.)
        signal = self.signal_gen.generate(timestamps, price_a, price_b, "A", "B")
        
        # 2. Find all potential entry points
        entry_mask = signal.entry_long.astype(bool) | signal.entry_short.astype(bool)
        entry_indices = np.where(entry_mask)[0]
        directions = np.where(signal.entry_long[entry_mask], 1, -1)
        
        # 3. Extract features at each entry point
        df_features = self.feat_eng.extract_features(...)
        
        # 4. RF filter: only keep high-confidence entries
        X = df_features.drop(['entry_idx', 'direction'], axis=1).values
        signals, confidences = self.rf_filter.predict(X)
        
        # 5. Apply Kelly sizing based on RF confidence
        sizes = self.kelly.size_array(confidences, win_return=..., loss_return=...)
        
        # 6. Build filtered signal arrays
        filtered_entry_long = np.zeros(len(timestamps))
        filtered_entry_short = np.zeros(len(timestamps))
        for i, (idx, direction, should_trade) in enumerate(
            zip(df_features['entry_idx'], df_features['direction'], signals)
        ):
            if should_trade:
                if direction == 1:
                    filtered_entry_long[idx] = 1
                else:
                    filtered_entry_short[idx] = 1
        
        # 7. Run backtest on filtered signals
        signal.entry_long = filtered_entry_long
        signal.entry_short = filtered_entry_short
        return self.bt_engine.run(signal)
