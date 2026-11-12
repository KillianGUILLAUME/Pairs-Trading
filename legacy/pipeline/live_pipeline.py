#TODO TO BE IMPLEMENTED


class LivePipeline:
    """
    Same logic as BacktestPipeline but operates on a streaming bar loop.
    Each new bar: update Kalman → check z-score → if entry candidate → RF filter → execute
    """
    
    def __init__(self, config):
        self.connector = BinanceConnector()
        self.kalman = KalmanPairFilter(**config.kalman)
        self.rf_filter = RFSignalFilter.load(config.rf_model_path)
        self.position = None
        self.capital = config.initial_capital
    
    def on_new_bar(self, price_a: float, price_b: float):
        """Called every bar (1h). Decides whether to act."""
        # 1. Update Kalman (causal – uses only this bar)
        spread, beta, S = self.kalman.update(np.log(price_b), np.log(price_a))
        zscore = spread / np.sqrt(S)
        
        # 2. Check entry/exit conditions
        if self.position is None and abs(zscore) > self.entry_threshold:
            # 3. Extract features, run RF
            features = self.extract_live_features(...)
            signal, confidence = self.rf_filter.predict(features.reshape(1, -1))
            
            if signal[0] == 1:
                direction = 1 if zscore < 0 else -1
                size = self.kelly.size(confidence[0], ...)
                self.execute_entry(direction, size)
        
        elif self.position is not None:
            # Check exit conditions
            pass
