import os
import joblib
import numpy as np

# On gère les imports avec le bon scope
try:
    from .features import extract_hmm_features
except ImportError:
    from features import extract_hmm_features

class RegimeDetector:
    """
    Oracle d'Intelligence Artificielle fournissant les probabilités
    instantanées d'appartenir aux différents régimes de marché stochastiques.
    """
    def __init__(self, model_path: str):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Modèle HMM introuvable à : {model_path}")
            
        self.model = joblib.load(model_path)
        self.n_components = self.model.n_components
        
    def predict_proba(self, log_prices: np.ndarray) -> np.ndarray:
        """
        Analyse une séquence temporelle et extrait le champ de probabilités du dernier instant (T).
        Input:
            log_prices: (N,) minimum 25 observations temporelles passées recommandées.
        Returns:
            np.ndarray de taille (3,) contenant [P(R0), P(R1), P(R2)] pour le dernier bar.
        """
        try:
            X, _ = extract_hmm_features(log_prices)
            
            if len(X) == 0:
                # Fallback neutre uniforme
                return np.ones(self.n_components) / self.n_components
                
            # predict_proba de hmmlearn retourne P(State_i | Tous X)
            probas = self.model.predict_proba(X)
            
            # On veut l'estimation live pour le point actuel (le tout dernier log_price)
            latest_prob = probas[-1]
            
            return latest_prob
            
        except Exception as e:
            # Sécurité incassable
            return np.ones(self.n_components) / self.n_components
