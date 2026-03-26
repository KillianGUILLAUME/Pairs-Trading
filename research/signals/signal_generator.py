"""
Signal Generation Pipeline
- Kalman Filter pour hedge ratio dynamique
- Z-score du spread
- Path Signature features
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional
from loguru import logger
from iisignature import sig, prepare

@dataclass
class SignalGeneratorConfig:
    # Signal params
    delta_beta:      float = 1e-8   # beta quasi-fixe
    delta_intercept: float = 1e-6   # intercept très lent
    obs_noise:       float = 1e-2 
    zscore_window:    int   = 120
    entry_threshold:  float = 2.0
    exit_threshold:   float = 0.3
    use_ewm:          bool  = False
    compute_signature: bool = True 

class KalmanPairFilter:

    def __init__(
        self,
        delta_beta:      float = 1e-8,
        delta_intercept: float = 1e-6,
        obs_noise:       float = 1e-2,
    ):
        self.delta_beta      = delta_beta
        self.delta_intercept = delta_intercept
        self.obs_noise       = obs_noise
        self._reset()

    def _reset(self):
        self.theta = np.array([1.0, 0.0])   # [beta, intercept]
        self.P     = np.eye(2) * 1.0
        self.Q     = np.diag([self.delta_beta, self.delta_intercept])
        self.R     = self.obs_noise

    def update(self, obs: float, regressor: float) -> tuple[float, float, float]:
        """
        obs       = log(price_b)   [observation]
        regressor = log(price_a)   [variable explicative]
        
        Modèle : obs = beta * regressor + intercept + ε
        Returns : (spread, beta, innovation_variance_S)
        """
        # 1. Predict
        # F = [regressor, 1]  (vecteur observation)
        F = np.array([regressor, 1.0])

        theta_pred = self.theta.copy()
        P_pred     = self.P + self.Q

        # 2. Innovation
        y_pred      = F @ theta_pred
        innovation  = obs - y_pred
        S           = float(F @ P_pred @ F) + self.R   # variance innovation

        # 3. Kalman Gain
        K = (P_pred @ F) / S

        # 4. Update
        self.theta = theta_pred + K * innovation
        self.P     = (np.eye(2) - np.outer(K, F)) @ P_pred

        spread = innovation  # résidu = spread centré sur 0

        return float(spread), float(self.theta[0]), float(S)

    def fit_history(
        self,
        log_a: np.ndarray,   # regressor (log prix A)
        log_b: np.ndarray,   # observation (log prix B)
        warmup: int = 300,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        IMPORTANT : warmup = période de convergence du filtre
        On retourne les valeurs VALIDES uniquement (pas de NaN artificiels)
        → le z-score est calculé sur la série complète post-warmup
        """
        self._reset()
        n = len(log_a)

        spreads   = np.empty(n)
        betas     = np.empty(n)
        variances = np.empty(n)
        kalman_zscores = np.full(n, np.nan)

        for i in range(n):
            s, b, v      = self.update(log_b[i], log_a[i])
            spreads[i]   = s
            betas[i]     = b
            variances[i] = v
            kalman_zscores[i] = self.zscore_kalman(s, v)

        # Masque warmup APRÈS la boucle → pas de NaN dans le tableau
        # Le z-score rolling ignorera naturellement les premiers bars
        # car la fenêtre n'est pas encore pleine (min_periods)
        spreads[:warmup]   = np.nan
        betas[:warmup]     = np.nan
        variances[:warmup] = np.nan
        kalman_zscores[:warmup] = np.nan

        return spreads, betas, variances, kalman_zscores
    
    def zscore_kalman(self, innovation: float, S: float) -> float:
        """Innovation normalisée = vrai z-score Kalman"""
        return innovation / np.sqrt(S)



class SpreadZScore:
    """
    Z-score du spread avec plusieurs méthodes de normalisation.
    """

    def __init__(self, window: int = 168):  # 168h = 1 semaine
        self.window = window

    def compute(self, spreads: np.ndarray) -> np.ndarray:
        """Z-score rolling."""
        s = pd.Series(spreads)
        mu = s.rolling(self.window, min_periods=self.window // 2).mean()
        sigma = s.rolling(self.window, min_periods=self.window // 2).std()
        return ((s - mu) / sigma).values

    def compute_ewm(self, spreads: np.ndarray, halflife: float = 84.0) -> np.ndarray:
        """Z-score EWM (exponentially weighted) → plus réactif."""
        s = pd.Series(spreads)
        mu = s.ewm(halflife=halflife).mean()
        sigma = s.ewm(halflife=halflife).std()
        return ((s - mu) / sigma).values

    def zscore_kalman(self, innovation: float, S: float) -> float:
        """Innovation normalisée = vrai z-score Kalman"""
        return innovation / np.sqrt(S)



class SignatureFeatures:
    """
    Calcule la signature de chemin (path signature) sur des fenêtres glissantes.
    
    La signature capture la géométrie du chemin de manière exhaustive.
    Utile pour le ML en Phase 5 (features non-linéaires riches).
    
    Path: (time, spread, z_score) → signature de niveau `depth`
    """

    def __init__(self, window: int = 50, depth: int = 3):
        self.window = window
        self.depth = depth
        self._prep = None

    def _build_path(
        self,
        t: np.ndarray,
        spread: np.ndarray,
        zscore: np.ndarray,
    ) -> np.ndarray:
        """Construit le chemin 3D : (time, spread, zscore)."""
        # Normalise time dans [0, 1]
        t_norm = (t - t[0]) / (t[-1] - t[0] + 1e-9)
        path = np.stack([t_norm, spread, zscore], axis=1)  # (T, 3)
        return path.astype(np.float32)

    def compute_window(
        self,
        spread: np.ndarray,
        zscore: np.ndarray,
    ) -> np.ndarray:
        """
        Calcule la signature sur des fenêtres glissantes.
        Returns: array de shape (n_windows, sig_dim)
        """
        n = len(spread)
        t = np.arange(n, dtype=float)

        # Dim de la signature : sum_{k=1}^{depth} d^k où d=3
        d = 3
        sig_dim = sum(d**k for k in range(1, self.depth + 1))

        results = []
        indices = []

        for end in range(self.window, n + 1):
            start = end - self.window
            path = self._build_path(
                t[start:end],
                spread[start:end],
                zscore[start:end],
            )
            # iisignature : sig(path, depth)
            s = sig(path, self.depth)
            results.append(s)
            indices.append(end - 1)

        if not results:
            return np.empty((0, sig_dim))

        return np.array(results, dtype=np.float32)

def calibrate_kalman(log_a: np.ndarray, log_b: np.ndarray, warmup: int = 500) -> dict:
    """
    Calibration automatique des hyperparamètres Kalman pour une paire.
    Règle : R_opt = var(innovations_AR1) * 3.0
    """
    from sklearn.linear_model import LinearRegression

    # OLS sur log-prix pour beta init
    X = log_a[warmup:].reshape(-1, 1)
    y = log_b[warmup:]
    lr = LinearRegression().fit(X, y)
    beta_init     = float(lr.coef_[0])
    intercept_init = float(lr.intercept_)

    # Résidus AR1 → variance innovations
    residuals = y - lr.predict(X)
    delta_res = np.diff(residuals)
    var_innov = float(np.var(delta_res))

    # Règle calibration
    R_opt           = var_innov * 3.0
    delta_beta      = R_opt * 1e-3
    delta_intercept = R_opt * 1e-2
    P_init          = np.diag([R_opt, R_opt * 10])

    return {
        "beta_init":       beta_init,
        "intercept_init":  intercept_init,
        "obs_noise":       R_opt,
        "delta_beta":      delta_beta,
        "delta_intercept": delta_intercept,
        "P_init":          P_init,
        "var_innov":       var_innov,
    }


@dataclass
class PairSignal:
    symbol_a: str
    symbol_b: str
    timestamps: np.ndarray
    price_a: np.ndarray
    price_b: np.ndarray
    spreads: np.ndarray
    betas: np.ndarray
    zscores: np.ndarray
    kalman_zscores:  np.ndarray
    signature_features: Optional[np.ndarray] = None

    # Signaux de trading
    entry_long: np.ndarray = field(default_factory=lambda: np.array([]))   # long spread
    entry_short: np.ndarray = field(default_factory=lambda: np.array([]))  # short spread
    exit_signal: np.ndarray = field(default_factory=lambda: np.array([]))

    def to_dataframe(self) -> pd.DataFrame:
        df = pd.DataFrame({
            "timestamp": self.timestamps,
            "price_a": self.price_a,
            "price_b": self.price_b,
            "spread": self.spreads,
            "beta": self.betas,
            "zscore": self.zscores,
            "kalman_zscore": self.kalman_zscores,
            "entry_long": self.entry_long if len(self.entry_long) else np.zeros(len(self.timestamps)),
            "entry_short": self.entry_short if len(self.entry_short) else np.zeros(len(self.timestamps)),
            "exit": self.exit_signal if len(self.exit_signal) else np.zeros(len(self.timestamps)),
        })
        return df

    def slice(self, start: int, end: int) -> "PairSignal":
        """Découpe le signal sans recalculer Kalman/z-score"""
        return PairSignal(
            timestamps         = self.timestamps[start:end],
            price_a           = self.price_a[start:end],
            price_b           = self.price_b[start:end],
            spreads            = self.spreads[start:end],
            betas              = self.betas[start:end],
            zscores            = self.zscores[start:end],
            kalman_zscores     = self.kalman_zscores[start:end],
            entry_long         = self.entry_long[start:end],
            entry_short        = self.entry_short[start:end],
            exit_signal        = self.exit_signal[start:end],
            symbol_a           = self.symbol_a,
            symbol_b           = self.symbol_b,
            signature_features = (
                self.signature_features[start:end]
                if self.signature_features is not None else None
            ),
        )


class RegimeFilter:
    """
    Désactive le trading si la cointégration s'est brisée récemment.
    Évite de trader sur des paires "zombie".
    """
    def __init__(self, window: int = 500, pval_threshold: float = 0.10):
        self.window = window
        self.pval_threshold = pval_threshold

    def compute_mask(
        self,
        price_a: np.ndarray,
        price_b: np.ndarray,
        step: int = 50,  # recalcule tous les 50 bars
    ) -> np.ndarray:
        """
        Returns: mask booléen, True = cointégration active → on peut trader
        """
        n = len(price_a)
        mask = np.zeros(n, dtype=bool)

        for i in range(self.window, n):
            # Recalcule seulement tous les `step` bars (coûteux)
            if (i - self.window) % step != 0:
                mask[i] = mask[i - 1]  # conserve le dernier état
                continue
            
            try:
                _, pval, _ = coint(
                    price_a[i - self.window:i],
                    price_b[i - self.window:i],
                )
                mask[i] = pval < self.pval_threshold
            except:
                mask[i] = False

        return mask

class SignalGenerator:
    """
    Orchestration complète du signal pour une paire.
    """

    def __init__(
        self,
        delta_beta: float = 1e-8,
        delta_intercept: float = 1e-6,
        obs_noise: float = 1e-2,
        zscore_window: int = 168,
        entry_threshold: float = 2.0,
        exit_threshold: float = 0.5,
        use_ewm: bool = False,
        compute_signature: bool = True,
        sig_window: int = 50,
        sig_depth: int = 3,
    ):
        self.delta_beta = delta_beta
        self.delta_intercept = delta_intercept
        self.obs_noise = obs_noise
        self.zscore = SpreadZScore(window=zscore_window)
        self.zscore_window = zscore_window  
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold
        self.use_ewm = use_ewm
        self.compute_signature = compute_signature
        self.sig_features = SignatureFeatures(window=sig_window, depth=sig_depth) if compute_signature else None

    def _regime_filter(
        self,
        spreads:  np.ndarray,
        zscores:  np.ndarray,
        vol_window: int = 168,
        vol_mult:   float = 2.0,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Détecte les régimes de haute volatilité et désactive les signaux.
        
        Retourne :
        - regime       : 0=calme, 1=volatile (trade interdit)
        - entry_long   : signaux filtrés
        - entry_short  : signaux filtrés
        """
        s = pd.Series(spreads)
        
        # Volatilité rolling du spread
        roll_vol     = s.rolling(vol_window, min_periods=50).std()
        median_vol   = roll_vol.rolling(vol_window * 4, min_periods=vol_window).median()
        
        # Régime volatile = vol actuelle > 2x la vol médiane long terme
        is_volatile  = (roll_vol > vol_mult * median_vol).values
        
        # Z-score adaptatif : seuil plus élevé en régime volatile
        # (optionnel, plus sophistiqué)
        
        return is_volatile, roll_vol.values, median_vol.values


    def generate(
        self,
        timestamps: np.ndarray,
        price_a:    np.ndarray,
        price_b:    np.ndarray,
        symbol_a:   str = "A",
        symbol_b:   str = "B",
        use_regime_filter: bool = False,
        auto_calibrate: bool = True,
    ) -> PairSignal:

        n = len(timestamps)

        # ✅ Log-prix : transformation ici, Kalman reçoit toujours des log-prix
        log_a = np.log(price_a.astype(float))
        log_b = np.log(price_b.astype(float))

        # ✅ Ordre correct : fit_history(log_a=regressor, log_b=observation)
        # Modèle : log_b = beta * log_a + intercept + spread
        if auto_calibrate:
            calib = calibrate_kalman(log_a, log_b, warmup=500)
            logger.info(
                f"Kalman calibré | R={calib['obs_noise']:.2e} | "
                f"beta_init={calib['beta_init']:.4f}"
            )
            kf = KalmanPairFilter(
                delta_beta      = calib["delta_beta"],
                delta_intercept = calib["delta_intercept"],
                obs_noise       = calib["obs_noise"],
            )
            kf.theta = np.array([calib["beta_init"], calib["intercept_init"]])
            kf.P     = calib["P_init"]
        else:
            kf = KalmanPairFilter(
                delta_beta      = self.delta_beta,
                delta_intercept = self.delta_intercept,
                obs_noise       = self.obs_noise,
            )
        spreads, betas, variances, kalman_zscores = kf.fit_history(log_a, log_b, warmup=300)

        # ✅ Z-score sur les NaN-aware spreads
        spreads_series = pd.Series(spreads)

        if self.use_ewm:
            roll_mean = spreads_series.ewm(span=self.zscore_window, min_periods=50).mean()
            roll_std  = spreads_series.ewm(span=self.zscore_window, min_periods=50).std()
        else:
            roll_mean = spreads_series.rolling(self.zscore_window, min_periods=50).mean()
            roll_std  = spreads_series.rolling(self.zscore_window, min_periods=50).std()

        zscores = ((spreads_series - roll_mean) / roll_std.replace(0, np.nan)).values

        # ✅ Pas de signal sur NaN
        valid = ~np.isnan(zscores) & ~np.isnan(kalman_zscores)

        entry_long  = np.zeros(n)
        entry_short = np.zeros(n)
        exit_signal = np.zeros(n)

        kalman_confirms_long  = kalman_zscores < -2.0   # Kalman aussi voit spread négatif
        kalman_confirms_short = kalman_zscores >  2.0   # Kalman aussi voit spread positif

        max_zscore = 4.0

        # Regime filter optionnel
        if use_regime_filter:
            rf = RegimeFilter(window=500, pval_threshold=0.10)
            regime_mask = rf.compute_mask(price_a, price_b, step=50)
            entry_long  *= regime_mask.astype(float)
            entry_short *= regime_mask.astype(float)

        is_volatile, roll_vol, median_vol = self._regime_filter(spreads, zscores)

        # Masquer les signaux en régime volatile & Kalman
        entry_long[valid]  = (
            (zscores[valid] < -self.entry_threshold) &
            (zscores[valid] >= -max_zscore) &
            kalman_confirms_long[valid] &
            ~is_volatile[valid]
        ).astype(float)

        entry_short[valid] = (
            (zscores[valid] >  self.entry_threshold) &
            (zscores[valid] <= max_zscore) &
            kalman_confirms_short[valid] &
            ~is_volatile[valid]
        ).astype(float)

        # Exit : l'un OU l'autre suffit (conservateur)
        exit_signal[valid] = (
            (np.abs(zscores[valid]) < self.exit_threshold) |
            (np.abs(kalman_zscores[valid])  < self.exit_threshold) |
            (np.abs(zscores[valid]) > max_zscore) |
            (np.abs(kalman_zscores[valid]) > max_zscore)
        ).astype(float)


        # Signature features
        sig_feats = None
        if self.compute_signature and self.sig_features is not None:
            sig_feats = self.sig_features.compute_window(spreads, zscores)

        return PairSignal(
            symbol_a           = symbol_a,
            symbol_b           = symbol_b,
            timestamps         = timestamps,
            price_a            = price_a,
            price_b            = price_b,
            spreads            = spreads,
            betas              = betas,
            zscores            = zscores,
            kalman_zscores     = kalman_zscores,
            signature_features = sig_feats,
            entry_long         = entry_long,
            entry_short        = entry_short,
            exit_signal        = exit_signal,
        )

