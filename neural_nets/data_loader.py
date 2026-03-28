import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import torch
from torch.utils.data import Dataset
import iisignature

def kalman_filter_spread(price_a: np.ndarray, price_b: np.ndarray, burn_in: int = 300) -> tuple:
    """
    Calcule le spread adaptatif via Filtre de Kalman itératif pour éviter le data leakage.
    On utilise les premiers `burn_in` points (via OLS) pour initialiser l'état.
    Retourne (log_a, log_b, spread) où toutes les séries ont la taille originelle.
    """
    log_a = np.log(price_a)
    log_b = np.log(price_b)
    n = len(log_a)
    
    if n <= burn_in:
        raise ValueError(f"La taille des données ({n}) doit être supérieure au burn_in ({burn_in}).")
        
    x_burn = log_b[:burn_in]
    y_burn = log_a[:burn_in]
    
    # OLS init: y = alpha + beta * x
    A = np.vstack([x_burn, np.ones(len(x_burn))]).T
    beta_init, alpha_init = np.linalg.lstsq(A, y_burn, rcond=None)[0]
    
    # Variance de l'erreur d'observation R
    residuals_burn = y_burn - (beta_init * x_burn + alpha_init)
    var_res = np.var(residuals_burn)
    R = var_res if var_res > 1e-8 else 1e-4
    
    # Matrice de covariance d'état Q (volatilité de beta et alpha)
    # Plus Q est faible, plus l'adaptation est lente et stable.
    Q = np.array([[1e-6, 0], [0, 1e-6]])
    
    # Matrice de covariance initiale P
    P = np.array([[1e-3, 0], [0, 1e-3]])
    
    # État initial [beta, alpha]
    theta = np.array([beta_init, alpha_init])
    
    spread = np.zeros(n)
    spread[:burn_in] = residuals_burn
    
    # Filtrage causal
    for t in range(burn_in, n):
        x_t = log_b[t]
        y_t = log_a[t]
        H_t = np.array([x_t, 1.0])
        
        # Predict
        P_pred = P + Q
        y_pred = np.dot(H_t, theta)
        
        # Innovation
        e_t = y_t - y_pred
        
        # Update
        S_t = np.dot(H_t, np.dot(P_pred, H_t.T)) + R
        K_t = np.dot(P_pred, H_t.T) / S_t
        
        theta = theta + K_t * e_t
        P = P_pred - np.outer(K_t, np.dot(H_t, P_pred))
        
        # Le spread est l'innovation (erreur de prédiction ex-ante)
        spread[t] = e_t
        
    return log_a, log_b, spread

class CryptoPairsDataset(Dataset):
    """
    Dataset pour la génération Seq2Seq.
    Fournit les logs-prix et un Spread dynamique (par Filtre de Kalman)
    sans lookahead bias.
    """
    def __init__(self, price_a, price_b, seq_len=60, sig_depth=3, compute_signature=False, mean=None, std=None, burn_in=300):
        self.seq_len = seq_len
        self.sig_depth = sig_depth
        self.compute_signature = compute_signature
        
        # 1. Calcul des features (dont Spread filtré conditionnellement au passé)
        log_a, log_b, spread = kalman_filter_spread(price_a, price_b, burn_in=burn_in)
        
        # On ignore la période de burn_in
        log_a = log_a[burn_in:]
        log_b = log_b[burn_in:]
        spread = spread[burn_in:]
        
        # On a 3 features: [log_pA, log_pB, Spread]
        self.raw_features = np.stack([log_a, log_b, spread], axis=1)
        
        # 2. Standardisation (centrage/réduction) proportionnelle
        if mean is not None and std is not None:
            self.mean = mean
            self.std = std
        else:
            self.mean = np.mean(self.raw_features, axis=0)
            self.std = np.std(self.raw_features, axis=0)
            
        safe_std = np.where(self.std == 0, 1e-8, self.std)
        self.scaled_features = (self.raw_features - self.mean) / safe_std
        
        # 3. Pré-calcul optionnel des signatures
        if self.compute_signature:
            print(f"Pré-calcul des signatures de chemins (profondeur {sig_depth})...")
            self.signatures = self._compute_all_signatures()
        else:
            self.signatures = None

    def _compute_all_signatures(self):
        n_samples = len(self.scaled_features) - self.seq_len

        windows = sliding_window_view(self.scaled_features, self.seq_len, axis=0)
        windows = np.swapaxes(windows, 1, 2)[:n_samples]

        time_col = np.linspace(0, 1, self.seq_len).reshape(1, self.seq_len, 1)
        time_cols = np.repeat(time_col, n_samples, axis=0)

        paths_with_time = np.concatenate([time_cols, windows], axis=2)
        sigs = iisignature.sig(paths_with_time, self.sig_depth)
        
        return np.array(sigs, dtype=np.float32)

    def __len__(self):
        return len(self.scaled_features) - self.seq_len

    def __getitem__(self, idx):
        target_seq = self.scaled_features[idx : idx + self.seq_len]
        target_seq_tensor = torch.tensor(target_seq, dtype=torch.float32)
        
        if self.compute_signature and self.signatures is not None:
            sig_tensor = torch.tensor(self.signatures[idx], dtype=torch.float32)
            return target_seq_tensor, sig_tensor
            
        return target_seq_tensor