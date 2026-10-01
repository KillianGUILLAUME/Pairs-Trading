import numpy as np
import pandas as pd
import statsmodels.api as sm
from loguru import logger
from research.institutional.hjb_solver_fpt import HJBSolverFPT
from research.institutional.hjb_solver_dp import HJBSolverDP
from dataclasses import dataclass

@dataclass
class InstitutionalSignal:
    timestamps: np.ndarray
    price_a: np.ndarray
    price_b: np.ndarray
    spreads: np.ndarray
    zscores: np.ndarray
    betas: np.ndarray
    entry_long: np.ndarray
    entry_short: np.ndarray
    exit_signal: np.ndarray
    b_star: np.ndarray
    d_star: np.ndarray

class InstitutionalSignalEngine:
    """
    Moteur de génération de signaux V2 (SDE & HJB).
    """
    def __init__(self, window: int = 156, solver_type: str = "fpt", rf: float = 0.05, cost: float = 0.001, use_kalman: bool = True):
        self.window = window
        self.solver_type = solver_type
        self.rf = rf
        self.cost = cost
        self.use_kalman = use_kalman

    def calibrate_ou(self, spread_window: np.ndarray):
        """
        Calibre les paramètres OU (theta, mu, sigma) via régression AR(1).
        x_{t+1} = a + b*x_t + eps
        """
        if len(spread_window) < 10:
            return 0.1, np.mean(spread_window), np.std(spread_window)
            
        y = spread_window[1:]
        x = sm.add_constant(spread_window[:-1])
        res = sm.OLS(y, x).fit()
        
        a, b = res.params
        residuals = res.resid
        
        # Mapping AR(1) -> OU
        dt = 1.0 # 1 bar = 1 unit of time
        # Clipping pour la stabilité numérique (b doit être > 0 et < 1)
        b = np.clip(b, 0.0001, 0.9999)
        
        theta = -np.log(b) / dt
        mu = a / (1 - b)
        sigma = np.std(residuals) * np.sqrt(-2 * np.log(b) / (dt * (1 - b**2)))
        
        return theta, mu, sigma

    def generate(self, timestamps, price_a, price_b, betas=None):
        from research.institutional.kalman_filter import KalmanHedgeRatio, KalmanOUState
        
        n = len(price_a)
        log_a = np.log(price_a)
        log_b = np.log(price_b)
        
        # Initialisation Kalman
        kh = KalmanHedgeRatio(delta=1e-5, R=1e-3)
        ko = KalmanOUState(q_mu=1e-4, r_obs=1e-2)
        
        dynamic_betas = np.zeros(n)
        spreads = np.zeros(n)
        mus = np.zeros(n)
        
        entry_long = np.zeros(n)
        entry_short = np.zeros(n)
        exit_signal = np.zeros(n)
        
        b_stars = np.full(n, np.nan)
        d_stars = np.full(n, np.nan)
        
        last_b, last_d = np.nan, np.nan
        recalib_freq = 8
        
        # Phase de Warmup pour initialiser le Kalman
        for i in range(n):
            # 1. Update Hedge Ratio (Beta)
            beta, alpha = kh.update(log_b[i], log_a[i])
            dynamic_betas[i] = beta
            
            # 2. Update Spread
            spreads[i] = log_a[i] - beta * log_b[i] - alpha
            
            # 3. Update Mu (Equilibrium)
            mus[i] = ko.update(spreads[i], 0.1) # 0.1 as proxy theta
            
            if i < self.window: continue

            # 4. HJB Solver (Recalibration périodique)
            if i % recalib_freq == 0:
                window_slice = spreads[i - self.window : i]
                try:
                    theta, mu, sigma = self.calibrate_ou(window_slice)
                    # On utilise le MU du Kalman pour plus de réactivité institutionnelle
                    target_mu = mus[i]
                    
                    if theta > 0 and not np.isnan(theta):
                        if self.solver_type == "fpt":
                            solver = HJBSolverFPT(theta, target_mu, sigma, rf=self.rf, cost=self.cost)
                        else:
                            solver = HJBSolverDP(theta, target_mu, sigma, rf=self.rf, cost=self.cost)
                            
                        res = solver.solve()
                        last_b, last_d = res["b_star"], res["d_star"]
                except:
                    pass

            b_stars[i], d_stars[i] = last_b, last_d

            # 5. Signal Logic
            if spreads[i] > b_stars[i]:
                entry_short[i] = 1
            elif spreads[i] < d_stars[i]:
                entry_long[i] = 1
                
            # Exit sur croisement de Mu (Kalman-based)
            if (spreads[i-1] > mus[i-1] and spreads[i] <= mus[i]) or (spreads[i-1] < mus[i-1] and spreads[i] >= mus[i]):
                exit_signal[i] = 1

        zscores = (spreads - mus) / (np.nanstd(spreads) + 1e-9)

        return InstitutionalSignal(
            timestamps=timestamps,
            price_a=price_a,
            price_b=price_b,
            spreads=spreads,
            zscores=zscores,
            betas=dynamic_betas,
            entry_long=entry_long,
            entry_short=entry_short,
            exit_signal=exit_signal,
            b_star=b_stars,
            d_star=d_stars
        )
