import numpy as np
from scipy.interpolate import interp1d
from loguru import logger

class HJBSolverDP:
    """
    Solveur HJB par Programmation Dynamique pour Processus d'Ornstein-Uhlenbeck.
    Détermine les frontières optimales d'entrée et de sortie.
    """
    def __init__(
        self, 
        theta: float, 
        mu: float, 
        sigma: float, 
        rf: float = 0.05, 
        cost: float = 0.001
    ):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.rf = rf           # Taux sans risque (discount)
        self.cost = cost       # Coût de transaction (notional fraction)
        
    def solve(self, x_grid=None, n_iter=1000, tol=1e-6):
        """
        Résolution par itération sur la fonction de valeur.
        On cherche à maximiser le profit espéré net de frais.
        """
        if x_grid is None:
            # Grille centrée sur Mu, couvrant +/- 5 Sigmas
            std_long_run = self.sigma / np.sqrt(2 * self.theta)
            x_grid = np.linspace(self.mu - 6 * std_long_run, self.mu + 6 * std_long_run, 500)
            
        dx = x_grid[1] - x_grid[0]
        dt = 0.01 # Pas de temps discret pour l'itération
        
        # Initialisation de la fonction de valeur (Valeur de l'option d'entrer)
        V = np.zeros_like(x_grid)
        
        # Opérateur Différentiel (Schéma de différence finie UPWIND)
        for i in range(n_iter):
            V_old = V.copy()
            
            # Dérivées centrées pour le second ordre
            d2V_dx2 = np.gradient(np.gradient(V, dx), dx)
            
            # Dérivées Upwind pour le premier ordre (stabilité)
            dV_up = np.zeros_like(V)
            drift = self.theta * (self.mu - x_grid)
            
            # forward diff if drift > 0, backward diff if drift < 0
            idx_pos = drift > 0
            idx_neg = drift < 0
            
            if np.any(idx_pos):
                dV_up[idx_pos] = (np.roll(V, -1)[idx_pos] - V[idx_pos]) / dx
            if np.any(idx_neg):
                dV_up[idx_neg] = (V[idx_neg] - np.roll(V, 1)[idx_neg]) / dx
            
            LV = drift * dV_up + 0.5 * (self.sigma**2) * d2V_dx2
            
            V_new = V + dt * (LV - self.rf * V)
            
            intrinsic_value = np.maximum(0, np.abs(x_grid - self.mu) - self.cost)
            V = np.maximum(V_new, intrinsic_value)
            
            # Clamping pour éviter l'explosion
            V = np.clip(V, 0, 10.0)
            
            if np.max(np.abs(V - V_old)) < tol:
                break
                
        # Extraction des frontières
        # b* est le point où V(x) == intrinsic_value (on arrête d'attendre)
        stopping_points = np.where(np.isclose(V, intrinsic_value, atol=1e-4))[0]
        
        if len(stopping_points) > 0:
            b_star_idx = stopping_points[x_grid[stopping_points] > self.mu]
            d_star_idx = stopping_points[x_grid[stopping_points] < self.mu]
            
            b_star = x_grid[b_star_idx[0]] if len(b_star_idx) > 0 else self.mu + 2 * std_long_run
            d_star = x_grid[d_star_idx[-1]] if len(d_star_idx) > 0 else self.mu - 2 * std_long_run
        else:
            b_star, d_star = self.mu + 2 * std_long_run, self.mu - 2 * std_long_run
            
        return {"b_star": b_star, "d_star": d_star, "grid": x_grid, "value_func": V}

if __name__ == "__main__":
    # Test simple
    solver = HJBSolverDP(theta=0.5, mu=0.0, sigma=0.02, cost=0.005)
    results = solver.solve()
    print(f"Optimal Entry Thresholds: {results['b_star']:.4f} / {results['d_star']:.4f}")
