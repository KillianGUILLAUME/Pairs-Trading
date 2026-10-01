import numpy as np
from scipy.special import pbdv
from scipy.optimize import minimize_scalar
from loguru import logger

class HJBSolverFPT:
    """
    Solveur Semi-Analytique basé sur le First Passage Time (FPT).
    Utilise les fonctions de Weber (Parabolic Cylinder Functions).
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
        self.rf = rf           # Discount rate
        self.cost = cost       # Coût de transaction
        
    def _phi(self, x):
        """
        Solution croissante de l'équation différentielle du processus OU.
        phi(x) est proportionnelle à exp(z^2/4) * D_{-v}(-z)
        où z = (x-mu) * sqrt(2*theta)/sigma et v = rf/theta
        """
        z = (x - self.mu) * np.sqrt(2 * self.theta) / self.sigma
        v = -self.rf / self.theta
        
        # On clip z pour éviter l'overflow dans pbdv ou exp
        z = np.clip(z, -100, 100)
        
        val, _ = pbdv(v, -z)
        
        # On utilise une valeur minimale pour éviter la division par zéro
        val = np.maximum(val, 1e-15)
        
        # log_phi pour éviter l'overflow de l'exponentielle
        return val * np.exp(z**2 / 4)

    def _expected_value(self, b):
        """
        Maximise (b - mu - cost) / phi(b)
        """
        if b <= self.mu + self.cost:
            return -1e10 # Valeur très basse pour éviter cette zone
            
        payoff = (b - self.mu) - self.cost
        denom = self._phi(b)
        
        if np.isnan(denom) or np.isinf(denom) or denom <= 0:
            return -1e10
            
        return payoff / denom

    def solve(self):
        """
        Recherche numérique du b* optimal.
        """
        std_lr = self.sigma / np.sqrt(2 * self.theta)
        
        # On cherche b* dans l'intervalle [mu + cost, mu + 5*sigma]
        res = minimize_scalar(
            lambda x: -self._expected_value(x),
            bounds=(self.mu + self.cost, self.mu + 6 * std_lr),
            method='bounded'
        )
        
        b_star = res.x
        # Par symétrie du processus OU
        d_star = 2 * self.mu - b_star
        
        return {"b_star": b_star, "d_star": d_star}

if __name__ == "__main__":
    solver = HJBSolverFPT(theta=0.5, mu=0.0, sigma=0.02, cost=0.005)
    results = solver.solve()
    print(f"Optimal FPT Thresholds: {results['b_star']:.4f} / {results['d_star']:.4f}")
