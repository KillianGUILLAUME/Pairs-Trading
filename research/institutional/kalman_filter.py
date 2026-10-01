import numpy as np
from loguru import logger

class KalmanHedgeRatio:
    """
    Filtre de Kalman pour estimer le Hedge Ratio dynamique (Beta).
    Y = Beta * X + Alpha + Error
    """
    def __init__(self, delta=1e-5, R=1e-3):
        self.delta = delta # Vitesse d'adaptation des paramètres
        self.R = R         # Bruit de mesure
        self.theta = np.zeros(2) # [Beta, Alpha]
        self.P = np.zeros((2, 2))
        self.initialized = False

    def update(self, x, y):
        # x: price_b, y: price_a
        state = np.array([x, 1.0])
        
        if not self.initialized:
            self.theta = np.array([y/x if x != 0 else 1.0, 0.0])
            self.P = np.eye(2)
            self.initialized = True

        # Prédiction (Random Walk for theta)
        # P = P + Q
        self.P += self.delta / (1 - self.delta) * np.eye(2)

        # Observation
        y_hat = np.dot(state, self.theta)
        res = y - y_hat

        # Mise à jour
        # S = H*P*H' + R
        S = np.dot(state, np.dot(self.P, state)) + self.R
        # K = P*H' * inv(S)
        K = np.dot(self.P, state) / S
        
        self.theta += K * res
        self.P -= np.outer(K, np.dot(state, self.P))
        
        return self.theta[0], self.theta[1] # Beta, Alpha

class KalmanOUState:
    """
    Filtre de Kalman pour tracker l'équilibre dynamiqe d'un spread (Mu).
    Le spread est supposé suivre un OU.
    """
    def __init__(self, q_mu=1e-4, r_obs=1e-2):
        self.mu = 0.0
        self.P = 1.0
        self.Q = q_mu # Incertitude sur le drift de Mu
        self.R = r_obs # Bruit du spread (volatilité court terme)
        self.initialized = False

    def update(self, spread, theta_dt):
        """
        theta_dt: vitesse de retour pré-estimée (ou constante).
        """
        if not self.initialized:
            self.mu = spread
            self.initialized = True
            return self.mu

        # Prédiction : Mu(t) = Mu(t-1)
        self.P += self.Q

        # Observation : dX = theta(mu - x)dt + sigma*dW -> x_t = (1-theta*dt)x_{t-1} + theta*dt*mu
        # On peut voir le spread actuel comme une observation de Mu : 
        # Mu_obs = (spread_t - (1 - theta_dt)*spread_{t-1}) / theta_dt
        # Mais plus simplement, on filtre le spread vers Mu.
        
        K = self.P / (self.P + self.R)
        self.mu += K * (spread - self.mu)
        self.P *= (1 - K)
        
        return self.mu
