import numpy as np

class FractionalKelly:
    """
    Sizing basé sur Kelly Criterion fractionné.
    f* = (p*b - q) / b  où b = win/loss ratio
    
    Adapté ML : utilise la probabilité prédite par le modèle.
    """

    def __init__(self, fraction: float = 0.25, max_leverage: float = 2.0):
        self.fraction      = fraction       # Kelly fractionné (0.25 = quart-Kelly)
        self.max_leverage  = max_leverage

    def size(
        self,
        prob_win: float,      # sortie du modèle ML (proba label=+1)
        win_return: float,    # return moyen sur trades gagnants
        loss_return: float,   # return moyen sur trades perdants (positif)
    ) -> float:
        """
        Retourne la fraction du capital à allouer [0, max_leverage]
        """
        if win_return < 1e-6 or loss_return < 1e-6:
            return 0.0

        b = win_return / loss_return
        p = prob_win
        q = 1.0 - p

        kelly = (p * b - q) / b
        kelly = max(0.0, kelly)              # pas de position si négatif
        sized = kelly * self.fraction        # Kelly fractionné
        return min(sized, self.max_leverage) # cap leverage

    def size_array(
        self,
        probs: np.ndarray,
        win_return: float,
        loss_return: float,
    ) -> np.ndarray:
        return np.array([
            self.size(p, win_return, loss_return) for p in probs
        ])
