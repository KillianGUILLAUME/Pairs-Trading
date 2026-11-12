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
        win_return,
        loss_return,
    ) -> np.ndarray:
        """
        Taille Kelly fractionnée pour chaque probabilité. `win_return` et `loss_return` sont
        des scalaires OU des tableaux de même longueur que `probs` (gain / perte propres à
        chaque barre, ex. issus du solveur HJB). Les valeurs NaN ou nulles donnent 0.
        """
        probs = np.asarray(probs, dtype=float)
        w = np.broadcast_to(np.asarray(win_return, dtype=float), probs.shape)
        l = np.broadcast_to(np.asarray(loss_return, dtype=float), probs.shape)
        ok = np.isfinite(probs) & np.isfinite(w) & np.isfinite(l) & (w >= 1e-6) & (l >= 1e-6)
        out = np.zeros(probs.shape)
        b = np.where(ok, w / np.where(l > 0, l, 1.0), 1.0)
        kelly = np.where(ok, (probs * b - (1.0 - probs)) / b, 0.0)
        out = np.minimum(np.maximum(kelly, 0.0) * self.fraction, self.max_leverage)
        return np.where(ok, out, 0.0)
