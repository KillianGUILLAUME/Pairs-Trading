"""
Pairs Formation Pipeline
1. Clustering (hierarchical + DBSCAN) → pré-filtrer l'univers
2. Cointegration tests (Engle-Granger, Johansen) → valider les paires
3. Scoring & Ranking
"""

import numpy as np
import polars as pl
import pandas as pd
from pathlib import Path
from itertools import combinations
from dataclasses import dataclass, field
from typing import Optional
from loguru import logger

from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import squareform
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

from statsmodels.tsa.stattools import coint, adfuller
from statsmodels.tsa.vector_ar.vecm import coint_johansen


# ============================================================
# DATA STRUCTURES
# ============================================================

@dataclass
class PairResult:
    symbol_a: str
    symbol_b: str
    correlation: float
    cluster_id: int
    
    # Engle-Granger
    eg_pvalue: Optional[float] = None
    eg_statistic: Optional[float] = None
    eg_hedge_ratio: Optional[float] = None
    
    # Johansen
    jh_trace_stat: Optional[float] = None
    jh_crit_95: Optional[float] = None
    jh_cointegrated: Optional[bool] = None
    
    # Spread stats
    half_life: Optional[float] = None
    hurst_exponent: Optional[float] = None
    spread_std: Optional[float] = None
    spread_mean: Optional[float] = None
    
    # Score final
    score: float = 0.0

    @property
    def is_cointegrated(self) -> bool:
        eg_ok = self.eg_pvalue is not None and self.eg_pvalue < 0.05
        jh_ok = self.jh_cointegrated is True
        return eg_ok or jh_ok


# ============================================================
# 1. CLUSTERING
# ============================================================

class PairsClustering:
    """
    Clustering sur les log-returns pour grouper les actifs similaires.
    On ne teste la cointégration qu'entre actifs du même cluster → gain de temps.
    """

    def __init__(self, method: str = "hierarchical", n_clusters: int = 5):
        self.method = method
        self.n_clusters = n_clusters
        self.labels_ = None
        self.linkage_matrix_ = None

    def fit(self, returns: pd.DataFrame) -> dict[str, int]:
        """
        Parameters
        ----------
        returns : pd.DataFrame
            colonnes = symboles, index = timestamps, valeurs = log-returns

        Returns
        -------
        dict symbol → cluster_id
        """
        # Matrice de corrélation → distance
        corr_matrix = returns.corr()
        distance_array = np.sqrt(0.5 * (1 - corr_matrix.to_numpy()))
        distance_array = np.array(distance_array, copy=True)
        np.fill_diagonal(distance_array, 0)

        if self.method == "hierarchical":
            return self._hierarchical(returns.columns.tolist(), distance_array)
        elif self.method == "dbscan":
            return self._dbscan(returns, distance_array)
        else:
            raise ValueError(f"Méthode inconnue : {self.method}")

    def _hierarchical(self, symbols: list[str], distance_matrix: np.ndarray) -> dict[str, int]:
        condensed = squareform(distance_matrix)  # ← enlève le .values
        self.linkage_matrix_ = linkage(condensed, method="ward")
        labels = fcluster(self.linkage_matrix_, t=self.n_clusters, criterion="maxclust")
        self.labels_ = labels
        return {sym: int(lbl) for sym, lbl in zip(symbols, labels)}

    def _dbscan(self, returns: pd.DataFrame, distance_matrix: pd.DataFrame) -> dict[str, int]:
        db = DBSCAN(eps=0.3, min_samples=2, metric="precomputed")
        labels = db.fit_predict(distance_matrix.values)
        self.labels_ = labels
        cluster_map = {sym: int(lbl) for sym, lbl in zip(returns.columns, labels)}
        logger.info(f"DBSCAN → {len(set(labels))} clusters (dont bruit=-1)")
        return cluster_map

    def plot_dendrogram(self, symbols: list, ax=None):
        import matplotlib.pyplot as plt
        if self.linkage_matrix_ is None:
            raise RuntimeError("Lance fit() d'abord")
        if ax is None:
            fig, ax = plt.subplots(figsize=(14, 6))
        dendrogram(self.linkage_matrix_, labels=symbols, ax=ax,
                   leaf_rotation=45, leaf_font_size=10)
        ax.set_title("Hierarchical Clustering - Crypto Universe")
        ax.set_ylabel("Distance")
        return ax


# ============================================================
# 2. COINTEGRATION TESTS
# ============================================================

class CointegrationTester:

    def __init__(self, significance: float = 0.05):
        self.significance = significance

    def test_pair(self, series_a: np.ndarray, series_b: np.ndarray) -> dict:
        results = {}

        # --- Engle-Granger ---
        try:
            eg_stat, eg_pval, _ = coint(series_a, series_b)
            # hedge ratio via OLS
            hedge_ratio = np.polyfit(series_b, series_a, 1)[0]
            results["eg_statistic"] = float(eg_stat)
            results["eg_pvalue"] = float(eg_pval)
            results["eg_hedge_ratio"] = float(hedge_ratio)
        except Exception as e:
            logger.warning(f"EG failed: {e}")
            results["eg_pvalue"] = 1.0

        # --- Johansen ---
        try:
            data = np.column_stack([series_a, series_b])
            jh = coint_johansen(data, det_order=0, k_ar_diff=1)
            trace_stat = jh.lr1[0]
            crit_95 = jh.cvt[0, 1]
            results["jh_trace_stat"] = float(trace_stat)
            results["jh_crit_95"] = float(crit_95)
            results["jh_cointegrated"] = bool(trace_stat > crit_95)
        except Exception as e:
            logger.warning(f"Johansen failed: {e}")
            results["jh_cointegrated"] = False

        return results

    def compute_spread_stats(
        self, series_a: np.ndarray, series_b: np.ndarray, hedge_ratio: float
    ) -> dict:
        spread = series_a - hedge_ratio * series_b

        # Half-life (Ornstein-Uhlenbeck)
        half_life = self._ou_half_life(spread)

        # Hurst exponent
        hurst = self._hurst_exponent(spread)

        return {
            "half_life": half_life,
            "hurst_exponent": hurst,
            "spread_std": float(np.std(spread)),
            "spread_mean": float(np.mean(spread)),
        }

    def _ou_half_life(self, spread: np.ndarray) -> float:
        """Régression AR(1) pour estimer la half-life de mean reversion"""
        y = np.diff(spread)
        x = spread[:-1]
        x = x - x.mean()
        beta = np.dot(x, y) / np.dot(x, x)
        if beta >= 0:
            return np.inf  # pas de mean reversion
        return float(-np.log(2) / beta)

    def _hurst_exponent(self, ts: np.ndarray, max_lag: int = 20) -> float:
        """Hurst < 0.5 → mean reverting, = 0.5 → random walk, > 0.5 → trending"""
        lags = range(2, max_lag)
        tau = [np.std(np.subtract(ts[lag:], ts[:-lag])) for lag in lags]
        poly = np.polyfit(np.log(lags), np.log(tau), 1)
        return float(poly[0])


# ============================================================
# 3. PAIRS FORMATION PIPELINE
# ============================================================

class PairsFormation:

    def __init__(
        self,
        n_clusters: int = 5,
        clustering_method: str = "hierarchical",
        significance: float = 0.05,
        min_half_life: float = 1.0,     # heures
        max_half_life: float = 500.0,   # heures
        min_correlation: float = 0.5,
    ):
        self.clustering = PairsClustering(clustering_method, n_clusters)
        self.tester = CointegrationTester(significance)
        self.significance = significance
        self.min_half_life = min_half_life
        self.max_half_life = max_half_life
        self.min_correlation = min_correlation

    def fit(self, prices: pd.DataFrame) -> list[PairResult]:
        """
        Parameters
        ----------
        prices : pd.DataFrame
            colonnes = symboles, index = timestamps, valeurs = prix close

        Returns
        -------
        list[PairResult] triée par score décroissant
        """
        logger.info(f"Universe : {prices.shape[1]} symboles, {prices.shape[0]} bars")

        # Log-returns pour clustering
        returns = np.log(prices).diff().dropna()

        # 1. Clustering
        logger.info("--- Clustering ---")
        cluster_map = self.clustering.fit(returns)

        # Corrélations pour filtrage rapide
        corr_matrix = returns.corr()

        # 2. Générer les paires intra-cluster
        symbols = prices.columns.tolist()
        pairs_to_test = []

        for sym_a, sym_b in combinations(symbols, 2):
            cid_a = cluster_map[sym_a]
            cid_b = cluster_map[sym_b]
            corr = corr_matrix.loc[sym_a, sym_b]

            # Filtre : même cluster ET corrélation suffisante
            if cid_a == cid_b and abs(corr) >= self.min_correlation:
                pairs_to_test.append((sym_a, sym_b, cid_a, corr))

        logger.info(f"Paires à tester : {len(pairs_to_test)}")

        # 3. Tests de cointégration
        logger.info("--- Cointegration tests ---")
        results = []

        for sym_a, sym_b, cluster_id, corr in pairs_to_test:
            series_a = prices[sym_a].values
            series_b = prices[sym_b].values

            coint_results = self.tester.test_pair(series_a, series_b)
            hedge_ratio = coint_results.get("eg_hedge_ratio", 1.0)
            spread_stats = self.tester.compute_spread_stats(series_a, series_b, hedge_ratio)

            pair = PairResult(
                symbol_a=sym_a,
                symbol_b=sym_b,
                correlation=corr,
                cluster_id=cluster_id,
                **coint_results,
                **spread_stats,
            )
            pair.score = self._score(pair)
            results.append(pair)

        # 4. Filtres qualité
        valid = [p for p in results if self._is_valid(p)]
        valid.sort(key=lambda x: x.score, reverse=True)

        logger.info(f"Paires valides : {len(valid)} / {len(results)}")
        return valid

    def _is_valid(self, p: PairResult) -> bool:
        if not p.is_cointegrated:
            return False
        hl = p.half_life
        if hl is None or np.isinf(hl) or hl < self.min_half_life or hl > self.max_half_life:
            return False
        if p.hurst_exponent is None or p.hurst_exponent >= 0.5:
            return False
        return True

    def _score(self, p: PairResult) -> float:
        """Score composite : on veut p-value basse, hurst bas, half-life raisonnable"""
        score = 0.0
        if p.eg_pvalue is not None:
            score += (1 - p.eg_pvalue) * 40          # max 40
        if p.hurst_exponent is not None:
            score += max(0, (0.5 - p.hurst_exponent)) * 60  # max 30
        if p.half_life is not None and not np.isinf(p.half_life):
            # Pénalise si trop court ou trop long
            optimal = 24.0  # 24h idéal
            score += max(0, 10 - abs(p.half_life - optimal) / optimal * 10)
        return float(score)


def rolling_coint_stability(
    series_a: np.ndarray,
    series_b: np.ndarray,
    window: int = 500,
    step: int = 100,
) -> pd.DataFrame:
    """
    Teste la cointégration sur des fenêtres glissantes.
    Une paire stable doit rester cointégrée tout au long de la période.
    """
    results = []
    for start in range(0, len(series_a) - window, step):
        end = start + window
        try:
            _, pval, _ = coint(series_a[start:end], series_b[start:end])
        except:
            pval = 1.0
        results.append({"start": start, "end": end, "pval": pval})
    
    df = pd.DataFrame(results)
    df["is_coint"] = df["pval"] < 0.05
    df["stability"] = df["is_coint"].mean()  # % du temps cointégré
    return df