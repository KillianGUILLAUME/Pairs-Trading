"""
neural_nets/screened_loader.py

Dataset d'entraînement de la Neural SDE à partir des fenêtres produites par
`research/pairs/screener.py` (schéma CAUSAL, novembre 2026) :

    - `log_prices_a`, `log_prices_b` : fenêtre CIBLE [t, t+L) à générer ;
    - `cond_*`   : descripteurs calculés sur la fenêtre PASSÉE [t-W, t) — ce sont
                   les seules features de conditionnement (causales) ;
    - `target_*` : diagnostics de la fenêtre cible (ADF, Hurst, demi-vie, OU),
                   utilisés UNIQUEMENT pour la curation (filtre), jamais comme entrée.

Prétraitement des chemins : ancrage à 0 (x_t - x_0) puis facteur d'échelle global
unique (1 / std de tous les points ancrés). Conditionnement : z-score des `cond_*`
+ spread initial (centré par la moyenne globale des log-prix, même facteur d'échelle).

Les statistiques (colonnes, moyennes, écarts-types, facteur d'échelle) sont
exposées par `conditioning_metadata()` et stockées dans le checkpoint pour que
l'inférence (`GenerativeSDEEngine.conditions_from_history`) reproduise exactement
la standardisation de l'entraînement.
"""

from __future__ import annotations

import json
import os
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

try:
    from neural_nets.signature_torch import signature_chunked
except ImportError:
    from signature_torch import signature_chunked

COND_PREFIX = "cond_"
TARGET_PREFIX = "target_"
HALF_LIFE_CLIP = 500.0
INITIAL_SPREAD_COL = "initial_spread_scaled"


def prepare_condition_matrix(df: pd.DataFrame, cond_cols: list[str], half_life_clip: float = HALF_LIFE_CLIP) -> np.ndarray:
    """Matrice brute des conditions (N, C) : clip de la demi-vie, NaN/inf -> valeurs sûres."""
    X = np.array(df[cond_cols].to_numpy(dtype=np.float64), dtype=np.float64, copy=True)  # copie : pandas>=3 renvoie une vue en lecture seule
    for j, c in enumerate(cond_cols):
        if c.endswith("half_life"):
            X[:, j] = np.clip(np.nan_to_num(X[:, j], nan=half_life_clip, posinf=half_life_clip), 0.0, half_life_clip)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return X.astype(np.float32)


def condition_vector_from_features(feature_dict: dict, log_a0: float, log_b0: float, meta: dict) -> np.ndarray:
    """
    Construit le vecteur de conditionnement (1, C+1) à l'inférence, avec EXACTEMENT
    la standardisation de l'entraînement.

    Args:
        feature_dict : sortie de `features_eng.get_screener_feature_dict` sur la fenêtre passée
                       (clés sans préfixe `cond_`).
        log_a0, log_b0 : log-prix au point d'ancrage (première barre générée).
        meta : `ScreenedPairsDataset.conditioning_metadata()` (stocké dans le checkpoint).
    """
    cols = meta["columns"]
    raw = np.array([[float(feature_dict[c[len(COND_PREFIX):]]) for c in cols]], dtype=np.float64)
    clip = float(meta.get("half_life_clip", HALF_LIFE_CLIP))
    for j, c in enumerate(cols):
        if c.endswith("half_life"):
            raw[:, j] = np.clip(np.nan_to_num(raw[:, j], nan=clip, posinf=clip), 0.0, clip)
    raw = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)
    mean = np.asarray(meta["mean"], dtype=np.float64)
    std = np.where(np.asarray(meta["std"], dtype=np.float64) == 0, 1e-8, np.asarray(meta["std"], dtype=np.float64))
    z = (raw - mean) / std
    pm = np.asarray(meta["price_mean"], dtype=np.float64)
    init_spread = ((log_a0 - pm[0]) - (log_b0 - pm[1])) * float(meta["scale_factor"])
    return np.concatenate([z, [[init_spread]]], axis=1).astype(np.float32)


class ScreenedPairsDataset(Dataset):
    """
    Args:
        parquet_path   : fichier produit par `build_screened_dataset.py` (schéma causal).
        max_adf_pvalue, max_hurst, max_half_life : filtres de curation.
        filter_on      : "target" (défaut : la fenêtre cible est stationnaire) ou "cond"
                         (la fenêtre passée l'était : inclut les ruptures de cointégration).
        sig_depth, sig_time_augment, sig_lead_lag : signature pré-calculée des chemins réels
                         (même fonction que la loss, canal temps inclus).
        mean / std / condition_stats : statistiques imposées (ex. jeu de validation externe).
    """

    def __init__(
        self,
        parquet_path: str,
        max_adf_pvalue: float = 1.0,
        max_hurst: float = 1.0,
        max_half_life: float = float("inf"),
        filter_on: str = "target",
        sig_depth: int = 3,
        sig_time_augment: bool = True,
        sig_lead_lag: bool = False,
        mean: Optional[np.ndarray] = None,
        std: Optional[float] = None,
        condition_stats: Optional[dict] = None,
        device: Optional[str] = None,
        half_life_clip: float = HALF_LIFE_CLIP,
    ):
        print(f"Chargement du dataset screened : {parquet_path}")
        df = pd.read_parquet(parquet_path)
        n_total = len(df)

        cond_cols = [c for c in df.columns if c.startswith(COND_PREFIX)]
        if not cond_cols:
            raise ValueError(
                f"{parquet_path} n'a pas de colonnes `cond_*` : schéma ancien (features calculées sur la "
                "fenêtre cible = fuite). Régénérer avec research/pairs/build_screened_dataset.py."
            )
        if filter_on not in ("target", "cond"):
            raise ValueError("filter_on doit être 'target' ou 'cond'")
        pfx = f"{filter_on}_"

        # 1. Curation
        hl = df[f"{pfx}half_life"].replace([np.inf, -np.inf], np.nan).fillna(half_life_clip)
        mask = (
            (df[f"{pfx}adf_pvalue"] <= max_adf_pvalue)
            & (df[f"{pfx}hurst"] <= max_hurst)
            & (hl <= max_half_life)
        )
        df = df[mask].reset_index(drop=True)
        n_filtered = len(df)
        print(f"   {n_filtered}/{n_total} fenêtres retenues "
              f"(filtre sur '{filter_on}' : ADF≤{max_adf_pvalue}, H≤{max_hurst}, HL≤{max_half_life})")
        if n_filtered == 0:
            raise ValueError("Aucune fenêtre après filtrage.")

        # 2. Chemins cibles
        self.log_prices_a = np.array(df["log_prices_a"].tolist(), dtype=np.float32)
        self.log_prices_b = np.array(df["log_prices_b"].tolist(), dtype=np.float32)
        self.seq_len = self.log_prices_a.shape[1]
        self.timestamps_start = df["timestamp_start"].to_numpy() if "timestamp_start" in df.columns else None
        self.pairs = df[["pair_a", "pair_b"]].to_numpy() if {"pair_a", "pair_b"} <= set(df.columns) else None

        # 3. Conditions causales (z-score)
        self.condition_columns = list(cond_cols)
        self.half_life_clip = float(half_life_clip)
        raw_metrics = prepare_condition_matrix(df, self.condition_columns, self.half_life_clip)
        if condition_stats is not None:
            self.metrics_mean = np.asarray(condition_stats["mean"], dtype=np.float32)
            self.metrics_std = np.asarray(condition_stats["std"], dtype=np.float32)
        else:
            self.metrics_mean = raw_metrics.mean(axis=0)
            self.metrics_std = raw_metrics.std(axis=0)
        safe_std = np.where(self.metrics_std == 0, 1e-8, self.metrics_std)
        self.metrics = torch.tensor((raw_metrics - self.metrics_mean) / safe_std, dtype=torch.float32)

        # 4. Ancrage à zéro + facteur d'échelle global
        self.raw_features = np.stack([self.log_prices_a, self.log_prices_b], axis=-1)   # (N, L, 2)
        all_points = self.raw_features.reshape(-1, 2)
        self.mean = np.asarray(mean, dtype=np.float32) if mean is not None else all_points.mean(axis=0)

        centered = self.raw_features - self.mean
        initial_points = centered[:, 0:1, :]
        anchored = centered - initial_points
        self.global_std = float(std) if std is not None else float(np.std(anchored))
        self.scale_factor = 1.0 / (self.global_std + 1e-8)
        self.scaled_features = (anchored * self.scale_factor).astype(np.float32)
        self.std = self.global_std

        initial_spreads = (initial_points[:, 0, 0] - initial_points[:, 0, 1]) * self.scale_factor
        self.metrics = torch.cat([self.metrics, torch.tensor(initial_spreads, dtype=torch.float32).unsqueeze(1)], dim=1)
        self.condition_dim = self.metrics.shape[1]
        print(f"   ⚖️ Ancrage à 0 + scaling global : std={self.global_std:.6f} | facteur={self.scale_factor:.2f} "
              f"| {self.condition_dim} conditions ({len(self.condition_columns)} causales + spread initial)")

        # 5. Signatures des chemins réels (même fonction que la loss)
        self.sig_depth = sig_depth
        self.sig_time_augment = sig_time_augment
        self.sig_lead_lag = sig_lead_lag
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🚀 Pré-calcul des {len(self.scaled_features)} signatures (depth={sig_depth}, temps={sig_time_augment}, lead-lag={sig_lead_lag})...")
        sigs = signature_chunked(torch.from_numpy(self.scaled_features), sig_depth, sig_time_augment, sig_lead_lag, device=device)
        self.sig_mean = sigs.mean(dim=0)
        self.sig_std = sigs.std(dim=0).clamp(min=1e-8)
        self.signatures = (sigs - self.sig_mean) / self.sig_std
        print(f"✅ Signatures pré-calculées : {tuple(self.signatures.shape)}")

        # 6. Méta-données du dataset (écrites par build_screened_dataset.py) si présentes
        self.dataset_meta = {}
        meta_path = os.path.splitext(parquet_path)[0] + "_meta.json"
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                self.dataset_meta = json.load(f)

    # ------------------------------------------------------------------
    def conditioning_metadata(self) -> dict:
        """Tout ce qu'il faut pour reproduire le conditionnement à l'inférence."""
        return {
            "columns": list(self.condition_columns),
            "mean": [float(x) for x in self.metrics_mean],
            "std": [float(x) for x in self.metrics_std],
            "half_life_clip": self.half_life_clip,
            "price_mean": [float(x) for x in self.mean],
            "scale_factor": float(self.scale_factor),
            "global_std": float(self.global_std),
            "past_window": int(self.dataset_meta.get("past_window", self.seq_len)),
            "fast_features": bool(self.dataset_meta.get("fast_features", True)),
            "seq_len": int(self.seq_len),
            "schema": "causal_v1",
        }

    def __len__(self):
        return len(self.scaled_features)

    def __getitem__(self, idx):
        path = torch.from_numpy(self.scaled_features[idx])
        return path, self.metrics[idx], self.signatures[idx]
