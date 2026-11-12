"""
research/backtest/generative_sde_engine.py

Moteur de génération de marchés synthétiques à partir d'un checkpoint Neural SDE.

- Charge le modèle, le facteur d'échelle global et les métadonnées de conditionnement
  (checkpoints produits par `neural_nets/train.py` à partir de novembre 2026).
- `conditions_from_history` reproduit EXACTEMENT le conditionnement d'entraînement
  (features causales sur la fenêtre passée, z-score stocké, spread initial scalé).
- `simulate_markets` : N trajectoires ancrées aux prix réels fournis. Les sauts de
  Poisson additionnels sont désactivés par défaut (ils modifiaient silencieusement la
  loi apprise).
"""

from __future__ import annotations

import os
import sys
import warnings

import numpy as np
import pandas as pd
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from neural_nets.neural_sde import GeneratorSDE                      # noqa: E402
from neural_nets.screened_loader import condition_vector_from_features  # noqa: E402
from features.features_eng import get_screener_feature_dict            # noqa: E402


class GenerativeSDEEngine:
    def __init__(self, model_path: str, device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading Neural SDE from {model_path}...")
        self.model, self.checkpoint = GeneratorSDE.load_from_checkpoint(model_path, device=self.device)
        self.model_path = model_path

        if "scale_factor" not in self.checkpoint:
            raise ValueError(
                f"{model_path} ne contient pas `scale_factor` : checkpoint antérieur au scaling global, "
                "dé-standardisation impossible (voir data/models/legacy/)."
            )
        self.scale_factor = float(self.checkpoint["scale_factor"])
        self.global_std = float(self.checkpoint.get("global_std", 1.0 / self.scale_factor))
        self.scaler = self.checkpoint.get("scaler", {"mean": [0.0, 0.0], "std": self.global_std})
        self.conditioning = self.checkpoint.get("conditioning")   # None : checkpoint sans métadonnées
        self._warned_zero_conditions = False
        self.model.eval()
        cond_info = "yes" if self.conditioning else "NO (conditions nulles à l'inférence)"
        print(f"✅ Model loaded. scale_factor={self.scale_factor:.4f} | condition_dim={self.model.condition_dim} "
              f"| conditioning metadata: {cond_info}")

    # ------------------------------------------------------------------
    def conditions_from_history(self, log_a, log_b, log_btc, log_eth) -> torch.Tensor:
        """
        Vecteur de conditionnement (1, C) à partir de l'historique RÉEL se terminant au
        point d'ancrage : features sur les `past_window` barres précédant la dernière
        barre, spread initial = dernière barre (= première barre générée).
        """
        if self.conditioning is None:
            raise ValueError(
                "Ce checkpoint n'a pas de métadonnées de conditionnement (entraîné avant le schéma causal). "
                "Réentraîner avec neural_nets/train.py, ou passer conditions=None (fenêtre moyenne)."
            )
        W = int(self.conditioning["past_window"])
        log_a, log_b, log_btc, log_eth = (np.asarray(x, dtype=float) for x in (log_a, log_b, log_btc, log_eth))
        if min(len(log_a), len(log_b), len(log_btc), len(log_eth)) < W + 1:
            raise ValueError(f"Historique insuffisant : {W + 1} barres requises (past_window={W} + point d'ancrage).")
        feats = get_screener_feature_dict(
            log_a[-W - 1:-1], log_b[-W - 1:-1], log_btc[-W - 1:-1], log_eth[-W - 1:-1],
            fast=bool(self.conditioning.get("fast_features", True)),
        )
        vec = condition_vector_from_features(feats, float(log_a[-1]), float(log_b[-1]), self.conditioning)
        return torch.from_numpy(vec)

    # ------------------------------------------------------------------
    @torch.no_grad()
    def simulate_markets(
        self,
        n_paths: int = 1,
        bars: int = 1000,
        p_a_init: float = 100.0,
        p_b_init: float = 100.0,
        conditions: torch.Tensor | None = None,
        jump_prob: float = 0.0,
        jump_vol: float = 0.0,
        seed: int | None = None,
        batch_size: int = 1024,
    ) -> list[pd.DataFrame]:
        """
        Simule `n_paths` marchés de `bars` barres ancrés aux prix (p_a_init, p_b_init).

        Args:
            conditions : (1, C) ou (n_paths, C). None -> vecteur nul = « fenêtre moyenne »
                         du dataset d'entraînement (un avertissement est émis).
            jump_prob / jump_vol : sauts de Poisson OPTIONNELS ajoutés aux prix (0 = aucun).
            seed : graine torch/numpy pour la reproductibilité.
        """
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        if conditions is None:
            if not self._warned_zero_conditions:
                warnings.warn("simulate_markets: conditions=None -> conditionnement nul (fenêtre moyenne du dataset). "
                              "Utiliser conditions_from_history(...) pour un scénario ancré sur l'historique réel.")
                self._warned_zero_conditions = True
            conditions = torch.zeros(1, self.model.condition_dim)
        conditions = conditions.to(self.device).float().view(-1, self.model.condition_dim)
        if conditions.shape[0] == 1:
            conditions = conditions.repeat(n_paths, 1)
        elif conditions.shape[0] != n_paths:
            raise ValueError("conditions doit avoir 1 ou n_paths lignes")

        if bars > 4 * self.model_seq_len:
            warnings.warn(f"bars={bars} >> longueur d'entraînement ({self.model_seq_len}) : extrapolation non validée.")

        chunks = []
        for start in range(0, n_paths, batch_size):
            b = min(batch_size, n_paths - start)
            y0 = torch.zeros(b, self.model.data_dim, device=self.device)
            chunks.append(self.model.sample(y0, bars, conditions[start:start + b]).cpu())
        generated_paths = torch.cat(chunks, dim=0)                     # (n_paths, bars, 2), ancré à 0

        log_diffs = generated_paths.numpy() / self.scale_factor
        log_prices_a = log_diffs[:, :, 0] + np.log(p_a_init)
        log_prices_b = log_diffs[:, :, 1] + np.log(p_b_init)
        prices_a = np.exp(log_prices_a)
        prices_b = np.exp(log_prices_b)

        market_dfs = []
        for i in range(n_paths):
            pa, pb = prices_a[i], prices_b[i]
            if jump_prob > 0 and jump_vol > 0:
                pa = pa * (1 + np.random.normal(0, jump_vol, bars) * (np.random.rand(bars) < jump_prob))
                pb = pb * (1 + np.random.normal(0, jump_vol, bars) * (np.random.rand(bars) < jump_prob))
            df = pd.DataFrame({"SYNTH_A": pa, "SYNTH_B": pb})
            df.index.name = "timestamp"
            market_dfs.append(df)
        return market_dfs

    @property
    def model_seq_len(self) -> int:
        if self.conditioning and "seq_len" in self.conditioning:
            return int(self.conditioning["seq_len"])
        return int(self.checkpoint.get("config", {}).get("training", {}).get("seq_len", 128))
