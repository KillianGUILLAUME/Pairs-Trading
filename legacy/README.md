# legacy/

Code et artefacts retirés de la chaîne active lors de l'audit du 10 novembre 2026
(`docs/audit_2026-11-10.md`). Conservés pour référence, **non importables tels quels**.

| Chemin | Pourquoi |
|--------|----------|
| `neural_nets/generate.py` | Appelait `GeneratorSDE.generate_synthetic_paths` (inexistant) et dé-standardisait par asset (schéma antérieur à l'ancrage global). Remplacé par `research/backtest/generative_sde_engine.py`. |
| `research/backtest/generate_MOT_data.py` | Importait `load_real_data` et `CausalVelocityField`, supprimés depuis (chaîne CFM autorégressive). |
| `research/backtest/run_synthetic_backtest.py` | Chemin de modèle en dur vers un fichier inexistant ; `run_generative_cvar.py` couvre le besoin. |
| `data/connectors/sde_connector.py` | Importait `core.models.sde_generator` (inexistant), `BaseConnector`/`logger` non définis. |
| `pipeline/backtest_pipeline.py`, `pipeline/live_pipeline.py` | Pseudo-code (`...`). La boucle live réelle est dans `live_platform/`. |
| `research/institutional/hjb_solver_dp.py` | Différences finies explicites avec CFL ≈ 17 : instable (seuils ≈ μ). Remplacé par `hjb_solvers.py` (ergodique + Leung-Li semi-analytiques). |

## `data/models/legacy/`

| Fichier | Pourquoi |
|---------|----------|
| `best_val_sde.pt` | `state_dict` brut d'une architecture non conditionnelle (première couche `(512, 2)`), écrasé à chaque run : incompatible avec `GeneratorSDE.load_from_checkpoint`. |
| `neural_sde_20260419_*`, `neural_sde_20260420_*` (×7) | `hidden_dim=512`, sans `scale_factor` : dé-standardisation impossible avec le code courant. |

Le seul checkpoint compatible avec le code courant reste `data/models/neural_sde_20260421_001939_world_model.pt`
(conditionnement **non causal** : features de la fenêtre cible, à réentraîner sur le dataset causal).
