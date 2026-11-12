# pairs_trading

Code de recherche sur une stratégie de retour à la moyenne sur paires de cryptomonnaies (Binance, barres horaires, 2022-2023) et sur un générateur de trajectoires (SDE neuronale).
Le résultat principal est **méthodologique** : un signal peut être cohérent avec la théorie et pourtant sans valeur encaissable.

> Ancienne documentation (approche autorégressive CFM, abandonnée) : `docs/README_historique.md`.

## Résultats

- **Théorie vérifiée sur simulation.** Pour un spread d'Ornstein--Uhlenbeck, la politique de retournement est optimale pour le critère ergodique, avec des formules fermées (`docs/strategie_hypotheses.pdf`). Le moteur réalise 337 % pour 335 % prédits.
- **Pas d'avantage brut sur données réelles** (153 paires, 2022-2023) : portefeuille équipondéré −5 % sur 2 ans, rendement brut déjà négatif, ni barrière de stationnarité, ni échelle de bandes, ni stop n'y changent rien.
- **Trois mécanismes de faux signal isolés** : spread recalculé à chaque rafraîchissement de la couverture, innovation de Kalman utilisée comme spread (filtre passe-haut), seuils ancrés sur une moyenne exponentielle rapide (`docs/papier_validation.pdf`, brouillon).
- **Générateur** : meilleur qu'une marche aléatoire (MMD de signature 4 fois plus faible), mais sans retour à la moyenne, sans queues épaisses, sans clustering de volatilité ; extensions implémentées (volatilité latente, bruit de Student, rappel à la moyenne), **entraînement non encore lancé**.

## Ce qui n'est pas validé

Les oracles XGBoost (`research/institutional/*oracle*`, `research/models/`) ont été entraînés sur d'anciens signaux et ne sont plus calibrés. La plateforme `live_platform/` n'a jamais passé d'ordre réel. Aucun résultat n'est établi hors 2022-2023, hors spot horaire, ni sans biais de survie dans la sélection de l'univers.

## Organisation

| Dossier | Contenu |
|---------|---------|
| `research/institutional/` | solveurs HJB (`hjb_solvers.py`), moteur de signaux (`signal_engine.py`), politique à couverture figée (`locked_policy.py`) |
| `research/backtest/` | moteur de gain, banc de mauvaise spécification, walk-forward réel, calibration du stop, évaluation du générateur |
| `research/pairs/` | screener causal des fenêtres cointégrées, construction du jeu d'entraînement |
| `neural_nets/` | SDE neuronale, signature tronquée en PyTorch, perte MMD, entraînement |
| `features/` | estimateurs (Hurst par variogramme, OU, queues) |
| `scripts/data_registry.py` | versionnement des données et modèles par empreinte (S3) |
| `tests/` | 68 tests (solveurs, backtest, moteur de signaux, signature, SDE, politique à couverture figée) |
| `docs/` | `strategie_hypotheses.tex` (preuves), `papier_validation.tex` (brouillon), `audit_2026-11-10.md` (journal) |
| `legacy/` | code retiré de la chaîne active, conservé pour référence |

## Reproduire

Python 3.12, environnement virtuel :

```bash
python -m venv venv && source venv/bin/activate && pip install -r requirements.txt
pytest tests
```

Données brutes (Binance, spot horaire) puis jeu d'entraînement causal :

```bash
python data/pipeline.py
python research/pairs/build_screened_dataset.py        # ~4 min (Mac, 8 cœurs)
```

Résultats du papier (durées approximatives mesurées sur un Mac 8 cœurs) :

```bash
python research/backtest/model_misspecification.py --paths 16    # ~10 min   banc sur simulation
python research/backtest/real_walk_forward.py                    # ~15 min   walk-forward sur 153 paires
python research/backtest/real_reversion_check.py                 # ~10 min   pente réalisé / prédit
python research/backtest/real_trade_decomposition.py --lock      # décomposition par trade (6 paires : --limit 6)
python research/backtest/real_stop_calibration.py                # ~20 min   stop calibré sur 2022, évalué sur 2023
```

Générateur (entraînement sur GPU conseillé, ~2 h ; `neural_nets/utils/launcher.py` le lance sur AWS) :

```bash
python neural_nets/train.py
python research/backtest/evaluate_generator.py --checkpoint data/models/<run>_world_model.pt
```

Les résultats sont écrits dans `data/optimization_results/`. Les données et modèles ne sont pas dans git : voir `scripts/data_registry.py`.

## Méthode

- Tout test sur données réelles est **causal** (couverture, calibration et seuils n'utilisent que le passé).
- Tout paramètre réglé sur du réel l'est sur 2022 puis jugé sur 2023, avec un critère fixé avant les résultats.
- Les résultats négatifs sont conservés et documentés.
