# Plan du livrable : validation d'une stratégie de paires

**Titre de travail** : *Pièges de validation d'une stratégie de retour à la moyenne sur paires de cryptomonnaies : du signal apparent au gain encaissable.*

**Thèse en une phrase** : un spread peut paraître prédictif dans le signal tout en n'offrant aucun gain encaissable, et trois mécanismes précis, tous mesurés ici, produisent cette illusion.

**Format** : 8 à 10 pages (LaTeX, dans la continuité de `docs/strategie_hypotheses.tex`), un README d'une page, un dépôt allégé. Ton : mathématique, sans langage de performance. Aucun rendement présenté comme un résultat positif.

---

## 1. Structure du document

| § | Contenu | Pages | Matériau existant |
|---|---------|-------|-------------------|
| 1 | Question et résumé des résultats (3 phrases, 1 tableau) | 0,5 | à rédiger |
| 2 | Cadre : OU, bandes symétriques, critère ergodique, retournement optimal | 2 | `strategie_hypotheses.tex` §1 à §3 (démontré, vérifié par simulation directe) |
| 3 | Protocole : données, walk-forward causal, séparation 2022 / 2023, coûts | 1 | scripts `real_walk_forward.py`, `real_wf_band_scan.py` |
| 4 | Premier résultat : sur simulation tout fonctionne (théorie 335 % contre 337 % mesurés) | 0,5 | `model_misspecification.py` |
| 5 | Second résultat : en réel, aucun avantage brut | 1 | `real_walk_forward.py`, `real_wf_diagnostics.py` |
| 6 | **Trois mécanismes de faux signal** (cœur du papier) | 3 | voir §2 ci-dessous |
| 7 | Ce qui reste vrai : le retour à la moyenne existe mais est ~3 fois plus faible que prévu ; stop et barrières n'y changent rien | 1 | `real_reversion_check.py`, `real_stop_calibration.py` |
| 8 | Limites et menaces à la validité | 0,5 | voir §4 |
| 9 | Annexe : le générateur (SDE) et sa batterie d'évaluation, état actuel | 1 | `evaluate_generator.py` |

## 2. Les trois mécanismes (cœur du papier)

Chaque mécanisme suit le même schéma : **hypothèse, expérience contrôlée en simulation, mesure sur données réelles, correctif, effet du correctif**.

1. **Spread recalculé à chaque rafraîchissement de couverture.** Le signal « revient vers zéro » sans gain encaissable.
   - Mesuré (6 paires) : écart signal / gain de −262, −127, −49 pb par trade pour un rafraîchissement toutes les 24, 240, 2 400 barres.
   - Correctif : couverture verrouillée à l'entrée ; corrélation signal / gain 0,99.
   - **À refaire sur les 153 paires avant de l'écrire** (voir §3).
2. **Innovation de Kalman comme spread.** Filtre passe-haut : φ du spread 0,98 devient 0,72 sur simulation (β = 1,5).
   - Preuve : proposition démontrée (approximation scalaire) + simulation (−62 % contre +386 % avec OLS roulante).
3. **Seuils ancrés sur une moyenne exponentielle rapide.** Deuxième filtre passe-haut.
   - Mesuré : oracle à 337 % avec μ calibré contre 84 % avec l'ancrage rapide ; la théorie prédit 335 %.

*Point de rigueur à écrire explicitement* : le gain réalisé a toujours été calculé sur les prix réels. Ce qui était faux, c'est le **signal** (il paraissait prédictif), pas la comptabilité. Sur le portefeuille, le verrouillage de la couverture ne change presque pas le total (−5,4 % contre −5,6 %, Sharpe −1,6 contre −1,4) : il change le nombre de trades (119 à 41 par paire) et supprime l'apparence de signal. Ne pas présenter le correctif comme une amélioration de performance.

## 3. Figures et tables

| # | Figure / table | Source | Statut |
|---|----------------|--------|--------|
| F1 | Simulation vs réel : rendement par scénario (barres) | `misspecification.csv`, `real_walk_forward_*` | données prêtes |
| F2 | Signal apparent vs gain réalisé, par trade, avant / après verrouillage | `real_trade_decomposition.py` | **à relancer sur 153 paires** |
| F3 | Écart signal / gain en fonction de la fréquence de rafraîchissement (24, 240, 2 400) | idem | **à relancer sur 153 paires** |
| F4 | φ du spread selon la couverture (Kalman vs OLS vs β vrai) | `hedge.py` (scratchpad) | à intégrer au dépôt |
| F5 | Pente réalisé / prédit par horizon (1, 6, 24, 72 h) | `real_reversion_check.csv` | données prêtes |
| F6 | Excursion adverse des gagnants et des perdants, avec les stops testés | `stop_calibration_trades.csv` | données prêtes |
| T1 | Ablation : couverture × ancrage × verrouillage, rendement et Sharpe du portefeuille 2022 / 2023 | à produire | **manquant** |
| T2 | Batterie du générateur : réel / SDE / marche aléatoire / plancher | `generator_evaluation.csv` | données prêtes (annexe) |

## 4. Menaces à la validité (à écrire, pas à cacher)

- **Deux ans seulement** (2022-2023) et un seul régime baissier puis reprise ; aucune période 2024-2026.
- **Univers choisi avec de l'information du futur** : 23 actifs retenus par volume actuel (biais de survie), et paires pré-filtrées par DBSCAN sur les corrélations de toute la période. Le walk-forward est causal pour la stratégie, pas pour la constitution de l'univers.
- **Barres horaires seulement**, spot, sans funding, sans impact de marché ; coût proportionnel de 5 pb par jambe.
- **Les réglages ont été choisis sur simulation**, ce qui est un avantage méthodologique, mais le simulateur est calibré sur ADA/AVAX.
- **Les oracles XGBoost et la plateforme live ne sont pas évalués** : ils sont exclus du papier.
- La calibration du stop est faite sur 2022 puis jugée sur 2023 ; le critère était fixé avant les résultats et la conclusion est « non validé ».

## 5. Travail restant, par priorité

1. **Refaire la décomposition par trade sur les 153 paires** (F2, F3) : c'est la preuve centrale, aujourd'hui sur 6 paires. Une quinzaine de minutes de calcul par configuration.
2. **Produire T1** (ablation) : 8 combinaisons, mêmes 153 paires, même protocole. Sans cette table le papier affirme des causes sans les isoler.
3. **Script unique `reproduce.py`** qui régénère toutes les figures et tables depuis les parquets (avec les empreintes du registre de données).
4. **Réécrire le §9** après l'entraînement du week-end : si la SDE retrouve le retour à la moyenne et le clustering de volatilité, le dire ; sinon, présenter la batterie comme contribution et la SDE comme travail en cours.
5. **Nettoyage du dépôt** : supprimer `legacy/` et les checkpoints incompatibles de l'index public, retirer les oracles XGBoost non recalibrés et la plateforme live, vérifier qu'aucun secret n'est tracé (`.env` est ignoré, à contrôler dans l'historique git), lock des dépendances, README d'une page avec la commande de reproduction.
6. **Relecture critique** par une personne extérieure au projet avant diffusion.

## 5 bis. Calendrier indicatif (environ 3 semaines)

| Semaine | Contenu |
|---------|---------|
| 1 | Points 1, 2 et 3 (expériences manquantes, script de reproduction) |
| 2 | Rédaction des §1 à §8, figures, nettoyage du dépôt |
| 3 | §9 après l'entraînement, relecture externe, corrections |

## 6. Ce que le document ne dira pas

- Aucune phrase du type « la stratégie génère un alpha ».
- Aucun Sharpe positif présenté sans condition.
- Pas de comparaison avec d'autres stratégies que celles réellement testées.
- Pas d'affirmation de causalité sur les mécanismes qui n'auraient pas été isolés par T1.
