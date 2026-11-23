> **Historical note (23 Nov):** this plan governed the four parallel workstreams. The code has since
> been simplified and the outputs moved into the milestone folders (`A_checkin_1124/`,
> `B_checkin_1201/`, `C_final_1208/`); paths below refer to the earlier layout.

# Work plan and contracts for the four workstreams

Four workstreams run **in parallel in this same folder**. To avoid overwriting each other they follow
the file-ownership and interface rules below. Course requirements: `_local/Course Project.md`
(Section 2 lists the six parts the report must cover). Proposal: `README.md`.

Milestones: **24 Nov** check-in (cleaned data, plots, correlations, initial features, baseline) ·
**1 Dec** check-in (model comparisons, validation, feature explanations, draft report/slides) ·
**8/10 Dec** presentation; submit report, slides, code (`.ipynb`/`.py`) and README.

## Global rules

- Do **not** commit, push, switch branches, or edit files owned by another workstream.
- Read the database from `src.common.database_path()` (`_local/database.sqlite`, or `$SOCCER_DB`).
  Never copy it into the repo; never write inside `_local/`.
- Shared, frozen files (coordinator only): `src/common.py`, `src/__init__.py`, `requirements.txt`,
  `README.md`, `docs/WORKPLAN.md`. Need a change? Write it under "Requests to coordinator" in
  your decisions file and work around it locally meanwhile.
- Python: the local interpreter (`python`, 3.13, pandas 3, scikit-learn 1.9). Code must also run on
  pandas ≥ 2.2. Use only packages already installed (numpy, pandas, scipy, scikit-learn,
  matplotlib, seaborn, nbconvert/ipykernel). No lightgbm/xgboost/shap.
- Put logic in `src/*.py`; notebooks are thin narrated drivers that import it. Every notebook must
  run top to bottom: `jupyter nbconvert --to notebook --execute --inplace notebooks/<file>.ipynb`.
- Random seeds: `src.common.SEED`. Splits, cohorts, budget and metrics: use `src.common` helpers
  (`load_cleaned`, `cohort`, `rolling_folds`, `final_split`, `budget_metrics`, `metrics_by_year`).
- Never use columns in `LEAKY_COLUMNS` or `ID_COLUMNS` as features. The 2014 test year is touched
  only for final evaluation, never for choosing features, models or settings.
- Write in English. Report sections are short, plain prose aimed at the course's six parts.

## Decision log (every workstream)

When something is ambiguous, **decide it yourself and keep going**, then log it in your own
`docs/decisions/<NN>_<area>.md`, one entry per decision:

```
### D<NN>-<k>: <short question>
- Options considered: ...
- Decision: ...
- Why: ...
- Impact if reversed: ...
- Needs team confirmation: yes/no
```

End the file with "Requests to coordinator" (changes needed in shared files) and
"Open issues" (things you could not resolve).

## Ownership and interfaces

| # | Workstream | Owns (only these paths) |
|---|---|---|
| 1 | Data cleaning (review) | `process_data.py`, `CHECKIN.md`, `checkin/`, `docs/decisions/01_cleaning.md`, `report/01_problem_data_cleaning.md` |
| 2 | Exploration & features | `src/features.py`, `notebooks/02_exploration_features.ipynb`, `figures/eda/`, `results/features/`, `docs/decisions/02_features.md`, `report/02_exploration_features.md` |
| 3 | Modelling | `src/modelling.py`, `notebooks/03_modelling.ipynb`, `results/modelling/`, `results/predictions/`, `docs/decisions/03_modelling.md`, `report/03_methods_model_selection.md` |
| 4 | Evaluation & explanations | `src/evaluation.py`, `notebooks/04_evaluation_explanations.ipynb`, `figures/evaluation/`, `results/evaluation/`, `docs/decisions/04_evaluation.md`, `report/04_evaluation_results.md` |

### Cleaned table (owned by 1, read by everyone)
`checkin/cleaned_player_years.csv.gz`, one row per (`year`, `player_api_id`). Workstream 1 may add
columns or change rows, but must not rename or remove existing columns, and must log any change
in row counts or labels.

### `src/features.py` (owned by 2)
- `build_feature_table(cleaned: pd.DataFrame | None = None) -> pd.DataFrame` — all cleaned columns
  plus engineered features, same rows and keys (`year`, `player_api_id`). Any extra database reads
  use only information dated before each row's 1 August cut-off.
- `FEATURE_SETS: dict[str, list[str]]` with at least these keys:
  - `ea_gap_only` → `["potential_gap"]`
  - `rating_no_potential` → rating + player features **without** `potential` / `potential_gap`
  - `rating_only` → rating + player features (incl. potential)
  - `combined` → `rating_only` + match-history features
- `CATEGORICAL: list[str]` (columns needing one-hot encoding; everything else numeric).
- `make_preprocessor(columns, scale=True, pca_skills=None) -> sklearn ColumnTransformer`, returned
  **unfitted**; median imputation, optional standard scaling, one-hot (unknown ignored), optional PCA
  on the skill columns. It is fitted only inside a model Pipeline on training rows.
- Until this file exists, other workstreams use the `BASE_*` lists in `src.common` with the same
  set names and an equivalent local preprocessor, then switch once it lands.

### Predictions (owned by 3, read by 4)
`results/predictions/validation_predictions.csv.gz` and `results/predictions/test_predictions.csv.gz`,
long format with columns:
`cohort` (all/young), `train_scope` (all/young), `model`, `feature_set`, `year`, `player_api_id`,
`score` (higher = more likely to break out), `breakout`.
Include the EA baseline as `model="ea_gap"`, `feature_set="ea_gap_only"`, `score=potential_gap`
(it needs no training; its rows are repeated under each `train_scope` used for that cohort).
Workstream 3 also writes `results/modelling/best_settings.csv` and `results/modelling/validation_summary.csv`.
Plus `results/modelling/validation_scores.csv` (per fold metrics for every candidate) and
`results/modelling/selected_models.json` (the chosen model, settings and feature set per cohort).
`src/modelling.py` exposes `build_pipeline(model, params, feature_set)` returning an unfitted
Pipeline, so workstream 4 can refit chosen models for permutation importance and sensitivity checks.
