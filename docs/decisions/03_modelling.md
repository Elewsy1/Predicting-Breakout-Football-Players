# Decisions — workstream 3 (methods, model selection, validation)

Code: `src/modelling.py` (`python -m src.modelling` reruns everything in ~2.5 min on 24 cores).
Notebook: `notebooks/03_modelling.ipynb`. Outputs: `results/modelling/`, `results/predictions/`.
All numbers below are validation-year (2011–2013) numbers; no 2014 metric was computed by this workstream.

### D03-1: Which feature definitions to use?
- Options considered: fallback lists from `src.common` (`BASE_*`) with a local preprocessor; the real `src/features.py`.
- Decision: `feature_api()` uses `src.features` whenever it imports cleanly and satisfies the contract (four set names,
  no leaky/ID columns), otherwise the fallback with the same set names. Every output records `features_source`.
  **The final run used the real `src/features.py`** (it landed before the first full run; the fallback was never used for results).
- Why: the WORKPLAN contract; a half-written module must not crash selection.
- Impact if reversed: none on code; feature lists changed by workstream 2 require a rerun (see Open issues).
- Needs team confirmation: no

### D03-2: Which model families and which fixed settings?
- Options considered: logistic regression, random forest, `HistGradientBoostingClassifier` (all covered in class);
  k-NN / SVM (slow on 18k rows, need careful scaling, weaker on tabular data); lightgbm/xgboost (not allowed).
- Decision: the three named families. Fixed: logistic regression `max_iter=5000` (lbfgs, L2); random forest
  300 trees; boosting 300 iterations, `min_samples_leaf=40`, `l2_regularization=1`, `early_stopping=False`.
  Seeds `src.common.SEED`.
- Why: a linear, interpretable reference plus two non-linear ensembles, as in the proposal. More trees only reduce
  variance, so tree count is not tuned. Boosting's early stopping uses an internal *random* split, which ignores time
  order and repeated players; the learning-rate grid controls fit instead.
- Impact if reversed: small; results are within ~2 points across families.
- Needs team confirmation: no

### D03-3: Hyperparameter grids
- Options considered: large random searches vs small grids.
- Decision: logistic regression `C` ∈ {0.01, 0.1, 1} × class weight {none, balanced} × PCA on skills {none, 3, 5, 10}
  (24); random forest `min_samples_leaf` {5, 25} × `max_features` {√p, 0.33p} × class weight {none, balanced_subsample}
  (8); boosting learning rate {0.01, 0.03, 0.1} × `max_leaf_nodes` {3, 7, 31} × class weight (18). PCA is dropped for
  the `ea_gap_only` set (no skills). 182 candidates over the four feature sets.
- Why: three folds and watch-lists of 140–380 players give Precision@10% a sampling error of about 2.5–4 points per
  year; finer grids would mostly fit noise. The spread across settings within a family is 1–10 points (random forest
  1–4.5, boosting 4–9, logistic regression 6–10).
- Impact if reversed: a bigger grid may gain ~1 point on validation, likely not on test.
- Needs team confirmation: no

### D03-4: Widening the grid after a first run
- Options considered: keep the first grid; widen once where the optimum sat on its edge; keep widening.
- Decision: widened once (boosting learning rate 0.01 and 3 leaves; PCA 3 and 10 components). First-round summary
  kept in `results/modelling/grid_round1_summary.csv`. The young-cohort boosting optimum now sits at 3 leaves (new edge);
  we stopped there because neighbouring settings are within ~1 point.
- Why: an edge optimum suggests the grid missed the best region; but repeated widening on the same three folds overfits
  the validation years. All of this used validation years only.
- Impact if reversed: about ±1 point on the selected models' validation precision (round 1 also used an earlier
  version of the features, so the two rounds are not exactly comparable).
- Needs team confirmation: no

### D03-5: Class weighting
- Options considered: none; balanced weights; resampling.
- Decision: class weight is a grid option in every family; no resampling.
- Why: prevalence is 11–26%. Result: weighting changes mean Precision@10% by < 1 point (−0.8 to +0.4), as expected
  because the metric only uses the within-year ranking.
- Impact if reversed: negligible.
- Needs team confirmation: no

### D03-6: Preprocessing per family
- Options considered: one preprocessor for all; scaling for all.
- Decision: `make_preprocessor(cols, scale=(model == "logreg"), pca_skills=...)` from `src.features`, inside the Pipeline,
  so it is fitted on each fold's training rows only. Trees get the same imputation and one-hot encoding, no scaling.
- Why: trees are scale-invariant; one shared preprocessor keeps the feature sets identical across families.
- Impact if reversed: none for trees.
- Needs team confirmation: no

### D03-7: Selection rule and aggregation over folds
- Options considered: pooled Precision@10% over the three years; mean of per-year values; worst year; AP as primary.
- Decision: mean of the three per-year Precision@10% values (each year weighted equally), tie-break by mean average
  precision. Applied per (cohort, train scope, family, feature set) to pick the setting, and then per cohort over all
  candidates to pick the primary model.
- Why: the budget is set per year, prevalence differs by year, and pooling would let high-prevalence years dominate.
- Impact if reversed: the primary choices could swap among near-tied candidates (D03-8).
- Needs team confirmation: no

### D03-8: Primary model per cohort despite near-ties
- Options considered: plain maximum; prefer the simplest model within one standard error; pick logistic regression for
  interpretability.
- Decision: plain maximum, as declared before looking at results. **All:** logistic regression, `combined`,
  `C=0.1`, no class weight, PCA with 3 skill components — mean P@10% 57.9% (69.3 / 61.2 / 43.3) vs EA gap 41.3%
  and random 16.6%.
  **Young:** gradient boosting trained on young players, `combined`, learning rate 0.03, 3 leaves, balanced —
  70.2% (83.6 / 70.8 / 56.3) vs EA gap 54.3% and random 28.1%. The next young candidates (logistic regression
  trained on all players 70.0%, on young players 69.5%) are within 1 point.
- Why: a fixed rule avoids choosing after the fact; near-ties are reported honestly instead.
  The feature-set ranking (combined > rating-only ≈ rating without potential > EA gap) is robust; the family choice is not.
- Impact if reversed: a logistic regression trained on all players costs 0.2 points for young players (70.0%, tuned
  `C=0.01`); the all-cohort model applied unchanged to young players reaches 69.3% (−0.9). One logistic regression is
  simpler and gives coefficients for explanation. Test scores for every family and scope are delivered, so workstream
  4 can show either. The exact all-cohort model's young scores are its `cohort="all"` rows restricted to age ≤ 24
  (join `age` from the cleaned table).
- Needs team confirmation: **yes** (rule-based winner vs one logistic regression for both cohorts as "the" model)

### D03-9: Young cohort trained on all players
- Options considered: separate fits restricted to young rows; reuse the all-player fit.
- Decision: a model trained on all players is fitted once per fold and scores everyone; its young-cohort result is its
  ranking restricted to young players (budget = floor(10%) of the young cohort).
- Why: identical to a separate fit, at half the cost.
- Impact if reversed: none.
- Needs team confirmation: no

### D03-10: Train only on the last N years?
- Options considered: expanding window only; also test a sliding window of the last 2 years; tune N.
- Decision: tested N = 2 for each family's best setting (folds 2012 and 2013 only; 2011 already uses two years) with a
  rule fixed in advance: adopt only if the mean gain > 1 point and > 75% of (candidate, fold) pairs improve.
  Result: mean −0.96 points, 30% of pairs improve → **keep the expanding window**; final refit on 2009–2013.
- Why: prevalence drift and EA's 2013 switch to more frequent updates could favour recent data, but losing half
  the training rows costs more.
- Impact if reversed: final models would be fitted on 2012–2013 only.
- Needs team confirmation: no

### D03-11: Stacking years and prevalence drift
- Options considered: year feature; per-year re-weighting; stacking with no adjustment.
- Decision: stack rows across years, no `year` feature, no re-weighting.
- Why: predictions are only used to rank within a year, so a level shift in base rate does not matter; a `year`
  feature cannot be extrapolated to an unseen test year.
- Impact if reversed: little on rankings; scores are not calibrated probabilities (see Open issues).
- Needs team confirmation: no

### D03-12: Repeated players across years
- Options considered: grouped CV by player; time-ordered folds.
- Decision: time-ordered folds only. A player can appear in training (earlier years) and validation (later year),
  just as in deployment; a validation row's own label is never in its training data.
- Why: the scouting use case predicts known players' futures. Player-level dependence matters for uncertainty
  (workstream 4's player bootstrap), not for leakage.
- Impact if reversed: grouped folds would answer a different question (new players only).
- Needs team confirmation: no

### D03-13: Baseline rows in the prediction files
- Options considered: EA rows once per cohort with `train_scope="none"`; repeated under every training scope of the cohort.
- Decision: EA rows (`model="ea_gap"`, `feature_set="ea_gap_only"`, `score=potential_gap`) are repeated under each
  training scope of the cohort (all: `all`; young: `all` and `young`), so every (cohort, train_scope) slice contains its
  comparator. Random selection is not in the prediction files (its expected precision is the year's prevalence); it is
  in `validation_scores.csv` as `model="random"`, where `train_scope`/`window` are `none`.
- Why: simple joins for workstream 4.
- Impact if reversed: workstream 4 would filter EA rows differently; take one copy when pooling scopes.
- Needs team confirmation: no

### D03-14: What goes into the prediction files
- Options considered: all 182 × scope candidates; best setting only.
- Decision: the best validated setting per (cohort, train scope, family, feature set) — 9 model rows + EA per cohort/scope
  slice in both validation and test files. Metrics for every setting are in `validation_scores.csv` (per fold) and
  `validation_summary.csv` (means); chosen settings in `best_settings.csv` and `selected_models.json`.
- Why: the contract has no `setting` column; this keeps the comparison feature set × family complete.
- Impact if reversed: none.
- Needs team confirmation: no

### D03-15: Models on `ea_gap_only`
- Options considered: drop them; keep them.
- Decision: keep. A model on one monotone feature reproduces the EA ranking (identical Precision@10%), which validates
  the pipeline; it is never selected.
- Needs team confirmation: no

### D03-16: Test year handling
- Decision: `final_predictions` refits each selected setting on 2009–2013 and writes 2014 scores only; no 2014 metric is
  computed or displayed anywhere in workstream 3. Selection used validation years only.
- Needs team confirmation: no

### D03-17: Feature sets as delivered by workstream 2
- Decision: used unchanged. Note that `combined` adds `position` (lineup-derived) as well as the numeric match-history
  features, so "+ match history" includes position.
- Impact if reversed: the + match history contrast could change slightly.
- Needs team confirmation: no

### D03-18: Guard against outcome-window columns (`horizon_days`)
- Options considered: rely on workstream 2's feature lists; enforce a block list in workstream 3.
- Decision: `src/modelling.py` refuses any feature set containing `common.LEAKY_COLUMNS`, `ID_COLUMNS`, the target, or
  `horizon_days` (`EXTRA_FORBIDDEN`). The check runs when the feature sets are loaded and again in `build_pipeline`.
  A violation in the real `src/features.py` raises `LeakyFeatureError`; it does not fall back to the default sets.
- Why: workstream 1 added `horizon_days` (time to the follow-up rating, so it depends on the future) to the cleaned
  table. The coordinator asked that it never enters a model. It is in none of the current sets.
- Impact if reversed: none today; it protects later reruns.
- Needs team confirmation: no

## Requests to coordinator
- WORKPLAN prediction contract: please confirm the convention for the EA baseline's `train_scope` (D03-13) and add
  `results/modelling/best_settings.csv` / `validation_summary.csv` to the documented outputs.
- No change needed in `src/common.py`.

## Open issues
- **Rerun after upstream changes.** Workstreams 1 and 2 were still editing the cleaned table and `src/features.py`
  during this work (the cleaned table lost one 2009 row, `combined` gained `club_lineup_coverage`, work-rate
  columns were renamed `*_ord`). The final outputs (run 2026-11-23 11:53) were produced from `checkin/cleaned_player_years.csv.gz`
  mtime 11:39:45, sha1 `7c8ea2871519` (21,631 rows for 2009–2014), and `src/features.py` mtime 11:42:36, sha1
  `9cb5a83fc5fb` (recorded under `inputs` in `selected_models.json`). The provisional 11:33 outputs are fully
  superseded. The coordinator confirmed workstreams 1 and 2 are frozen at these versions, and workstream 4 reproduced
  the 2014 scores exactly, so these outputs are final. If either changes, rerun
  `python -m src.modelling` (~2.5 min) or the notebook; `check_feature_table` fails early on missing or mistyped columns.
- **Harmless joblib noise.** On Windows, nbconvert prints `KeyError: ...joblib_memmapping_folder...` tracebacks from
  loky's resource tracker at shutdown. The notebook still executes fully (exit code 0); no result is affected.
- **Near-ties.** Top candidates per cohort are within ~2 points; the family winner is not a robust result (D03-8).
- **2013 shrinkage.** The margin over EA is smallest in 2013 (all players: +7.1 points for combined logistic regression
  vs +24.9 in 2011), when prevalence fell to 11% and EA moved to more frequent updates. The 2014 gain may be smaller.
- **Scores are not calibrated probabilities** (prevalence drifts by year; class weighting distorts levels). Use them
  only to rank within a year.
- **Selection bias** from dropping players without a follow-up rating (README) affects all strategies equally but
  limits generalisation.
- Leakage spot-check: `rating_vs_teammates` (process_data.py) and the squad features in `src/features.py` use only ratings
  before 1 August and are computed before the follow-up filter. Workstream 4's permutation importance should still
  confirm that no single match feature dominates suspiciously.
