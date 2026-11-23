# Decisions · Workstream 2 (exploration, correlations, feature engineering)

Code: `src/features.py`. Narrative: `notebooks/02_exploration_features.ipynb`. Figures: `figures/eda/`.
Tables: `results/features/`. All exploration and screening use development years 2009–2013 only;
2014 labels were never looked at.

### D02-1: Which columns go into which feature set?
- Options considered: (a) put `league` and `position` in every set; (b) treat both as match history;
  (c) `league` in every rating set, `position` only in `combined`.
- Decision: (c). `rating_no_potential` = current overall + 33 skills + rating history + player
  attributes + `league` (44 columns). `rating_only` = that + `potential`, `potential_gap`,
  `potential_change_prev_year` (47). `combined` = `rating_only` + match history + `position` (58).
  `ea_gap_only` = `["potential_gap"]`.
- Why: the league of a player's club is also in EA's own database, so it belongs with "what the
  rating data alone knows". Position here is derived from line-up pitch coordinates, so it is match
  history. Skills already encode position strongly for the rating-only sets.
- Impact if reversed: small; position adds < 0.5 points Precision@10% on top of `rating_only`
  in the screening (`+position` variant, ~0.44 → 0.44 logistic, 0.47 → 0.47 HGB).
- Needs team confirmation: yes

### D02-2: How to define rating momentum and other rating-history features?
- Options considered: change over the previous 12 months using the latest rating in
  [cut-off−2y, cut-off−1y) (mirrors the label's windows); change since the first rating in the
  current window; slope over all past ratings.
- Decision: keep both of the first two: `rating_change_prev_year` (current overall − latest overall
  in [cut-off−2y, cut-off−1y)) and `rating_change_in_window` (current overall − earliest overall in
  [cut-off−1y, cut-off)); plus `potential_change_prev_year` and a `has_prev_rating` flag. Differences
  are taken against the cleaned table's own `overall_rating`.
- Why: same window logic as the label, easy to explain, no look-ahead. The flag is the strongest of
  the four (players new to EA's data break out 40% vs 15%).
- Impact if reversed: removing all four costs 0–1.5 points in the screening; low risk either way.
- Needs team confirmation: no

### D02-3: How to treat isolated rating spikes in the rating history?
- Options considered: ignore; reuse workstream 1's rule on all ratings; apply the same rule but
  only to ratings dated before each cut-off.
- Decision: same rule as `process_data.py` (≥ 8 points from both neighbours that agree within 2),
  applied separately per cohort year to ratings dated before that year's cut-off.
- Why: consistency with the cleaned current rating without letting the filter peek at
  post-cut-off ratings. The two approaches disagree on the current rating for 1 of 25,502 rows.
- Impact if reversed: negligible.
- Needs team confirmation: no

### D02-4: Use height and weight even though the `Player` table has no date?
- Options considered: drop them; include them.
- Decision: include `height_cm` and `weight_kg` (lb → kg, rounded to 0.1), always derived here
  from the `Player` table (see D02-16).
- Why: adult height is stable; weight changes little. The values are probably a snapshot from the
  end of the data (2015–16), a mild look-ahead mainly for teenagers. They show no association with
  breakout (Spearman ≈ 0), so the risk of leaked signal is small; models may still use them in
  interactions.
- Impact if reversed: screening shows ≤ 1.5 points change together with foot/work rates.
- Needs team confirmation: yes (the static-snapshot caveat should appear in the report's limitations)

### D02-5: How to encode EA's work rates and preferred foot?
- Options considered: one-hot categories; ordinal 0/1/2; drop.
- Decision: ordinal `attacking_work_rate_ord`, `defensive_work_rate_ord` (low/medium/high → 0/1/2,
  every other code such as `None`, `norm`, `_0`, digits → missing, then median-imputed);
  `left_footed` = 1/0. Always derived here from the current rating record, never from text
  columns in the cleaned table; new names so they never clash with those (see D02-16).
- Why: the categories are ordered; one numeric column is simpler for every model. Missing codes are
  16% of 2009 rows and < 1% later.
- Impact if reversed: none on results (no marginal association); only interpretation.
- Needs team confirmation: no

### D02-6: How to handle previous-season starts for 2009 and for players absent last season?
- Options considered: NaN for both; 0 for both; 0 if the season exists, NaN for 2009.
- Decision: `prev_starts` = 0 when the player did not start in the previous season (season in the
  database); NaN for every 2009 row (season 2007/08 is missing) with `has_prev_season = 0`;
  `starts_change = starts − prev_starts`.
- Why: absence from the previous season's line-ups is real information (newcomers, players from
  other leagues). For 2009 nothing is known, so median imputation plus the flag is honest.
  `has_prev_season` is constant within a year (0 only in 2009), so it only lets models separate the
  imputed 2009 values in training; it is always 1 in validation/test.
- Impact if reversed: affects only 2009 training rows.
- Needs team confirmation: no

### D02-7: How to compute squad context (team mean rating, rank in team)?
- Options considered: from the cleaned rows only; from all starters of the season with a current
  rating.
- Decision: from all starters of the season just finished who have a current rating, before any
  follow-up filter. `team_mean_rating` = mean current overall of the main club's rated starters;
  `team_rating_rank_pct` = the player's percentile within them (1 = best).
- Why: the cleaned rows require a *future* rating to exist, so using them would leak follow-up
  availability into a feature. This matches how `rating_vs_teammates` was built.
- Impact if reversed: small numeric differences; possible subtle leakage.
- Needs team confirmation: no

### D02-8: Keep the rating-calendar features `n_rating_updates_1y` and `rating_age_days`?
- Options considered: include; exclude.
- Decision: compute and keep them in the table, but leave them out of every feature set
  (`EXCLUDED_ENGINEERED`).
- Why: EA switched from two updates a year to near-monthly updates in 2013. The mean number of
  updates jumps from ~1.9 (2012) to ~4.4 (2013); the days since the last rating drop from ~175 to ~95.
  Their within-year correlation with breakout fades (`rating_age_days`: +0.06 → 0.00) or flips sign
  (`n_rating_updates_1y`: −0.10 → +0.02). They mostly encode the year and calendar, which the models
  cannot extrapolate to 2014. `rating_age_days` is also part of the outcome horizon. Adding them in
  the screening does not help.
- Impact if reversed: likely unstable models in 2013–2014; no gain in validation.
- Needs team confirmation: yes

### D02-9: Keep features with no marginal association (height, weight, foot, work rates)?
- Options considered: drop to keep sets small; keep.
- Decision: keep in the rating sets.
- Why: cheap, no leakage risk, possible interactions for trees; removing them changes validation
  Precision@10% by −1.5 to +1.3 points depending on model and cohort, within noise.
- Impact if reversed: none material.
- Needs team confirmation: no

### D02-10: Which of workstream 1's new coverage columns become features?
- Options considered: all; none; selected.
- Decision: `club_lineup_coverage` goes into `combined` (it tells a model when `starts` is
  undercounted). `league_match_coverage` is excluded (constant in 4 of 5 development years).
  `position_missing` is excluded (identical to the one-hot `position=Unknown` level).
  `horizon_days` is never a feature (`NEVER_FEATURES`; `make_preprocessor` refuses it).
- Why: as above. In screening, the flags change results by < 1 point either way.
- Impact if reversed: negligible.
- Needs team confirmation: no

### D02-11: Preprocessing (`make_preprocessor`)
- Options considered: mean vs median imputation; scaling for all models vs linear only; one-hot vs
  ordinal for league/position; PCA always vs optional.
- Decision: median imputation (`keep_empty_features=True` so a fold without values does not drop a
  column); standard scaling only when `scale=True` (linear models); `league`/`position` one-hot with
  missing → `"Unknown"` and `handle_unknown="ignore"`; dense output; optional PCA on the skill
  columns (`pca_skills` = int components or float variance share), always on standardised skills.
  Fitted only inside the model pipeline on training rows of each fold.
- Why: skewed counts and a few missing skills favour the median; trees are scale-invariant;
  leagues are nominal; 6 components explain 80% and 12 explain 90% of skill variance, so PCA is a
  useful option for logistic regression but not required.
- Impact if reversed: minor.
- Needs team confirmation: no

### D02-12: Unknown position is almost entirely Poland
- Options considered: impute position from skills; drop Poland; keep `Unknown` as its own level.
- Decision: keep `Unknown` as a level and say clearly in the report that it is a league artefact
  (801 of 808 development rows are Polish; Polish line-ups lack pitch coordinates).
- Why: honest about missingness and costless for trees; skills carry position information anyway.
- Impact if reversed: imputing from skills would add modelling complexity for < 5% of rows.
- Needs team confirmation: no

### D02-13: Correlation measure for the heatmap
- Options considered: Pearson; Spearman.
- Decision: Spearman for the feature heatmap (counts and rating changes are skewed, many variables
  are ordinal); Pearson for the skill matrix (comparable 0–100 scales) and reported alongside AUC in
  `univariate_association.csv`.
- Why: robustness to skew and outliers; the check-in used Pearson, and the two agree in sign.
- Needs team confirmation: no

### D02-14: Age variable in the feature sets
- Options considered: `age` (completed years, used for the cohort) and/or `age_years` (continuous).
- Decision: `age_years` only.
- Why: the two are almost identical; one avoids redundancy. `age` remains in the table for cohorts.
- Needs team confirmation: no

### D02-15: Feature-group screening on the validation folds
- Options considered: choose features only from univariate statistics; run a light screening.
- Decision: a light screening (`screen_feature_groups`) with two fixed, untuned models (L2 logistic
  regression, default HistGradientBoosting) on the rolling folds 2011–2013. It decides only whether
  a group enters a set, not models or hyperparameters (workstream 3).
- Why: the course asks to choose features with validation data; using fixed models keeps this
  separate from model selection. Key results (mean Precision@10%, all / young):
  `ea_gap_only` 0.41 / 0.54; `rating_no_potential` 0.44–0.47 / 0.56–0.58; `rating_only` 0.44–0.45 /
  0.57; `combined` 0.52–0.54 / 0.65.
- Impact if reversed: none on the interface.
- Needs team confirmation: no

### D02-16: What if the cleaned table already has a column with a feature's name?
- Options considered: keep the cleaned version; always derive our own and overwrite.
- Decision: always derive our own (coordinator ruling, 23 Nov: workstream 2 owns all feature
  engineering). `build_feature_table` drops any same-named cleaned column before merging its own
  version (`height_cm`, `weight_kg`, `rating_age_days`, the work-rate/foot encodings, rating
  history, squad context, previous starts). Raw EA text columns (`preferred_foot`,
  `attacking_work_rate`, `defensive_work_rate`) are passed through untouched but are in no feature
  set. `check_feature_types` raises if any feature-set column is missing or if a non-categorical
  one is not numeric; it runs on build and on every cache load. A module-level assertion keeps
  `horizon_days`, `position_missing`, excluded, leaky and ID columns out of every `FEATURE_SETS` entry.
- Why: an earlier cleaned table had raw work-rate strings under the feature names, and that broke
  every non-EA pipeline downstream. Tested with a deliberately bad layout (text values in
  `height_cm`, `left_footed`, `*_work_rate_ord`, `starts_change`): the output is identical to the
  normal build.
- Impact if reversed: downstream breakage whenever the cleaning layout changes.
- Needs team confirmation: no

## Requests to coordinator

1. Add `horizon_days` (and `position_missing` if desired) to a forbidden list in `src/common.py`
   (e.g. `LEAKY_COLUMNS` or a new `NON_FEATURE_COLUMNS`). Meanwhile `src/features.py` refuses them via
   `NEVER_FEATURES`.
2. Optional: point `src.common.BASE_*` to `src.features.FEATURE_SETS` in a comment, so nobody keeps
   using the base lists now that the module exists.
3. `club_lineup_coverage` is the one feature taken from the cleaned table rather than derived here.
   If a future cleaned table drops it, `build_feature_table` raises a clear error instead of
   silently filling it. Please keep that column stable.

## Open issues

1. `height_cm`/`weight_kg` come from an undated snapshot (D02-4); mention in limitations.
2. EA's 2013 switch to near-monthly updates also changes the label: the "latest rating in window"
   is fresher from 2013 onward, which may explain part of the drop in breakout rate in 2013 (11%).
   This is a label/cleaning question for workstreams 1/3, not a feature question.
3. `club_lineup_coverage` and `position=Unknown` partly act as league indicators (Poland, Belgium);
   their importance should not be read as a causal effect of line-up data quality.
4. The match-history gain comes mostly from `starts`/`start_share` conditional on rating and age
   (fewer starts at a given rating → more breakouts). Workstream 4 should check this is not driven by
   the follow-up selection (players who stop being rated are excluded).
