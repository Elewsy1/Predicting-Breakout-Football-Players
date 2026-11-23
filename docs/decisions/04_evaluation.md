# Decisions — workstream 4 (final evaluation, explanations, results)

Owner files: `src/evaluation.py`, `notebooks/04_evaluation_explanations.ipynb`, `figures/evaluation/`,
`results/evaluation/`, `report/04_evaluation_results.md`, this file.
Numbers quoted here come from the run whose outputs are in `results/evaluation/` (see "Status of
inputs" at the end). They will move slightly if upstream files change and the notebook is re-run.

### D04-1: Which model is "the selected model" for each cohort?
- Options considered: re-select from `validation_scores.csv` myself; take workstream 3's choice.
- Decision: take the `primary` entry per cohort in `results/modelling/selected_models.json`
  (final run: all players — logistic regression, `combined` features, C=0.1, no class weights, 3 skill
  PCs, trained on all players; young — HistGradientBoosting, `combined`, lr 0.03, 3 leaves, balanced
  class weights, trained on young players only).
  `load_selection` also accepts a flat or list format.
- Why: model choice belongs to workstream 3 and was made on validation years only; re-selecting here
  would risk choosing with the test year in view.
- Impact if reversed: none on code; any other entry can be evaluated by editing the selection.
- Needs team confirmation: no

### D04-2: What is the random baseline?
- Options considered: one simulated random list; the expected value; many simulated lists.
- Decision: expected value (precision = prevalence, hits = k × prevalence). For the scouting
  simulation the 95% range of a random list's hits comes from the exact hypergeometric distribution
  (N players, B breakouts, k draws). In bootstrap replicates the random value is the replicate's
  prevalence, so "model − random" differences are paired too.
- Why: exact, seed-free, and what the proposal calls the "prevalence" baseline.
- Impact if reversed: none on point estimates; a single random draw would only add noise.
- Needs team confirmation: no

### D04-3: How is uncertainty estimated?
- Options considered: row bootstrap; player-clustered bootstrap; analytic binomial intervals.
- Decision: player-clustered bootstrap — resample player IDs with replacement (all of a player's rows
  move together), 1,000 replicates, `SEED`=42, percentile 95% intervals. Watchlist size is recomputed
  as floor(10%) of each resampled cohort-year. Validation periods report the mean of per-year metrics
  over 2011–13; the test period is 2014 only (each player appears once, so it equals a row bootstrap).
  The same draws are used for every model, so differences between models (vs EA, between feature sets)
  are paired. `share_reps_not_better` = share of replicates where the difference is ≤ 0 (a one-sided
  bootstrap p-value).
- Why: proposal asked for bootstrapping players because players recur across years; pairing removes
  the shared year-to-year noise and makes model comparisons much tighter than separate intervals.
- Impact if reversed: row bootstrap would give slightly narrower (over-confident) validation intervals.
- Needs team confirmation: no

### D04-4: Tie-breaking for EA's integer potential gap
- Options considered: `rank_order` (lower player ID wins ties); random tie-breaking; expected value.
- Decision: use `src.common.rank_order` everywhere (the shared contract). Checked the effect: in 2014
  the 10% cut falls inside a tie (all players: gap = 10, 113 of 117 tied players needed); random
  tie-breaking would give 134.8 expected hits vs 134 with ID order (young: 63.5 vs 63).
- Why: one rule for all strategies; the difference is under one hit.
- Impact if reversed: < 1 hit for EA in 2014.
- Needs team confirmation: no

### D04-5: How to describe list overlap and "union"
- Decision: report players on both lists, model-only, EA-only, and the union. The union is labelled
  "uses up to 2× the budget" and is not a fair competitor; it only shows the ceiling of combining.
- Why: the overlap explains where the extra hits come from (the model-only picks hit far more often
  than the EA-only picks).
- Needs team confirmation: no

### D04-6: Permutation importance — which data, which scoring, which features
- Options considered: (a) 2014 test year with the final model; (b) training rows; (c) each rolling
  validation fold (fit on years < v, permute on year v) — scoring by Precision@10%, average precision
  or log-loss; one column at a time vs groups.
- Decision: (c) as the primary estimate (mean over 2011, 2012, 2013, with the fold range), scored by
  **average precision** (Precision@10% drop also reported). Columns are shuffled **jointly in groups**
  (EA potential; overall rating; 33 skills; rating history; age; body & EA tags; league; playing
  time; team context; position; line-up coverage), 10 repeats; single-column importance is also
  saved. The 2014 importance (final 2009–13 refit, scored on 2014) is shown as a descriptive check
  only. No model is ever refitted on 2014.
- Why: validation folds keep the test year untouched and show stability across years; AP uses the
  whole ranking and is far less noisy than a 10% cut of ~370 players; joint shuffling avoids the
  well-known understatement of correlated features (33 skills, rating/potential/gap).
- Impact if reversed: rankings of the top groups are the same on 2014 (see figure), so low.
- Needs team confirmation: no

### D04-7: Which logistic regression gives the coefficients?
- Options considered: a separate unpenalised LR on raw features; workstream 3's tuned LR pipeline.
- Decision: workstream 3's pipeline (`build_pipeline`) with its tuned setting, fitted on 2009–13
  (all players: the selected model itself; young: the best validated logistic regression for the
  young cohort/scope/feature set, because the selected young model is gradient boosting — labelled as
  a stand-in). Where the setting uses skill PCA, coefficients are for the components and the loadings
  are saved; component signs are aligned to the full fit in every bootstrap replicate. CIs: 200
  player-bootstrap refits. The figure shows numeric (per-SD) terms only; league/position dummies are
  in the table.
- Why: explains the model actually evaluated, not a different one.
- Impact if reversed: coefficients on raw skills would be unstable (collinear) and harder to read.
- Needs team confirmation: no

### D04-8: How is the circularity question answered?
- Decision: within the selected model family, paired bootstrap contrasts of feature sets on both
  validation (2011–13, basis for conclusions) and test (2014, confirmation):
  `rating_no_potential − EA gap`, `rating_only − rating_no_potential` (value of EA potential),
  `combined − rating_only` (value of match history), `combined − rating_no_potential`. Plus the
  EA-independent next-season-starts check (D04-13).
- Why: if the gain over EA came only from EA's own potential judgement, the model without potential
  would not beat the EA list and adding potential would add a lot. Results: see report.
- Needs team confirmation: yes (the team should agree this is the framing used in the report)

### D04-9: Max-rating label (sensitivity a)
- Options considered: keep the main yearly median; recompute the median on the max-based change.
- Decision: change_max = (max overall rating in [1 Aug y, 1 Aug y+1)) − current rating; breakout_max =
  change_max ≥ yearly median of change_max + 3. Rows are unchanged (same players need ≥1 rating in the
  window). The main label is recomputed from the database as a check, using the same duplicate and
  isolated-spike filters as `process_data.py`; it matches the cleaned table on 100% of rows
  (`sensitivity_label_stats.csv`). Because the median of the max-based change is 1 (not 0) in 2009,
  2011 and 2012, this label differs from the main one on 822 of 21,631 rows (2009–2014); with the
  median kept at the main label's value the coordinator's earlier count was 427.
- Why: keeps the definition "unusual relative to that year" — taking the max shifts every change
  upwards, so the old median would inflate prevalence.
- Impact if reversed: prevalence would be higher; ranking comparisons barely change.
- Needs team confirmation: no

### D04-10: Short-horizon rows (sensitivity b)
- Decision: horizon = outcome_date − rating_date (workstream 1's `horizon_days` used when present);
  rows with horizon < 180 days are dropped from **both** training and evaluation; the label and the
  yearly median are left unchanged.
- Why: these rows mostly cannot break out (too little time), so they are easy negatives for every
  ranking; dropping them everywhere asks whether conclusions hold on comparable horizons.
- Needs team confirmation: no

### D04-11: Which other sensitivity analysis (c)?
- Options considered: breakout threshold ±1; selection-bias bound for the 11.48% without follow-up;
  dropping leagues with weak line-up coverage; training on recent years only.
- Decision: threshold median + 2 and median + 4 (the "+3" is the most arbitrary modelling choice and
  changes prevalence from ~8% to ~22% in 2014). The others are listed under "Open issues".
- Why: cheap, directly addresses the label definition, and tests the ranking at very different
  prevalences. Recent-years training was already checked by workstream 3 (`window_check`).
- Needs team confirmation: yes

### D04-12: Re-tune under each sensitivity scenario?
- Decision: no — reuse the selected setting and feature set, refit on each scenario's labels/rows
  (rolling folds 2011–13 and 2009–13 → 2014), and compare with EA's list under the same label.
- Why: tests whether the *chosen* pipeline's advantage is robust; re-tuning would mix robustness with
  re-selection and multiply compute.
- Impact if reversed: re-tuned models could only do better, so this is conservative for the model.
- Needs team confirmation: no

### D04-13: EA-independent check — next-season starts
- Decision: count league starts in season 2014/15 from the Match line-ups (all 22 slots) for every
  2014 cohort player; players with no line-up appearance (left the 11 leagues, injured, benched) count
  as 0. Report raw differences (listed − not listed) and an OLS-adjusted difference controlling for
  previous-season starts, age and current rating, with a 1,000-replicate row bootstrap. 2014/15
  line-up coverage is ≥ 92.5% of matches in every league (Poland lowest).
- Why: an outcome EA does not set; adjustment is needed because both lists favour young players and
  players whose playing time is already changing.
- Impact if reversed: descriptive only; not used for any choice.
- Needs team confirmation: yes (whether to show it in the main report or the appendix)

### D04-14: Work-rate columns arrive as text (upstream conflict, workaround)
- Context: at 11:34 workstream 1 added raw `attacking_work_rate`/`defensive_work_rate` strings to the
  cleaned table; `src.features` keeps cleaned columns as they are, so the table briefly held text and
  `build_pipeline` failed. Workstream 2 has since switched its sets to `*_work_rate_ord` columns.
- Decision: in `feature_resources`, if those two columns are non-numeric, map low/medium/high → 0/1/2
  (the encoding `src.features` intends). Harmless when they are already numeric or unused.
- Why: keeps evaluation running while upstream settles; logged to the coordinator.
- Update: workstream 1 removed those columns again (final cleaned table 25,502 × 63) and workstream 2
  uses `*_work_rate_ord`, so the workaround no longer triggers in the final run.
- Needs team confirmation: no (can be deleted)

### D04-15: Do refits describe the evaluated model?
- Decision: `step_consistency` refits each selected pipeline on 2009–13 and compares its 2014 scores
  with `test_predictions.csv.gz` (Spearman, max score difference, watchlist overlap); a mismatch is
  printed as a NOTE and saved to `refit_consistency.csv`. All headline test metrics come from the
  prediction files, never from my refits.
- Why: explanations and sensitivity runs use refits; the team must know whether they match.
- Needs team confirmation: no

### D04-16: Precision at other budgets
- Decision: show precision for budgets 1–30% in 2014 as a robustness view; it is computed after the
  fact and not used to choose anything (the 10% budget is fixed by the proposal).
- Needs team confirmation: no

### D04-17: Placeholder model
- Decision: if workstream 3's files are missing, `load_context` builds the EA baseline plus an L2
  logistic regression (C=0.1) on the base feature sets and writes only to
  `results/evaluation/placeholder/` and `figures/evaluation/placeholder/`, with "[PLACEHOLDER MODEL]"
  in every figure title. Real predictions existed before the first full run, so no placeholder outputs
  are kept (the code path was tested and its test outputs removed).
- Needs team confirmation: no

### D04-18: Headline figure
- Decision: hits among the 2014 watchlist (with 95% CIs) for our model, EA's list and a random list,
  side by side for all and young players; overlap composition as the second figure.
- Why: "how many of the players you watch actually break out" is the scout's question and needs no
  metric vocabulary.
- Needs team confirmation: yes (presentation choice)

## Requests to coordinator
1. Whenever the cleaned table, `src/features.py` or workstream 3's predictions change again, re-run
   this notebook (`jupyter nbconvert --to notebook --execute --inplace
   notebooks/04_evaluation_explanations.ipynb`, ~7 min) and check `results/evaluation/refit_consistency.csv`
   shows `reproduced = True`. (The work-rate conflict reported earlier is resolved.)
2. Add `horizon_days` to `LEAKY_COLUMNS` in `src/common.py` (workstream 1 asked for this too); this
   module only uses it to filter rows in sensitivity (b), never as a feature.
3. `process_data.py`'s spike filter is duplicated in `rating_history()` here (thresholds 8 / 2). If
   workstream 1 changes them, this copy must follow; a shared helper in `src/common.py` would avoid it.

## Open issues
- **Selection-bias bound not computed.** The 11.48% of candidates without a follow-up rating are
  excluded from every metric. A worst-case bound (score them, count them as non-breakouts) needs
  features for rows that are not in the cleaned table.
- **Line-up missingness sensitivity not run** (e.g. dropping Poland/Belgium and 2008/09 line-ups).
- **Single test year.** 2014 is one draw; validation years show precision drifting downwards
  (2011 → 2013) along with prevalence, so the 2014 numbers are not a forecast for later years.
- Inputs were still changing during this run (see "Status of inputs" in the report); final numbers
  require the re-run described in Request 1.
