# Check-in 2 (1 Dec) · Model comparisons, validation, feature explanations, draft report and slides

Bring everything in `A_checkin_1124/` plus this folder.

| Requirement | Where |
|---|---|
| Model comparisons | [03_modelling.ipynb](03_modelling.ipynb); [tables/best_settings.csv](tables/best_settings.csv), [tables/validation_summary.csv](tables/validation_summary.csv) |
| Validation | Rolling years 2011–2013 ([tables/validation_scores.csv](tables/validation_scores.csv)); final 2014 test in [04_evaluation_explanations.ipynb](04_evaluation_explanations.ipynb), [figures/precision_by_year.png](figures/precision_by_year.png) |
| Feature explanations | [figures/permutation_importance_groups.png](figures/permutation_importance_groups.png), [figures/logistic_coefficients.png](figures/logistic_coefficients.png), [figures/feature_set_comparison.png](figures/feature_set_comparison.png) |
| Scouting simulation (the action) | [figures/headline_scouting_2014.png](figures/headline_scouting_2014.png), [tables/scouting_summary_2014.csv](tables/scouting_summary_2014.csv) |
| Sensitivity checks | [figures/sensitivity.png](figures/sensitivity.png), [tables/sensitivity_summary.csv](tables/sensitivity_summary.csv) |
| Draft report | [report_draft.md](report_draft.md) (all six course parts) |
| Draft slides | [slides_outline.md](slides_outline.md) (outline; deck still to build) |

Code: [`src/modelling.py`](../src/modelling.py) (selection, ~2 min) and
[`src/evaluation.py`](../src/evaluation.py) (evaluation, ~5 min). Reproduce from the repository root
after check-in 1's steps:

```sh
jupyter nbconvert --to notebook --execute --inplace B_checkin_1201/03_modelling.ipynb
jupyter nbconvert --to notebook --execute --inplace B_checkin_1201/04_evaluation_explanations.ipynb
```

Folders: `predictions/` holds the validation and 2014 scores of every best model (input to the
evaluation), `tables/` every result table and `selected_models.json`, `figures/` the report figures.

## Results in one table (2014 test year, watch 10% of the cohort)

| Cohort | Our model | EA potential gap | Random | Model − EA [95% CI] |
|---|---:|---:|---:|---:|
| All players (watch 362) | 169 hits (46.7%) | 134 (37.0%) | 49 (13.5%) | +9.7 pp [4.1, 14.4] |
| Young ≤ 24 (watch 138) | 76 hits (55.1%) | 63 (45.7%) | 33 (24.1%) | +9.4 pp [1.4, 18.8] |

Selected on 2011–2013 only: logistic regression (all players) and gradient boosting trained on young
players (young cohort), both on ratings + EA potential + match history.
