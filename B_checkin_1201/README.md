# Predicting Breakout Football Players — Check-in 2 (1 December)

Team: Shiyan Wang, Junius, Noah Feng, Xianghe Meng · IDSC-GH 5010, Fall 2026

Every 1 August we want to rank football players so that a scout who can only follow 10% of them
finds as many "breakouts" as possible (players whose FIFA rating jumps by 3+ points in the next
year). This folder contains the whole project so far, from cleaning to the final test, and runs on
its own.

## What is in this folder

| Check-in item | Where |
|---|---|
| Model comparisons | `03_modelling.ipynb`, `tables/best_settings.csv`, `figures/validation_precision_by_year.png` |
| Validation | rolling years 2011–2013 in `03_modelling.ipynb`; one final test on 2014 in `04_evaluation.ipynb` |
| Feature explanations | `figures/permutation_importance_groups.png`, `figures/logistic_coefficients.png`, `figures/feature_set_comparison.png` |
| Draft report | `report_draft.md` |
| Draft slides | `slides_outline.md` |

```
01_data_cleaning.ipynb          cleaning and the checks we ran
02_exploration_features.ipynb   plots, correlations, new features
03_modelling.ipynb              choosing the model on 2011-2013
04_evaluation.ipynb             final test on 2014, explanations, robustness checks
report_draft.md                 draft report (all six parts)
slides_outline.md               plan for the slides
src/                            the code the notebooks call
data/ tables/ figures/ predictions/   outputs (rebuilt when you run the notebooks)
```

## How to run

1. Download `database.sqlite` from https://www.kaggle.com/datasets/hugomathien/soccer and put it in
   `data/` (or set the environment variable `SOCCER_DB` to where it is).
2. Install the packages (Python 3.11 or newer): `pip install -r requirements.txt`
3. Run the four notebooks in order (about 12 minutes in total), in Jupyter or with:

```sh
jupyter nbconvert --to notebook --execute --inplace 01_data_cleaning.ipynb
jupyter nbconvert --to notebook --execute --inplace 02_exploration_features.ipynb
jupyter nbconvert --to notebook --execute --inplace 03_modelling.ipynb
jupyter nbconvert --to notebook --execute --inplace 04_evaluation.ipynb
```

## Main result

On the 2014 test year, watching the top 10%:

| | Our model | EA's potential gap | Random |
|---|---:|---:|---:|
| All players (watch 362) | 169 breakouts (46.7%) | 134 (37.0%) | about 49 (13.5%) |
| Young players (watch 138) | 76 (55.1%) | 63 (45.7%) | about 33 (24.1%) |

The model does not need EA's potential to beat EA's list. The extra information comes from match
history: playing time and how a player compares with his team.
