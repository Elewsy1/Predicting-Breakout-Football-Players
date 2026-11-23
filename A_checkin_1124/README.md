# Predicting Breakout Football Players — Check-in 1 (24 November)

Team: Shiyan Wang, Junius, Noah Feng, Xianghe Meng · IDSC-GH 5010, Fall 2026

Every 1 August we want to rank football players so that a scout who can only follow 10% of them
finds as many "breakouts" as possible (players whose FIFA rating jumps by 3+ points in the next
year). This folder has everything for the first check-in and runs on its own.

## What is in this folder

| Check-in item | Where |
|---|---|
| Cleaned data | `data/cleaned_player_years.csv.gz` (25,502 rows), `data/cleaned_preview.csv` (20 rows) |
| Cleaning checks | `01_data_cleaning.ipynb`, `data/cleaning_audit.csv` |
| Plots | `figures/` (made by `02_exploration_features.ipynb`) |
| Correlations | `figures/04_feature_correlation_heatmap.png`, `figures/05_skill_correlation_clustermap.png` |
| First features | `tables/feature_sets.csv`, `data/feature_table.csv.gz` |
| Baseline | `tables/baseline.csv` |
| Write-up | `report.md` |

```
01_data_cleaning.ipynb          cleaning and the checks we ran
02_exploration_features.ipynb   plots, correlations, new features
report.md                       write-up for this check-in
src/                            the code the notebooks call
data/ tables/ figures/          outputs (rebuilt when you run the notebooks)
```

## How to run

1. Download `database.sqlite` from https://www.kaggle.com/datasets/hugomathien/soccer and put it in
   `data/` (or set the environment variable `SOCCER_DB` to where it is).
2. Install the packages (Python 3.11 or newer): `pip install -r requirements.txt`
3. Run the notebooks in order, either in Jupyter or with:

```sh
jupyter nbconvert --to notebook --execute --inplace 01_data_cleaning.ipynb
jupyter nbconvert --to notebook --execute --inplace 02_exploration_features.ipynb
```

This takes about 3 minutes.

## Main points so far

* 25,502 player-years (8,903 players) from 11 leagues, 2009–2015. Between 11% and 26% break out each year.
* Age and current rating matter most: 48% of players aged 20 or under break out, but only 13% at 25–29.
* EA's potential gap is mostly age, so we will also test models without it.
* Ranking by EA's potential gap finds 36–44% breakouts in the top 10%, against 11–20% for a random
  list. This is the baseline to beat.
