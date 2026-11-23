# Predicting Breakout Football Players — Revised Project Proposal

**Team:** Shiyan Wang, Junius, Noah Feng, Xianghe Meng · IDSC-GH 5010, Fall 2026

**Question and action**
Can FIFA ratings and past match records predict which players will achieve an unusually large increase in their FIFA overall rating over the following year? On 1 August of each year we predict a "breakout": by 1 August next year, the player's FIFA overall rating rises at least 3 points more than the median change of all players that year. EA index is the rating of the players provided by FIFA. A scouting team that can follow only 10% of players would use our ranking to choose whom to watch. We run the study twice: on all players, and on young players (aged 24 or younger in completed years on 1 August), the group scouts care about most. 

**Dataset**
European Soccer Database (kaggle.com/datasets/hugomathien/soccer; SQLite, 313 MB; licence: Open Database for the database, contents © original authors, author asks for non-commercial use). It has 25,979 matches in 11 European leagues (2008/09–2015/16) with line-ups and pitch positions, 11,060 players, and 183,978 dated rating records (overall, potential, 33 skills). One instance is one player in one year who started at least one league match in the season just finished. **Confirmed samples:** all ages, 25,502 player-years (8,903 players; 3,317–3,871 per year; breakout rate 11–26% by year); age ≤ 24 (completed years), 9,687 player-years (4,724 players).
**Rating coverage:** Ratings are recorded on irregular dates, and not every player has a valid rating each year. For each 1 August prediction date, we identify players who started at least one match in the preceding season and require a non-missing overall rating in both the preceding and following 12-month windows, using the latest available rating within each window. Across 2009–2015, 29,149 candidate player-year observations yield 25,503 usable observations involving 8,903 unique players. Among candidates with a valid rating in the preceding 12 months, 11.48% lack a rating in the following 12 months and are excluded. This exclusion may introduce selection bias and limits the generalizability of our findings.

**Approach and evaluation**
*Cleaning:* Remove 836 empty rating rows and 21 isolated rating spikes (entry errors). Check IDs, dates, types, and outliers; use only the latest rating before each cut-off.
*Exploration and features:* Plot breakout rates by age, league, position, and rating, and check correlations among skills. From line-ups we build playing-time and team features (share of matches started, pitch position, team points per game, rating compared with teammates). Because the 33 skills are highly correlated, we will also try a principal component analysis (PCA) to summarise them into a few components. Linear models get median imputation, scaling, and one-hot league codes, all fitted on training rows only.

*Methods:* Compare logistic regression, random forest, and gradient boosting with random selection and EA's potential-minus-current-rating ranking. Compare rating-only and combined feature sets to assess whether match history adds predictive value. Logistic regression provides an interpretable reference and tree ensembles capture nonlinear relationships. Choose model settings through validation. Fit imputation, encoding, and any scaling on each training fold only.

*Validation:* Use rolling-year validation for 2011–2013, training on earlier years. Refit the selected pipeline on 2009–2013 and reserve 2014 for final testing. The 2015 cohort is dropped because ratings end on 7 July 2016, so its follow-up window is incomplete. Players may recur across years, reflecting prediction of future outcomes for historically observed players.

*Evaluation:* Primary metric: Precision@10%, the proportion of selected players who break out each test year. Also report breakout counts, Recall@10%, and PR-AUC, comparing the same scouting budget across strategies. Estimate uncertainty by bootstrapping players to account for repeated observations. Explain predictions using permutation importance and logistic-regression coefficients; interpret associations without causal claims.

**Feasibility and Plan**
The four roles will cover data cleaning, exploration and features, modelling, and evaluation and explanations. Everyone will help write and present. We aim to finish cleaning and a baseline by 24 November and model comparisons by 1 December. We will keep event-text features outside the main scope. Main risks are missing ratings and FIFA ratings being a subjective measure of improvement. We would like feedback on the breakout definition and whether the scouting action is useful.

## Data

Mathien, H. (2016). *European Soccer Database*, Kaggle, version 10 (2016-10-23).
https://www.kaggle.com/datasets/hugomathien/soccer. Original sources: football-data.mx-api.enetscores.com
(matches, line-ups), sofifa.com (EA Sports FIFA ratings; FIFA data are EA Sports' property).

Download and unzip it, then either put `database.sqlite` at `_local/database.sqlite` (ignored by git) or set
the `SOCCER_DB` environment variable to its path. Never commit the database (313 MB).

## Reproduce the analysis

Python 3.11 or newer. Create an environment and install the requirements:

```sh
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
```

Then run the four steps in order from the repository root (each overwrites its own outputs):

```sh
python -m src.cleaning                                                                         # 1. cleaning (~10 s)
jupyter nbconvert --to notebook --execute --inplace A_checkin_1124/02_exploration_features.ipynb   # 2. exploration + features (~2 min)
jupyter nbconvert --to notebook --execute --inplace B_checkin_1201/03_modelling.ipynb              # 3. model selection (~2 min)
jupyter nbconvert --to notebook --execute --inplace B_checkin_1201/04_evaluation_explanations.ipynb # 4. evaluation (~5 min)
```

Steps 3 and 4 can also run as scripts: `python -m src.modelling`, `python -m src.evaluation`.

## Repository layout

Deliverables are grouped by course milestone. Each check-in adds a folder: check-in 1 = A,
check-in 2 = A + B, final submission = A + B + C.

| Path | Contents |
|---|---|
| [`A_checkin_1124/`](A_checkin_1124/README.md) | **24 Nov:** cleaned data and audits (`data/`), exploration notebook, plots, correlations, initial features, baseline, report sections 1–3 |
| [`B_checkin_1201/`](B_checkin_1201/README.md) | **1 Dec:** model selection and evaluation notebooks, predictions, result tables and figures, draft report, slide outline |
| [`C_final_1208/`](C_final_1208/README.md) | **8/10 Dec:** final report, slides and reflections (checklist and templates) |
| `src/` | All code: `common.py` (paths, splits, metrics), `cleaning.py`, `features.py`, `eda.py`, `modelling.py`, `evaluation.py` |
| `docs/` | Work plan and the decision logs that still need team confirmation |
| `_local/` | Your own copy of the database and course files (ignored by git) |
