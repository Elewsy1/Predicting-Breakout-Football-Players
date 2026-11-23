# Predicting Breakout Football Players — Revised Project Proposal

**Team:** Shiyan Wang, Junius, Noah Feng, Xianghe Meng · IDSC-GH 5010, Fall 2026

**Question and action**
Can FIFA ratings and past match records predict which players will achieve an unusually large increase in their FIFA overall rating over the following year? On 1 August of each year we predict a "breakout": by 1 August next year, the player's FIFA overall rating rises at least 3 points more than the median change of all players that year. EA index is the rating of the players provided by FIFA. A scouting team that can follow only 10% of players would use our ranking to choose whom to watch. We run the study twice: on all players, and on young players (age ≤ 24), the group scouts care about most. 

**Dataset**
European Soccer Database (kaggle.com/datasets/hugomathien/soccer; SQLite, 313 MB, non-commercial use). It has 25,979 matches in 11 European leagues (2008/09–2015/16) with line-ups and pitch positions, 11,060 players, and 183,978 dated rating records (overall, potential, 33 skills). One instance is one player in one year who started at least one league match in the season just finished. **Confirmed samples:** all ages, 25,503 player-years (8,903 players; 3,318–3,871 per year; breakout rate 11–26% by year); age ≤ 24, 7,551 player-years (3,972 players).
**Rating coverage:** Ratings are recorded on irregular dates, and not every player has a valid rating each year. For each 1 August prediction date, we identify players who started at least one match in the preceding season and require a non-missing overall rating in both the preceding and following 12-month windows, using the latest available rating within each window. Across 2009–2015, 29,149 candidate player-year observations yield 25,503 usable observations involving 8,903 unique players. Among candidates with a valid rating in the preceding 12 months, 11.48% lack a rating in the following 12 months and are excluded. This exclusion may introduce selection bias and limits the generalizability of our findings.

**Approach and evaluation**
*Cleaning:* Remove 836 empty rating rows. Check IDs, dates, types, and outliers; use only the latest rating before each cut-off.
*Exploration and features:* Plot breakout rates by age, league, position, and rating, and check correlations among skills. From line-ups we build playing-time and team features (share of matches started, pitch position, team points per game, rating compared with teammates). Because the 33 skills are highly correlated, we will also try a principal component analysis (PCA) to summarise them into a few components. Linear models get median imputation, scaling, and one-hot league codes, all fitted on training rows only.

*Methods:* Compare logistic regression, random forest, and gradient boosting with random selection and EA's potential-minus-current-rating ranking. Compare rating-only and combined feature sets to assess whether match history adds predictive value. Logistic regression provides an interpretable reference and tree ensembles capture nonlinear relationships. Choose model settings through validation. Fit imputation, encoding, and any scaling on each training fold only.

*Validation:* Use rolling-year validation for 2011–2013, training on earlier years. Refit the selected pipeline on 2009–2013 and reserve 2014–2015 for final testing, retaining only years with complete follow-up windows. Confirm coverage through 1 August 2016 before including the 2015 cohort. Players may recur across years, reflecting prediction of future outcomes for historically observed players.

*Evaluation:* Primary metric: Precision@10%, the proportion of selected players who break out each test year. Also report breakout counts, Recall@10%, and PR-AUC, comparing the same scouting budget across strategies. Estimate uncertainty by bootstrapping players to account for repeated observations. Explain predictions using permutation importance and logistic-regression coefficients; interpret associations without causal claims.

**Feasibility and Plan**
The four roles will cover data cleaning, exploration and features, modelling, and evaluation and explanations. Everyone will help write and present. We aim to finish cleaning and a baseline by 24 November and model comparisons by 1 December. We will keep event-text features outside the main scope. Main risks are missing ratings and FIFA ratings being a subjective measure of improvement. We would like feedback on the breakout definition and whether the scouting action is useful.

## Reproduce the first check-in

Download and unzip the [European Soccer Database by Hugo Mathien](https://www.kaggle.com/datasets/hugomathien/soccer). From this repository, run the commands below with Python 3.11 or newer. Replace the example database path with the location of the extracted file.

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python process_data.py /path/to/database.sqlite
```

Open [the first check-in](CHECKIN.md) for the results and a walkthrough. The script reads the source database and writes the cleaned table, coverage audit, figures, correlations and baseline scores to `checkin/`. Pass `--out DIRECTORY` to choose another destination.
