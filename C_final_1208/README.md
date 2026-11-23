# Predicting Breakout Football Players

Team: Shiyan Wang, Junius, Noah Feng, Xianghe Meng · IDSC-GH 5010 Introduction to Data Science, Fall 2026

Every 1 August we rank the football players who started a league match in the season that just
ended. A scout who can only follow 10% of them should find as many "breakouts" as possible: players
whose FIFA overall rating rises by at least the year's median change plus 3 points in the next year.
We compare our models with EA's own potential-minus-overall ranking and with a random list.

**Result:** on the 2014 test year our list of 362 players found 169 breakouts, against 134 for EA's
ranking and about 49 for a random list.

## Files

```
report.md                       the report (all six parts)
slides_outline.md               plan for the presentation slides
01_data_cleaning.ipynb          cleaning and the checks we ran
02_exploration_features.ipynb   plots, correlations, new features
03_modelling.ipynb              choosing the model on 2011-2013
04_evaluation.ipynb             final test on 2014, explanations, robustness checks
src/                            the code the notebooks call
  common.py                     paths, splits and the Precision@10% metric
  cleaning.py                   step 1: cleaning
  features.py, eda.py           step 2: features and exploration
  modelling.py                  step 3: model selection
  evaluation.py                 step 4: final test and explanations
data/ tables/ figures/ predictions/   outputs (rebuilt when you run the notebooks)
```

## Getting the data

The data is the European Soccer Database by Hugo Mathien on Kaggle:
https://www.kaggle.com/datasets/hugomathien/soccer (one SQLite file, 313 MB, free account needed).
Download and unzip it, then put `database.sqlite` in the `data/` folder. If you keep it somewhere
else, set the environment variable `SOCCER_DB` to its path.

## Running the code

Python 3.11 or newer.

```sh
pip install -r requirements.txt
jupyter nbconvert --to notebook --execute --inplace 01_data_cleaning.ipynb
jupyter nbconvert --to notebook --execute --inplace 02_exploration_features.ipynb
jupyter nbconvert --to notebook --execute --inplace 03_modelling.ipynb
jupyter nbconvert --to notebook --execute --inplace 04_evaluation.ipynb
```

You can also open the notebooks in Jupyter and run them from top to bottom, in this order. Each one
reads the outputs of the one before. The whole run takes about 12 minutes on a laptop. All random
steps use a fixed seed, so the numbers come out the same every time.

## Sources

Mathien, H. (2016). *European Soccer Database*, version 10. Kaggle. Code uses pandas, NumPy, SciPy,
scikit-learn, matplotlib and seaborn.
