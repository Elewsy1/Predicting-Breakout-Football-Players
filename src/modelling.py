"""Step 3 - modelling: rolling-year model selection and the final 2014 scores.

Run: python -m src.modelling   (~3 min; writes B_checkin_1201/tables and predictions)
The 2014 test year is only scored here; its metrics are computed in src.evaluation.
"""
from __future__ import annotations

import itertools
import json
import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.common import (B_FIGURES, B_TABLES, COHORTS, COLORS, DEV_YEARS, PLAYER, PREDICTIONS, SEED,
                        SKILLS, TARGET, TEST_YEAR, VALIDATION_YEARS, budget_metrics, cohort,
                        final_split, save)
from src.features import FEATURE_SETS, load_feature_table, make_preprocessor

SETS = list(FEATURE_SETS)
TRAIN_SCOPES = {"all": ["all"], "young": ["all", "young"]}   # training scopes compared per cohort
PRED_COLUMNS = ["cohort", "train_scope", "model", "feature_set", "year", PLAYER, "score", TARGET]
KEYS = ["cohort", "train_scope", "model", "feature_set", "setting", "window"]

# Logistic regression: interpretable reference; C from strong to weak L2 shrinkage (collinear
# skills), optional PCA on the 33 skills. Random forest: 300 trees; leaf size and features per
# split regularise. Gradient boosting: slow vs fast learning, shallow vs deeper trees.
# Class weighting is tried everywhere because only 11-26% of players break out.
GRIDS = {
    "logreg": {"C": [0.01, 0.1, 1.0], "class_weight": [None, "balanced"],
               "pca_skills": [None, 3, 5, 10]},
    "random_forest": {"min_samples_leaf": [5, 25], "max_features": ["sqrt", 0.33],
                      "class_weight": [None, "balanced_subsample"]},
    "hist_gb": {"learning_rate": [0.01, 0.03, 0.1], "max_leaf_nodes": [3, 7, 31],
                "class_weight": [None, "balanced"]},
}
FIXED = {
    "logreg": {"max_iter": 5000},
    "random_forest": {"n_estimators": 300, "n_jobs": 1, "random_state": SEED},
    "hist_gb": {"max_iter": 300, "min_samples_leaf": 40, "l2_regularization": 1.0,
                "early_stopping": False, "random_state": SEED},
}
ESTIMATORS = {"logreg": LogisticRegression, "random_forest": RandomForestClassifier,
              "hist_gb": HistGradientBoostingClassifier}


def build_pipeline(model: str, params: dict | None, feature_set: str) -> Pipeline:
    """Unfitted preprocessor + estimator. Only logistic regression is scaled; `pca_skills` in
    `params` goes to the preprocessor."""
    params = dict(params or {})
    pca = params.pop("pca_skills", None)
    prep = make_preprocessor(FEATURE_SETS[feature_set], scale=model == "logreg",
                             **({"pca_skills": pca} if pca is not None else {}))
    return Pipeline([("prep", prep), ("model", ESTIMATORS[model](**FIXED[model], **params))])


def grid(model: str, feature_set: str) -> list[dict]:
    g = dict(GRIDS[model])
    if "pca_skills" in g and not set(FEATURE_SETS[feature_set]) & set(SKILLS):
        g["pca_skills"] = [None]
    return [dict(zip(g, v)) for v in itertools.product(*g.values())]


def fit_score(model, params, feature_set, train, rows) -> np.ndarray:
    from threadpoolctl import threadpool_limits
    cols = FEATURE_SETS[feature_set]
    with threadpool_limits(1), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pipe = build_pipeline(model, params, feature_set).fit(train[cols], train[TARGET])
        return pipe.predict_proba(rows[cols])[:, 1]


def scored_rows(df, scope, year):
    """An all-player model scores everyone; a young-trained model only young players."""
    return cohort(df[df.year == year], scope)


def train_rows(df, scope, year, window=None):
    t = cohort(df, scope)
    return t[(t.year < year) & (t.year >= (year - window if window else 0))]


def run_validation(df, candidates, scopes=("all", "young"), window=None, years=VALIDATION_YEARS):
    """Fit every (model, feature set, setting) in every rolling fold; preprocessing is inside the
    pipeline, so it is fitted on that fold's training rows only. Returns (scores, predictions)."""
    jobs = [(sc, y, m, fs, p) for sc in scopes for y in years for (m, fs, p) in candidates]
    preds = Parallel(n_jobs=-1, max_nbytes=None)(
        delayed(fit_score)(m, p, fs, train_rows(df, sc, y, window), scored_rows(df, sc, y))
        for sc, y, m, fs, p in jobs)
    scores, frames = [], []
    for (sc, y, m, fs, p), s in zip(jobs, preds):
        rows = scored_rows(df, sc, y)
        frame = rows[["year", PLAYER, TARGET, "age"]].assign(score=s)
        for coh in [c for c in COHORTS if sc in TRAIN_SCOPES[c]]:
            sub = cohort(frame, coh)
            tag = dict(cohort=coh, train_scope=sc, model=m, feature_set=fs,
                       setting=json.dumps(p, sort_keys=True, default=str))
            scores.append({**tag, "window": window or "expanding", "year": y,
                           "train_rows": len(train_rows(df, sc, y, window)),
                           **budget_metrics(sub[TARGET], sub.score, sub[PLAYER])})
            frames.append(sub.drop(columns="age").assign(**tag))
    return pd.DataFrame(scores), pd.concat(frames, ignore_index=True)


def baselines(df, years):
    """EA potential-gap ranking and the analytic random expectation per cohort-year. EA rows
    are repeated under each training scope so every slice of the predictions has its comparator."""
    rows, preds = [], []
    for coh in COHORTS:
        for y in years:
            sub = cohort(df[df.year == y], coh)
            m = budget_metrics(sub[TARGET], sub.potential_gap, sub[PLAYER])
            base = dict(cohort=coh, train_scope="none", setting="{}", window="none", year=y, train_rows=0)
            rows.append({**base, "model": "ea_gap", "feature_set": "ea_gap_only", **m})
            rows.append({**base, "model": "random", "feature_set": "none", **m,
                         "hits": m["watchlist"] * m["prevalence"], "precision_at_10pct": m["prevalence"],
                         "recall_at_10pct": m["watchlist"] / m["players"],
                         "average_precision": m["prevalence"]})
            for sc in TRAIN_SCOPES[coh]:
                preds.append(sub[["year", PLAYER, TARGET]].assign(
                    cohort=coh, train_scope=sc, model="ea_gap", feature_set="ea_gap_only",
                    score=sub.potential_gap.astype(float)))
    return pd.DataFrame(rows), pd.concat(preds, ignore_index=True)


def summarise(scores):
    """Mean, SD and minimum Precision@10% across the validation years, per candidate."""
    g = scores.groupby(KEYS, dropna=False)
    out = g.agg(mean_p10=("precision_at_10pct", "mean"), sd_p10=("precision_at_10pct", "std"),
                min_p10=("precision_at_10pct", "min"), mean_ap=("average_precision", "mean"),
                mean_recall=("recall_at_10pct", "mean"), folds=("year", "nunique")).reset_index()
    wide = scores.pivot_table(index=KEYS, columns="year", values="precision_at_10pct").add_prefix("p10_")
    wide.columns = [str(c) for c in wide.columns]
    return out.merge(wide.reset_index(), on=KEYS)


def rank(summary):
    """Selection rule: highest mean Precision@10%, ties broken by mean average precision."""
    return summary.sort_values(["mean_p10", "mean_ap"], ascending=False, kind="mergesort")


def select_best(summary, by=("cohort", "train_scope", "model", "feature_set")):
    ranked = rank(summary[~summary.model.isin(["ea_gap", "random"])])
    return ranked.groupby(list(by), sort=False).head(1).sort_values(list(by)).reset_index(drop=True)


def final_predictions(df, best):
    """Refit each best setting on 2009-2013 and score 2014 (no test metrics here)."""
    dev, test = final_split(df)
    fits = list(best[["train_scope", "model", "feature_set", "setting"]].drop_duplicates()
                .itertuples(index=False, name=None))
    scores = Parallel(n_jobs=-1, max_nbytes=None)(
        delayed(fit_score)(m, json.loads(st), fs, cohort(dev, sc), cohort(test, sc))
        for sc, m, fs, st in fits)
    by_fit = dict(zip(fits, scores))
    frames = []
    for r in best.itertuples(index=False):
        rows = cohort(test, r.train_scope)[["year", PLAYER, TARGET, "age"]]
        rows = rows.assign(score=by_fit[(r.train_scope, r.model, r.feature_set, r.setting)])
        frames.append(cohort(rows, r.cohort).assign(cohort=r.cohort, train_scope=r.train_scope,
                                                    model=r.model, feature_set=r.feature_set))
    return pd.concat(frames + [baselines(df, [TEST_YEAR])[1]], ignore_index=True)[PRED_COLUMNS]


def window_check(df, scores, best, window=2):
    """Does training only on the last 2 years beat the expanding window? Paired over the same
    candidate and fold (2011 already trains on 2 years, so only 2012-2013 can differ)."""
    parts = []
    for sc, g in best.groupby("train_scope"):
        cands = [(r.model, r.feature_set, json.loads(r.setting)) for r in g.drop_duplicates(
            ["model", "feature_set", "setting"]).itertuples()]
        parts.append(run_validation(df, cands, scopes=(sc,), window=window, years=[2012, 2013])[0])
    k = ["cohort", "train_scope", "model", "feature_set", "setting", "year"]
    old = scores[(scores.window == "expanding") & scores.year.isin([2012, 2013])]
    d = old[k + ["precision_at_10pct"]].merge(pd.concat(parts)[k + ["precision_at_10pct"]], on=k)
    diff = d.iloc[:, -1] - d.iloc[:, -2]
    mean, share = float(diff.mean()), float((diff > 0).mean())
    return {"pairs": len(d), "mean_diff_p10": round(mean, 4), "share_window_better": round(share, 3),
            "rule": "adopt only if mean gain > 1 pp and > 75% of (candidate, fold) pairs improve",
            "verdict": "adopt 2-year window" if mean > 0.01 and share > 0.75
            else "keep expanding window (2009-2013 refit)"}


def describe(row) -> dict:
    return {"train_scope": row.train_scope, "model": row.model, "feature_set": row.feature_set,
            "params": json.loads(row.setting), "fixed_params": FIXED[row.model],
            "features": FEATURE_SETS[row.feature_set],
            "validation": {"mean_precision_at_10pct": round(row.mean_p10, 4),
                           "precision_at_10pct_by_year": {str(y): round(row[f"p10_{y}"], 4)
                                                          for y in VALIDATION_YEARS},
                           "mean_average_precision": round(row.mean_ap, 4)}}


def main():
    df = load_feature_table()
    df = df[df.year.isin(DEV_YEARS + [TEST_YEAR])].reset_index(drop=True)
    cands = [(m, fs, p) for fs in SETS for m in GRIDS for p in grid(m, fs)]
    print(f"validating {len(cands)} candidates x 3 folds x 2 training scopes")
    scores, preds = run_validation(df, cands)
    base_scores, base_preds = baselines(df, VALIDATION_YEARS)
    scores = pd.concat([scores, base_scores], ignore_index=True)
    summary = summarise(scores)
    best = select_best(summary)
    for obj, name in [(scores, "validation_scores"), (summary, "validation_summary"),
                      (best, "best_settings")]:
        save(obj, B_TABLES, name, fmt=None)
    wc = window_check(df, scores, best)
    print("window check:", wc["verdict"])
    PREDICTIONS.mkdir(parents=True, exist_ok=True)
    keep = preds.merge(best[["cohort", "train_scope", "model", "feature_set", "setting"]])
    pd.concat([keep[PRED_COLUMNS], base_preds[PRED_COLUMNS]]).sort_values(PRED_COLUMNS[:6]).to_csv(
        PREDICTIONS / "validation_predictions.csv.gz", index=False)
    final_predictions(df, best).sort_values(PRED_COLUMNS[:6]).to_csv(
        PREDICTIONS / "test_predictions.csv.gz", index=False)
    selected = {"selection_rule": "per cohort: highest mean Precision@10% over 2011-2013 (rolling, "
                "expanding window), tie-break mean average precision; refit on 2009-2013",
                "window_check": wc, "cohorts": {}}
    for coh in COHORTS:
        s = summary[summary.cohort == coh].set_index("model")
        selected["cohorts"][coh] = {
            "primary": describe(rank(best[best.cohort == coh]).iloc[0]),
            "baselines": {m: round(s.loc[m, "mean_p10"], 4) for m in ["ea_gap", "random"]}}
    (B_TABLES / "selected_models.json").write_text(json.dumps(selected, indent=2))
    return selected


# ----------------------------------------------------------------------------- notebook helpers
LABELS = {"ea_gap_only": "EA gap only", "rating_no_potential": "Rating, no potential",
          "rating_only": "Rating incl. potential", "combined": "Rating + match history",
          "ea_gap": "EA potential gap", "random": "Random"}
SET_COLORS = {"rating_no_potential": "#1baf7a", "rating_only": "#2a78d6", "combined": "#eb6834"}


def load_outputs():
    return [pd.read_csv(B_TABLES / f"{n}.csv") for n in ["validation_scores", "validation_summary",
                                                          "best_settings"]]


def p10(row) -> np.ndarray:
    return np.array([row[f"p10_{y}"] for y in VALIDATION_YEARS], dtype=float)


def comparison_table(best, summary, coh):
    """Best setting per (scope, family, feature set) next to the baselines, in %."""
    t = pd.concat([best[best.cohort == coh],
                   summary[(summary.cohort == coh) & summary.model.isin(["ea_gap", "random"])]])
    cols = [f"p10_{y}" for y in VALIDATION_YEARS] + ["mean_p10", "sd_p10", "mean_recall", "mean_ap"]
    t[cols] = (100 * t[cols]).round(1)
    return t[["train_scope", "model", "feature_set", *cols, "setting"]].sort_values(
        "mean_p10", ascending=False).reset_index(drop=True)


def plot_validation(best, summary):
    """Precision@10% per validation year: best model per feature set vs EA gap and random."""
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    for ax, (coh, sc) in zip(axes, [("all", "all"), ("young", "all"), ("young", "young")]):
        b = best[(best.cohort == coh) & (best.train_scope == sc)]
        for fs in SETS[1:]:   # a model on the gap alone reproduces the EA line
            ax.plot(VALIDATION_YEARS, 100 * p10(rank(b[b.feature_set == fs]).iloc[0]), marker="o",
                    lw=2, color=SET_COLORS[fs], label=LABELS[fs])
        for m, style in [("ea_gap", dict(color=COLORS["ink"], lw=2.5, marker="s")),
                         ("random", dict(color=COLORS["muted"], ls="--", lw=1.5))]:
            r = summary[(summary.cohort == coh) & (summary.model == m)].iloc[0]
            ax.plot(VALIDATION_YEARS, 100 * p10(r), label=LABELS[m], **style)
        ax.set(title=f"{coh} players - trained on {sc}", xticks=VALIDATION_YEARS,
               xlabel="validation year", xlim=(2010.8, 2013.2))
    axes[0].set_ylabel("Precision@10% (%)")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=5)
    fig.suptitle("Rolling validation: share of the 10% watchlist that broke out", x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.07, 1, 0.95))
    return fig


def feature_set_contrasts(best, summary):
    """Within each model family, per-year P@10% gains (pp) from changing only the features."""
    pairs = [("rating_only", "rating_no_potential", "+ potential"),
             ("combined", "rating_only", "+ match history"),
             ("rating_no_potential", "ea_gap", "rating w/o potential vs EA gap"),
             ("combined", "ea_gap", "combined vs EA gap")]
    rows = []
    for (coh, sc, m), b in best.groupby(["cohort", "train_scope", "model"]):
        get = {fs: g.iloc[0] for fs, g in b.groupby("feature_set")}
        get["ea_gap"] = summary[(summary.cohort == coh) & (summary.model == "ea_gap")].iloc[0]
        for a, z, label in pairs:
            d = 100 * (p10(get[a]) - p10(get[z]))
            rows.append({"cohort": coh, "train_scope": sc, "model": m, "contrast": label,
                         **{str(y): round(v, 1) for y, v in zip(VALIDATION_YEARS, d)},
                         "mean_pp": round(d.mean(), 1), "years_better": int((d > 0).sum())})
    return pd.DataFrame(rows)


def scope_contrasts(best):
    """Young cohort: training on young players only minus training on all players (pp)."""
    rows = []
    for (m, fs), g in best[best.cohort == "young"].groupby(["model", "feature_set"]):
        d = 100 * (p10(g[g.train_scope == "young"].iloc[0]) - p10(g[g.train_scope == "all"].iloc[0]))
        rows.append({"model": m, "feature_set": fs, "mean_pp": round(d.mean(), 1),
                     "years_better": int((d > 0).sum())})
    return pd.DataFrame(rows)


def grid_effects(summary):
    """How much tuning matters: spread of mean P@10% across settings, and the class-weight effect."""
    s = summary[~summary.model.isin(["ea_gap", "random"])].copy()
    spread = (100 * s.groupby(["cohort", "train_scope", "model", "feature_set"]).mean_p10
              .agg(["min", "median", "max"])).round(1).reset_index()
    s["weighted"] = s.setting.map(lambda x: json.loads(x).get("class_weight") is not None)
    cw = (100 * s.pivot_table(index=["cohort", "train_scope", "model"], columns="weighted",
                              values="mean_p10")).round(1).set_axis(["unweighted", "weighted"], axis=1)
    return spread, cw.reset_index()


if __name__ == "__main__":
    main()
