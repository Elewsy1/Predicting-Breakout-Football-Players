"""Helpers for the exploration notebook (02_exploration_features.ipynb).
Only development years 2009-2013 are explored; 2014 labels are never looked at."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.common import COLORS, PLAYER, SEED, SKILLS, TARGET, budget_metrics, cohort, rolling_folds
from src.features import make_preprocessor

AGE_BANDS = ([0, 20, 24, 29, 34, 100], ["≤20", "21–24", "25–29", "30–34", "35+"])
RATING_BANDS = ([0, 59, 69, 79, 100], ["≤59", "60–69", "70–79", "80+"])


def add_bands(df):
    return df.assign(age_band=pd.cut(df.age, AGE_BANDS[0], labels=AGE_BANDS[1]),
                     rating_band=pd.cut(df.overall_rating, RATING_BANDS[0], labels=RATING_BANDS[1]))


def rate_table(df, by):
    """Rows, breakouts and breakout rate per group (missing groups shown as 'Unknown')."""
    key = df[by] if df[by].dtype == "category" else df[by].astype("string").fillna("Unknown")
    return df.groupby(key, observed=True)[TARGET].agg(rows="size", breakouts="sum", rate="mean").reset_index()


def univariate_scores(df, features):
    """Non-missing share, Pearson r, Spearman rho and ranking AUC of each feature vs breakout."""
    rows, y = [], df[TARGET]
    for c in features:
        x = pd.to_numeric(df[c], errors="coerce")
        ok = x.notna()
        auc = roc_auc_score(y[ok], x[ok]) if ok.any() and y[ok].nunique() == 2 else np.nan
        rows.append({"feature": c, "nonmissing_share": ok.mean(), "pearson_r": x.corr(y),
                     "spearman_rho": x.corr(y, method="spearman"), "auc": auc,
                     "auc_oriented": max(auc, 1 - auc)})
    return pd.DataFrame(rows)


def potential_gap_vs_age(df):
    """How much of EA's potential gap is age: R² on single-year age means, and the ranking power
    of the raw gap, of age alone and of the age-adjusted gap."""
    resid = df.potential_gap - df.groupby(df.age.clip(upper=36)).potential_gap.transform("mean")
    y = df[TARGET]
    return {"rows": len(df), "r2_gap_on_age": 1 - (resid ** 2).sum() / ((df.potential_gap - df.potential_gap.mean()) ** 2).sum(),
            "corr_gap_age": df.potential_gap.corr(df.age_years), "auc_gap": roc_auc_score(y, df.potential_gap),
            "auc_minus_age": roc_auc_score(y, -df.age_years), "auc_age_adjusted_gap": roc_auc_score(y, resid)}


def skill_pca(df):
    """Explained variance ratios and loadings of PCA on median-imputed, standardised skills."""
    pipe = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler()),
                     ("pca", PCA(random_state=SEED))]).fit(df[SKILLS])
    pca = pipe[-1]
    return pca.explained_variance_ratio_, pd.DataFrame(
        pca.components_.T, index=SKILLS, columns=[f"PC{i + 1}" for i in range(pca.n_components_)])


def binned_rate(df, feature, bins=None, q=8):
    """Breakout rate by quantile bins (or given bins) of a numeric feature; missing = own bin."""
    x = pd.to_numeric(df[feature], errors="coerce")
    key = (pd.qcut(x, q, duplicates="drop") if bins is None else pd.cut(x, bins)).astype("string").fillna("missing")
    t = df.groupby(key, sort=False)[TARGET].agg(rows="size", rate="mean")
    t["x_mid"] = x.groupby(key, sort=False).median()
    t = t.reset_index().set_axis(["bin", "rows", "rate", "x_mid"], axis=1)
    return t.assign(feature=feature).sort_values("x_mid", na_position="last")


def screen_feature_groups(df, variants, cohort_name="all"):
    """Quick check, on the rolling folds, whether a feature group adds signal: two fixed,
    untuned models (scaled logistic regression, default gradient boosting). Not model selection."""
    rows = []
    for name, cols in variants.items():
        for model in ("logistic", "hgb"):
            for year, train, valid in rolling_folds(cohort(df, cohort_name)):
                est = LogisticRegression(C=1.0, max_iter=2000) if model == "logistic" else \
                    HistGradientBoostingClassifier(random_state=SEED)
                pipe = Pipeline([("pre", make_preprocessor(cols, scale=model == "logistic")), ("model", est)])
                s = pipe.fit(train[cols], train[TARGET]).predict_proba(valid[cols])[:, 1]
                m = budget_metrics(valid[TARGET], s, valid[PLAYER])
                rows.append({"cohort": cohort_name, "variant": name, "model": model, "year": year,
                             "n_features": len(cols), "precision_at_10pct": m["precision_at_10pct"],
                             "average_precision": m["average_precision"]})
    return pd.DataFrame(rows)


def plot_rate_bars(ax, tables, group_col, order, title, min_rows=30):
    """Grouped horizontal bars of breakout rate, one series per cohort; small groups hidden."""
    h = 0.8 / len(tables)
    for i, (name, t) in enumerate(tables.items()):
        t = t.set_index(group_col).reindex(order)
        rate = t.rate.where(t.rows >= min_rows)
        y = np.arange(len(order)) + (i - (len(tables) - 1) / 2) * h
        ax.barh(y, 100 * rate, height=h * 0.92, color=COLORS[name], label=name)
        for yy, r in zip(y, rate):
            if pd.notna(r):
                ax.text(100 * r + 0.6, yy, f"{100 * r:.0f}", va="center", fontsize=8, color=COLORS["ink2"])
    ax.set_yticks(np.arange(len(order)), order)
    ax.invert_yaxis()
    ax.set(xlabel="breakout rate (%)", title=title)
    ax.grid(axis="y", visible=False)
