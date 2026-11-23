"""Step 2 - features: engineered features, the four feature sets and the preprocessor.

Every database read for cohort year y uses only records dated before its 1 August cut-off.
The one exception is the undated Player table (height, weight).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.common import (FEATURE_TABLE, ID_COLUMNS, LEAKY_COLUMNS, PLAYER, SEED, SKILLS,
                        connect, load_cleaned, load_ratings, season_starts, spike_mask)

# ----------------------------------------------------------------------------- feature groups
RATING_CURRENT = ["overall_rating", *SKILLS]            # latest rating before 1 August
POTENTIAL = ["potential", "potential_gap", "potential_change_prev_year"]
RATING_HISTORY = ["rating_change_prev_year",            # now minus latest rating 1-2 years ago
                  "has_prev_rating",                    # 1 if that earlier rating exists
                  "rating_change_in_window"]            # now minus earliest rating in last year
PLAYER_ATTRS = ["age_years", "height_cm", "weight_kg", "left_footed",
                "attacking_work_rate_ord", "defensive_work_rate_ord"]   # low/medium/high = 0/1/2
MATCH_HISTORY = ["starts", "start_share", "club_ppg", "rating_vs_teammates",
                 "prev_starts", "starts_change", "has_prev_season",     # previous season's starts
                 "team_mean_rating", "team_rating_rank_pct",            # squad context
                 "club_lineup_coverage"]                                # line-up data quality
CATEGORICAL = ["league", "position"]
# Engineered but kept out of every set: they mostly encode EA's update calendar (D02-8).
EXCLUDED_ENGINEERED = ["n_rating_updates_1y", "rating_age_days"]
NEVER_FEATURES = ["horizon_days", "position_missing"]

_NO_POT = [*RATING_CURRENT, *RATING_HISTORY, *PLAYER_ATTRS, "league"]
FEATURE_SETS = {
    "ea_gap_only": ["potential_gap"],
    "rating_no_potential": _NO_POT,
    "rating_only": [*_NO_POT, *POTENTIAL],
    "combined": [*_NO_POT, *POTENTIAL, *MATCH_HISTORY, "position"],
}
_FORBIDDEN = set(LEAKY_COLUMNS + ID_COLUMNS + NEVER_FEATURES)
assert not any(set(cols) & (_FORBIDDEN | set(EXCLUDED_ENGINEERED)) for cols in FEATURE_SETS.values())
ENGINEERED = ["rating_change_prev_year", "has_prev_rating", "rating_change_in_window",
              "rating_age_days", "n_rating_updates_1y", "potential_change_prev_year",
              "height_cm", "weight_kg", "left_footed", "attacking_work_rate_ord",
              "defensive_work_rate_ord", "prev_starts", "starts_change", "has_prev_season",
              "team_mean_rating", "team_rating_rank_pct"]


def work_rate(values: pd.Series) -> pd.Series:
    """'low'/'medium'/'high' -> 0/1/2; stray codes (None, 'norm', '_0', ...) -> NaN."""
    return values.astype("string").str.strip().str.lower().map(
        {"low": 0.0, "medium": 1.0, "high": 2.0}).astype(float)


def year_features(ratings, db, year, people) -> pd.DataFrame:
    """Pre-cut-off quantities for every player rated in the year before 1 Aug `year`."""
    cutoff = pd.Timestamp(year, 8, 1)
    one, two = cutoff - pd.DateOffset(years=1), cutoff - pd.DateOffset(years=2)
    before = ratings[ratings.date < cutoff]
    before = before[~spike_mask(before)]          # spikes judged on pre-cut-off records only
    win = before[before.date >= one]
    latest = lambda w: w.drop_duplicates(PLAYER, keep="last").set_index(PLAYER)
    cur, prev = latest(win), latest(before[before.date < one][lambda d: d.date >= two])
    f = pd.DataFrame({
        "prev_overall": prev.overall_rating, "prev_potential": prev.potential,
        "first_overall": win.drop_duplicates(PLAYER).set_index(PLAYER).overall_rating,
        "rating_age_days": (cutoff - cur.date).dt.days,
        "n_rating_updates_1y": win.groupby(PLAYER).size(),
        "left_footed": (cur.preferred_foot.astype("string").str.strip().str.lower() == "left")
        .astype(float).where(cur.preferred_foot.notna()),
        "attacking_work_rate_ord": work_rate(cur.attacking_work_rate),
        "defensive_work_rate_ord": work_rate(cur.defensive_work_rate),
    }).reindex(cur.index)
    # Squad context among the season's starters with a current rating (no future information).
    squad = season_starts(db, f"{year-1}/{year}", f"{year}-08-01").join(cur.overall_rating, how="inner")
    g = squad.groupby("team").overall_rating
    f["team_mean_rating"] = g.transform("mean")
    f["team_rating_rank_pct"] = g.rank(pct=True, method="average")
    prev_season = season_starts(db, f"{year-2}/{year-1}", f"{year-1}-08-01")
    has_prev = not prev_season.empty                  # 2007/08 is not in the database
    f["prev_starts"] = prev_season.starts.reindex(f.index).fillna(0).astype(float) if has_prev else np.nan
    f["has_prev_season"] = int(has_prev)
    f["height_cm"] = people.height.reindex(f.index).round(1)
    f["weight_kg"] = (people.weight.reindex(f.index) * 0.45359237).round(1)
    return f.assign(year=year).rename_axis(PLAYER).reset_index()


def build_feature_table(cleaned: pd.DataFrame | None = None) -> pd.DataFrame:
    """Cleaned table plus engineered features: same rows, same order."""
    cleaned = load_cleaned() if cleaned is None else cleaned
    with connect() as db:
        ratings = load_ratings(db, ["potential", "preferred_foot", "attacking_work_rate",
                                    "defensive_work_rate"])
        people = pd.read_sql("SELECT player_api_id, height, weight FROM Player", db).set_index(PLAYER)
        eng = pd.concat([year_features(ratings, db, int(y), people)
                         for y in sorted(cleaned.year.unique())], ignore_index=True)
    out = cleaned.drop(columns=[c for c in ENGINEERED if c in cleaned]).merge(
        eng, on=["year", PLAYER], how="left", validate="one_to_one")
    out["rating_change_prev_year"] = out.overall_rating - out.pop("prev_overall")
    out["has_prev_rating"] = out.rating_change_prev_year.notna().astype(int)
    out["potential_change_prev_year"] = out.potential - out.pop("prev_potential")
    out["rating_change_in_window"] = out.overall_rating - out.pop("first_overall")
    out["starts_change"] = out.starts - out.prev_starts
    assert (out[PLAYER].values == cleaned[PLAYER].values).all()
    return out


def load_feature_table(include_incomplete: bool = False, save: bool = False) -> pd.DataFrame:
    """Build the feature table (~5 s); `save` also writes a copy for the check-in folder."""
    t = build_feature_table(load_cleaned(include_incomplete=True))
    if save:
        t.to_csv(FEATURE_TABLE, index=False, float_format="%.7g")
    return t if include_incomplete else t[t.split != "incomplete"].reset_index(drop=True)


def make_preprocessor(columns, scale: bool = True, pca_skills=None) -> ColumnTransformer:
    """Unfitted: median imputation (+ scaling for linear models), one-hot for categoricals
    ('Unknown' for missing, unseen levels ignored), optional PCA on standardised skills.
    It is fitted only inside a model pipeline, on training rows."""
    columns = list(columns)
    if set(columns) & _FORBIDDEN:
        raise ValueError(f"Not allowed as features: {sorted(set(columns) & _FORBIDDEN)}")
    cats = [c for c in columns if c in CATEGORICAL]
    skills = [c for c in columns if c in SKILLS] if pca_skills else []
    nums = [c for c in columns if c not in cats and c not in skills]
    impute = lambda: SimpleImputer(strategy="median", keep_empty_features=True)
    parts = [("num", Pipeline([("impute", impute())] + ([("scale", StandardScaler())] if scale else [])), nums)]
    if skills:
        parts.append(("skills_pca", Pipeline([("impute", impute()), ("scale", StandardScaler()),
                                              ("pca", PCA(n_components=pca_skills, random_state=SEED))]), skills))
    if cats:
        parts.append(("cat", Pipeline([("impute", SimpleImputer(strategy="constant", fill_value="Unknown")),
                                       ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), cats))
    return ColumnTransformer([p for p in parts if p[2]], remainder="drop", sparse_threshold=0.0,
                             verbose_feature_names_out=False)
