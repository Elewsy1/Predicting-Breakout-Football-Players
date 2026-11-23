"""Shared paths, constants, data access, metrics and plot style for every step."""
from __future__ import annotations

import os
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

ROOT = Path(__file__).resolve().parents[1]
# Deliverables are grouped by milestone: A = 24 Nov check-in, B = 1 Dec check-in, C = final.
A_DIR, B_DIR, C_DIR = ROOT / "A_checkin_1124", ROOT / "B_checkin_1201", ROOT / "C_final_1208"
DATA = A_DIR / "data"
CLEANED = DATA / "cleaned_player_years.csv.gz"
FEATURE_TABLE = DATA / "feature_table.csv.gz"
A_TABLES, A_FIGURES = A_DIR / "tables", A_DIR / "figures"
B_TABLES, B_FIGURES, PREDICTIONS = B_DIR / "tables", B_DIR / "figures", B_DIR / "predictions"

SEED = 42
PLAYER, TARGET = "player_api_id", "breakout"
COHORTS = ["all", "young"]
YOUTH_MAX_AGE = 24         # completed years on 1 August, inclusive
BUDGET = 0.10              # scouts watch floor(10%) of each year's cohort

# A cohort year y is predicted on 1 Aug y and its label is known on 1 Aug y+1,
# so validation year v may train on every year < v. 2015 (incomplete follow-up) is never used.
VALIDATION_YEARS = [2011, 2012, 2013]
DEV_YEARS = [2009, 2010, *VALIDATION_YEARS]
TEST_YEAR = 2014

# Outcome or outcome-timed columns (never features) and identifiers.
LEAKY_COLUMNS = ["outcome_date", "next_rating", "change", "annual_median_change", TARGET,
                 "horizon_days"]
ID_COLUMNS = [PLAYER, "player_name", "birthday", "team", "league_id", "year", "split",
              "rating_date"]

SKILLS = """crossing finishing heading_accuracy short_passing volleys dribbling curve
free_kick_accuracy long_passing ball_control acceleration sprint_speed agility reactions
balance shot_power jumping stamina strength long_shots aggression interceptions
positioning vision penalties marking standing_tackle sliding_tackle gk_diving
gk_handling gk_kicking gk_positioning gk_reflexes""".split()


# ----------------------------------------------------------------------------- data access
def database_path() -> Path:
    """$SOCCER_DB if set, else _local/database.sqlite."""
    return Path(os.environ.get("SOCCER_DB", ROOT / "_local" / "database.sqlite"))


def connect(path: Path | None = None) -> sqlite3.Connection:
    path = Path(path or database_path()).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Database not found at {path}; set $SOCCER_DB.")
    return sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)


def spike_mask(r: pd.DataFrame, points: int = 8, agree: int = 2) -> pd.Series:
    """Isolated rating spikes (e.g. 64, 51, 64): at least `points` from both neighbours while
    the neighbours agree within `agree`. Data-entry errors, not rating changes."""
    o = r.sort_values([PLAYER, "date"]).groupby(PLAYER).overall_rating
    before, after = o.shift(1), o.shift(-1)
    x = r.overall_rating
    return ((x - before).abs() >= points) & ((x - after).abs() >= points) & \
        ((before - after).abs() <= agree)


def load_ratings(db, columns=("overall_rating",)) -> pd.DataFrame:
    """Rating records with an overall rating, one per (player, date) (last by id), sorted by
    player and date. Spikes are NOT removed here (callers decide over which records)."""
    cols = ["id", PLAYER, "date", *dict.fromkeys(["overall_rating", *columns])]
    r = pd.read_sql(f"SELECT {', '.join(cols)} FROM Player_Attributes", db, parse_dates=["date"])
    r = r.dropna(subset=[PLAYER, "date", "overall_rating"]).sort_values(["date", "id"])
    r = r[~r.duplicated([PLAYER, "date"], keep="last")]
    return r.sort_values([PLAYER, "date"])


def season_starts(db, season: str, before: str | None = None) -> pd.DataFrame:
    """League starts per player in one season (all 22 line-up slots, once per match) and the
    club he started for most often (ties: lower team id). Index: player id."""
    slots = [f"{s}_player_{i}" for s in ("home", "away") for i in range(1, 12)]
    query = f"SELECT id, home_team_api_id, away_team_api_id, {', '.join(slots)} FROM Match " \
            "WHERE season=?" + (" AND date<?" if before else "")
    m = pd.read_sql(query, db, params=(season, before) if before else (season,))
    parts = [m.melt(id_vars=["id", f"{s}_team_api_id"], value_vars=[c for c in slots if s in c],
                    value_name=PLAYER).rename(columns={f"{s}_team_api_id": "team"})
             for s in ("home", "away")]
    s = pd.concat(parts).dropna(subset=[PLAYER]).drop_duplicates(["id", PLAYER])
    s[PLAYER] = s[PLAYER].astype(int)
    club = (s.groupby([PLAYER, "team"]).size().rename("n").reset_index()
            .sort_values([PLAYER, "n", "team"], ascending=[True, False, True])
            .drop_duplicates(PLAYER).set_index(PLAYER).team)
    return pd.concat([s.groupby(PLAYER).size().rename("starts"), club], axis=1)


def load_cleaned(include_incomplete: bool = False) -> pd.DataFrame:
    df = pd.read_csv(CLEANED, parse_dates=["birthday", "rating_date", "outcome_date"])
    return df if include_incomplete else df[df.split != "incomplete"].reset_index(drop=True)


# ----------------------------------------------------------------------------- splits
def cohort(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """'all', or 'young' = completed age on 1 August <= 24."""
    return df if name == "all" else df[df.age <= YOUTH_MAX_AGE]


def rolling_folds(df: pd.DataFrame):
    """(validation_year, rows of all earlier years, rows of that year)."""
    for year in VALIDATION_YEARS:
        yield year, df[df.year < year], df[df.year == year]


def final_split(df: pd.DataFrame):
    return df[df.year.isin(DEV_YEARS)], df[df.year == TEST_YEAR]


# ----------------------------------------------------------------------------- metrics
def watchlist_size(n: int) -> int:
    return int(n * BUDGET)


def rank_order(scores, player_ids) -> np.ndarray:
    """Indices by score descending, ties by lower player id, NaN last."""
    s = np.nan_to_num(np.asarray(scores, dtype=float), nan=-np.inf)
    return np.lexsort((np.asarray(player_ids), -s))


def budget_metrics(y_true, scores, player_ids) -> dict:
    """Precision/recall of the 10% watchlist and average precision, for ONE cohort-year."""
    y = np.asarray(y_true).astype(int)
    n, pos, k = len(y), int(y.sum()), watchlist_size(len(y))
    hits = int(y[rank_order(scores, player_ids)[:k]].sum())
    s = np.nan_to_num(np.asarray(scores, dtype=float), nan=-np.inf)
    return {"players": n, "breakouts": pos, "watchlist": k, "hits": hits,
            "precision_at_10pct": hits / k if k else np.nan,
            "recall_at_10pct": hits / pos if pos else np.nan,
            "average_precision": average_precision_score(y, s) if pos else np.nan,
            "prevalence": pos / n if n else np.nan}


def metrics_by_year(df: pd.DataFrame, score_col: str, label_col: str = TARGET) -> pd.DataFrame:
    return pd.DataFrame([{"year": y, **budget_metrics(g[label_col], g[score_col], g[PLAYER])}
                         for y, g in df.groupby("year")])


# ----------------------------------------------------------------------------- output
COLORS = {"all": "#2a78d6", "young": "#eb6834", "model": "#2a78d6", "ea": "#eb6834",
          "third": "#1baf7a", "both": "#1baf7a", "random": "#8c8b86", "muted": "#898781",
          "ink": "#0b0b0b", "ink2": "#52514e", "grid": "#e1e0d9", "surface": "#fcfcfb"}
DIVERGING = ["#104281", "#2a78d6", "#86b6ef", "#f0efec", "#f1a3a2", "#e34948", "#a32222"]


def set_style():
    import matplotlib as mpl
    mpl.rcParams.update({
        "figure.facecolor": COLORS["surface"], "axes.facecolor": COLORS["surface"],
        "savefig.facecolor": COLORS["surface"], "axes.edgecolor": "#c3c2b7",
        "axes.labelcolor": COLORS["ink2"], "xtick.color": COLORS["ink2"],
        "ytick.color": COLORS["ink2"], "text.color": COLORS["ink"], "axes.grid": True,
        "grid.color": COLORS["grid"], "grid.linewidth": 0.6, "axes.spines.top": False,
        "axes.spines.right": False, "axes.axisbelow": True,
        "font.family": ["Segoe UI", "DejaVu Sans", "sans-serif"], "font.size": 10,
        "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "legend.frameon": False, "figure.dpi": 110, "savefig.dpi": 150, "savefig.bbox": "tight"})


def diverging_cmap():
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list("blue_red", DIVERGING)


def save(obj, folder: Path, name: str, **kw) -> Path:
    """Save a DataFrame as CSV (4 decimals) or a matplotlib figure as PNG."""
    folder.mkdir(parents=True, exist_ok=True)
    if isinstance(obj, pd.DataFrame):
        path = folder / f"{name}.csv"
        obj.to_csv(path, index=kw.get("index", False), float_format=kw.get("fmt", "%.4f"))
    else:
        path = folder / f"{name}.png"
        obj.savefig(path)
    return path
