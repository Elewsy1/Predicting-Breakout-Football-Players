"""Build August scouting cohorts and reproduce the first check-in."""
import argparse
import math
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
from reportlab.graphics import renderSVG
from reportlab.graphics.charts.barcharts import HorizontalBarChart
from reportlab.graphics.shapes import Drawing, String
from reportlab.lib.colors import HexColor

PLAYER = "player_api_id"
SKILLS = """crossing finishing heading_accuracy short_passing volleys dribbling curve
free_kick_accuracy long_passing ball_control acceleration sprint_speed agility reactions
balance shot_power jumping stamina strength long_shots aggression interceptions
positioning vision penalties marking standing_tackle sliding_tackle gk_diving
gk_handling gk_kicking gk_positioning gk_reflexes""".split()


def load_ratings(db):
    ratings = pd.read_sql("SELECT * FROM Player_Attributes", db, parse_dates=["date"])
    fields = ["overall_rating", "potential", *SKILLS]
    missing_overall = ratings.overall_rating.isna().sum()
    print(f"Ratings: {len(ratings):,} rows, {missing_overall:,} missing overall.")
    ratings = ratings.dropna(subset=[PLAYER, "date", "overall_rating"]).copy()
    for field in fields:
        ratings[field] = pd.to_numeric(ratings[field], errors="raise")
        if not ratings[field].dropna().between(0, 100).all():
            raise ValueError(f"Rating outside 0–100: {field}")
    ratings = ratings.sort_values(["date", "id"])
    duplicate = ratings.duplicated([PLAYER, "date"], keep="last")
    print(f"Duplicate valid player/date records removed: {duplicate.sum()}")
    return ratings.loc[~duplicate, [PLAYER, "date", *fields]]


def match_features(db, year):
    # Use the named season, including July fixtures, up to the forecast date.
    basic = ["id", "season", "league_id", "home_team_goal", "away_team_goal"]
    columns = basic + [f"{side}_team_api_id" for side in ("home", "away")]
    for side in ("home", "away"):
        for slot in range(1, 12):
            columns.extend([f"{side}_player_{slot}", f"{side}_player_Y{slot}"])
    query = f"SELECT {', '.join(columns)} FROM Match WHERE season=? AND date<?"
    matches = pd.read_sql(query, db, params=(f"{year-1}/{year}", f"{year}-08-01"))
    appearances, teams = [], []
    for side, other in [("home", "away"), ("away", "home")]:
        match = matches[["id", "league_id", f"{side}_team_api_id"]].copy()
        match.columns = ["match", "league_id", "team"]
        goals, against = matches[f"{side}_team_goal"], matches[f"{other}_team_goal"]
        match["points"] = (3 * (goals > against) + (goals == against)).astype(int)
        teams.append(match)
        for slot in range(1, 12):
            row = match.assign(player_api_id=matches[f"{side}_player_{slot}"],
                               pitch_y=matches[f"{side}_player_Y{slot}"])
            appearances.append(row.dropna(subset=[PLAYER]))
    starts = pd.concat(appearances, ignore_index=True).drop_duplicates(["match", PLAYER])
    starts[PLAYER] = starts[PLAYER].astype(int)
    # Pitch Y increases towards the opponent's goal in the lineup data.
    starts["position"] = pd.cut(starts.pitch_y, [0, 1, 5, 8, 11],
                                labels=["Goalkeeper", "Defender", "Midfielder", "Forward"])
    groups = starts.groupby(PLAYER)
    player = groups.size().rename("starts").to_frame()
    positions = starts.dropna(subset=["position"]).value_counts([PLAYER, "position"])
    modes = positions.rename("count").reset_index().sort_values(
        [PLAYER, "count", "position"], ascending=[True, False, True])
    player["position"] = modes.drop_duplicates(PLAYER).set_index(PLAYER).position
    # Assign transfers to the team where they started most often.
    counts = starts.groupby([PLAYER, "team", "league_id"]).size().rename("club_starts")
    main = counts.reset_index().sort_values([PLAYER, "club_starts", "team"],
                                           ascending=[True, False, True]).drop_duplicates(PLAYER)
    team_stats = pd.concat(teams).groupby(["team", "league_id"]).points.agg(
        club_matches="size", club_ppg="mean")
    main = main.join(team_stats, on=["team", "league_id"]).set_index(PLAYER)
    main["start_share"] = main.club_starts / main.club_matches
    return player.join(main)


def build_cohorts(db, out):
    ratings = load_ratings(db)
    people = pd.read_sql("SELECT player_api_id, player_name, birthday FROM Player", db,
                         parse_dates=["birthday"]).set_index(PLAYER, verify_integrity=True)
    leagues = pd.read_sql("SELECT id, name FROM League", db).set_index("id").name
    panels, audit = [], []
    for year in range(2009, 2016):
        cutoff = pd.Timestamp(year, 8, 1)
        end = cutoff + pd.DateOffset(years=1)

        def snapshot(start, stop):
            window = ratings.loc[(ratings.date >= start) & (ratings.date < stop)]
            return window.drop_duplicates(PLAYER, keep="last").set_index(PLAYER)

        panel = match_features(db, year).join(people)
        panel["league"] = panel.league_id.map(leagues)
        panel["age"] = year - panel.birthday.dt.year - (
            (panel.birthday.dt.month > 8) |
            ((panel.birthday.dt.month == 8) & (panel.birthday.dt.day > 1))).astype(int)
        panel["age_years"] = (cutoff - panel.birthday).dt.days / 365.25
        current = snapshot(cutoff - pd.DateOffset(years=1), cutoff)
        panel = panel.join(current.rename(columns={"date": "rating_date"}))
        panel["potential_gap"] = panel.potential - panel.overall_rating
        team = panel.groupby("team").overall_rating
        peers = team.transform("count") - 1
        panel["rating_vs_teammates"] = panel.overall_rating - (
            (team.transform("sum") - panel.overall_rating) / peers.where(peers > 0))
        future = snapshot(cutoff, end)[["date", "overall_rating"]]
        panel = panel.join(future.set_axis(["outcome_date", "next_rating"], axis=1))
        valid_now = panel.overall_rating.notna()
        usable = panel.loc[valid_now & panel.next_rating.notna()].copy()
        complete = ratings.date.max() >= end - pd.Timedelta(days=1)
        audit.append([year, len(panel), valid_now.sum(), len(usable), complete,
                      ratings.date.max().date()])
        usable["year"] = year
        usable["split"] = ("incomplete" if not complete else "holdout" if year >= 2014
                           else "validation" if year >= 2011 else "training")
        usable["change"] = usable.next_rating - usable.overall_rating
        usable["annual_median_change"] = usable.change.median()
        usable["breakout"] = (usable.change >= usable.annual_median_change + 3).astype(int)
        panels.append(usable.reset_index())
    coverage = pd.DataFrame(audit, columns=["year", "candidates", "current_rating",
                                            "paired_ratings", "full_window", "last_source_rating"])
    coverage.to_csv(out / "coverage.csv", index=False)
    data = pd.concat(panels, ignore_index=True)
    assert not data.duplicated(["year", PLAYER]).any()
    assert data.age.between(14, 50).all() and data.start_share.between(0, 1).all()
    data.to_csv(out / "cleaned_player_years.csv.gz", index=False, float_format="%.7g")
    print(coverage.to_string(index=False))
    return data


def explore(data, out):
    dev = data[data.split.isin(["training", "validation"])].copy()
    preview = dev[dev.year == 2011].sample(min(20, (dev.year == 2011).sum()), random_state=12)
    preview[["year", PLAYER, "player_name", "age_years", "overall_rating", "potential_gap",
             "starts", "position", "next_rating", "change", "annual_median_change",
             "breakout"]].to_csv(out / "cleaned_preview.csv", index=False, float_format="%.4f")
    dev["age_band"] = pd.cut(dev.age, [0, 20, 24, 29, 34, 100],
                             labels=["≤20", "21–24", "25–29", "30–34", "35+"])
    dev["rating_band"] = pd.cut(dev.overall_rating, [0, 59, 69, 79, 100],
                                labels=["≤59", "60–69", "70–79", "80+"])
    tables = []
    for field in ["age_band", "league", "position", "rating_band"]:
        group = dev[field].astype("string").fillna("Unknown")
        stats = dev.groupby(group).breakout.agg(players="size", breakouts="sum", rate="mean")
        stats.index.name = "group"
        if field in ["age_band", "rating_band"]:
            stats = stats.reindex(dev[field].cat.categories).dropna()
            stats.index.name = "group"
        tables.append(stats.reset_index().assign(variable=field))
        height = 80 + 26 * len(stats)
        drawing = Drawing(610, height)
        drawing.add(String(12, height-20, f"Breakout rate by {field.replace('_', ' ')}",
                           fontName="Helvetica-Bold", fontSize=16))
        drawing.add(String(12, height-37, "2009–2013 pooled player-years  |  percent",
                           fontSize=10))
        chart = HorizontalBarChart()
        chart.x, chart.y, chart.width, chart.height = 175, 24, 375, height-72
        chart.data = [(100 * stats.rate).tolist()[::-1]]
        chart.categoryAxis.categoryNames = stats.index.astype(str).tolist()[::-1]
        chart.valueAxis.valueMin = 0
        chart.valueAxis.valueMax = max(40, math.ceil(10 * stats.rate.max()) * 10)
        chart.valueAxis.valueStep = 10
        chart.bars[0].fillColor = HexColor("#187c77")
        chart.bars[0].strokeColor = None
        chart.barLabelFormat = "%.1f"
        drawing.add(chart)
        renderSVG.drawToFile(drawing, str(out / f"rates_by_{field}.svg"))
    pd.concat(tables).to_csv(out / "breakout_rates.csv", index=False)
    dev[SKILLS].corr().to_csv(out / "skill_correlations.csv")
    features = ["age", "overall_rating", "potential", "potential_gap", "starts",
                "start_share", "club_ppg", "rating_vs_teammates", *SKILLS]
    pd.DataFrame({"pearson_with_breakout": dev[features].corrwith(dev.breakout),
                  "nonmissing": dev[features].count()}).to_csv(out / "feature_correlations.csv")


def baselines(data, out):
    rows = []
    validation = data[data.split == "validation"]
    for cohort, subset in [("All", validation), ("Age ≤24", validation[validation.age_years <= 24])]:
        for year, group in subset.groupby("year"):
            # Exactly floor(10%) places. Lower player ID resolves equal scores.
            n, positives = len(group), group.breakout.sum()
            k = n // 10
            ranked = group.sort_values(["potential_gap", PLAYER], ascending=[False, True])
            hits = ranked.head(k).breakout.sum()
            # Average precision integrates precision at each distinct score threshold.
            scores = group.potential_gap.fillna(-np.inf)
            bins = group.groupby(scores).breakout.agg(["sum", "size"]).sort_index(ascending=False)
            ap = (bins["sum"] / positives * bins["sum"].cumsum() / bins["size"].cumsum()).sum()
            for strategy, found, precision, recall, pr_auc in [
                ("EA gap", hits, hits/k, hits/positives, ap),
                ("Random expectation", k*positives/n, positives/n, k/n, positives/n),
            ]:
                rows.append([cohort, year, strategy, n, positives, k, found, precision,
                             recall, pr_auc])
    result = pd.DataFrame(rows, columns=["cohort", "year", "strategy", "players", "breakouts",
                          "watchlist", "hits", "precision_at_10pct", "recall_at_10pct",
                          "average_precision"])
    result.to_csv(out / "baseline.csv", index=False)
    print(result.to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("database", type=Path)
    parser.add_argument("--out", type=Path, default=Path("checkin"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(args.database.resolve().as_uri() + "?mode=ro", uri=True) as db:
        panel = build_cohorts(db, args.out)
    explore(panel, args.out)
    baselines(panel, args.out)
