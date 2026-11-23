"""Step 1 - cleaning: audit the source tables and build one row per player and cohort year.

Run: python -m src.cleaning   (writes A_checkin_1124/data/ and the baseline table)
"""
import os

import numpy as np
import pandas as pd

from src.common import (A_TABLES, DATA, PLAYER, SKILLS, TARGET, VALIDATION_YEARS, YOUTH_MAX_AGE,
                        cohort, connect, load_ratings, metrics_by_year, save, spike_mask)

# Pitch depth Y (1 = own goal, 11 = attack): 1 goalkeeper, 3 back line, 5-9 midfield lines
# (incl. the three behind a lone striker and a second striker), 10-11 strikers. Y=0 only occurs
# with X=0 in 22 slots of two 2008/09 leagues: a placeholder, treated as missing.
POSITION_BINS, POSITIONS = [0, 1, 4, 9, 11], ["Goalkeeper", "Defender", "Midfielder", "Forward"]
GAMES_PER_TEAM = {19694: 38, 24558: 36}   # Scotland, Switzerland: not a plain double round robin
QUALITY_COLUMNS = ["horizon_days", "position_missing", "club_lineup_coverage",
                   "league_match_coverage"]   # appended at the end of the table
audit = []


def note(area, check, value, action=""):
    audit.append({"area": area, "check": check, "value": value, "action": action})


def clean_ratings(db):
    raw = pd.read_sql("SELECT id, overall_rating, attacking_work_rate, defensive_work_rate "
                      "FROM Player_Attributes", db)
    note("ratings", "rows", len(raw))
    note("ratings", "duplicate row ids", raw.id.duplicated().sum())
    note("ratings", "rows with no overall rating (all fields empty)",
         raw.overall_rating.isna().sum(), "removed")
    for field in ["attacking_work_rate", "defensive_work_rate"]:
        v = raw[field].astype("string").str.strip().str.lower()
        note("ratings", f"{field}: values outside low/medium/high",
             (v.notna() & ~v.isin(["low", "medium", "high"])).sum(), "missing in features.py")
    r = load_ratings(db, ["potential", *SKILLS])
    for field in ["overall_rating", "potential", *SKILLS]:
        r[field] = pd.to_numeric(r[field], errors="raise")
        if not r[field].dropna().between(0, 100).all():
            raise ValueError(f"Rating outside 0-100: {field}")
    note("ratings", "numeric fields outside 0-100", 0, "checked")
    seven = r[["volleys", "curve", "agility", "balance", "jumping", "vision",
               "sliding_tackle"]].isna().all(axis=1)
    note("ratings", "valid rows missing the same 7 skills (dated 2007-02 to 2010-02)",
         seven.sum(), "kept; left missing for imputation")
    note("ratings", "valid rows with potential < overall", (r.potential < r.overall_rating).sum(),
         "kept (EA's own values)")
    note("ratings", "duplicate valid player/date records",
         raw.overall_rating.notna().sum() - len(r), "keep the last by id")
    spike = spike_mask(r)
    note("ratings", "isolated spikes (>= 8 from both neighbours, which agree within 2)",
         spike.sum(), "removed")
    r = r[~spike]
    note("ratings", "rows used", len(r))
    return r.drop(columns="id")


def clean_players(db):
    p = pd.read_sql("SELECT player_api_id, player_name, birthday, height, weight FROM Player", db,
                    parse_dates=["birthday"])
    if not p[PLAYER].is_unique:
        raise ValueError("Duplicate player_api_id in Player")
    kg, inches = p.weight * 0.45359237, p.height / 2.54
    for check, value, action in [
            ("rows", len(p), ""), ("duplicate player_api_id", 0, "checked"),
            ("players sharing a name (different birthdays)",
             p.player_name.duplicated(keep=False).sum(), "kept; IDs, not names, are keys"),
            ("missing birthday/height/weight", p[["birthday", "height", "weight"]].isna().sum().sum(), ""),
            ("birthday range", f"{p.birthday.min().date()} to {p.birthday.max().date()}", ""),
            ("heights that are whole inches in cm",
             f"{np.isclose(inches, inches.round(), atol=0.01).mean():.1%}", ""),
            ("height cm range", f"{p.height.min():.1f}-{p.height.max():.1f}", "plausible; kept"),
            ("weight range lb (kg)", f"{p.weight.min()}-{p.weight.max()} ({kg.min():.1f}-{kg.max():.1f})",
             "plausible; kg conversion in features.py"),
            ("BMI range", f"{(kg / (p.height / 100) ** 2).min():.1f}-"
                          f"{(kg / (p.height / 100) ** 2).max():.1f}", "no outliers removed")]:
        note("players", check, value, action)
    return p[[PLAYER, "player_name", "birthday"]].set_index(PLAYER)


def load_lineups(db):
    """Matches, and one row per line-up slot (match, side, team, player id, pitch X/Y)."""
    base = ["id", "season", "league_id", "date", "home_team_api_id", "away_team_api_id",
            "home_team_goal", "away_team_goal"]
    slot_cols = [f"{s}_player_{k}{i}" for s in ("home", "away") for i in range(1, 12)
                 for k in ("", "X", "Y")]
    m = pd.read_sql(f"SELECT {', '.join(base + slot_cols)} FROM Match", db, parse_dates=["date"])
    slots = pd.concat([pd.DataFrame({
        "match": m.id, "season": m.season, "league_id": m.league_id, "side": s, "slot": i,
        "team": m[f"{s}_team_api_id"], PLAYER: m[f"{s}_player_{i}"],
        "pitch_x": m[f"{s}_player_X{i}"], "pitch_y": m[f"{s}_player_Y{i}"]})
        for s in ("home", "away") for i in range(1, 12)], ignore_index=True)
    return m[base], slots


def audit_matches(db, m, slots):
    """Match and line-up checks; returns each league-season's share of expected matches."""
    note("matches", "rows", len(m))
    note("matches", "duplicate ids / (league, season, date, home, away)",
         m.id.duplicated().sum() + m.duplicated(["league_id", "season", "date",
                                                 "home_team_api_id", "away_team_api_id"]).sum())
    note("matches", "home team equals away team", (m.home_team_api_id == m.away_team_api_id).sum())
    first = m.season.str[:4].astype(int)
    note("matches", "dates outside 1 Jul-30 Jun of the named season",
         ((m.date < pd.to_datetime(first.astype(str) + "-07-01"))
          | (m.date > pd.to_datetime((first + 1).astype(str) + "-06-30"))).sum())
    goals = m[["home_team_goal", "away_team_goal"]]
    note("matches", "missing or negative goals", (goals.isna() | (goals < 0)).sum().sum())
    note("matches", "max goals by one team", int(goals.max().max()), "plausible; kept")
    teams = set(pd.read_sql("SELECT team_api_id FROM Team", db).team_api_id)
    note("matches", "team ids missing from Team table",
         (~m.home_team_api_id.isin(teams)).sum() + (~m.away_team_api_id.isin(teams)).sum())
    side_teams = slots.drop_duplicates(["match", "side"])
    note("matches", "teams in two leagues in one season",
         (side_teams.groupby(["season", "team"]).league_id.nunique() > 1).sum())
    # Completeness: observed / expected matches. A season with far fewer clubs than usual is
    # truncated (Belgium 2013/14: 4 clubs, 12 play-off games), so use the league's usual size.
    clubs = side_teams.groupby(["league_id", "season"]).team.nunique()
    usual = clubs.groupby(level="league_id").transform("median")
    clubs = clubs.where(clubs >= 0.75 * usual, usual)
    games = clubs.index.get_level_values(0).map(GAMES_PER_TEAM)
    expected = clubs * np.where(pd.isna(games), 2 * (clubs - 1), games) / 2
    coverage = (m.groupby(["league_id", "season"]).size() / expected).rename("league_match_coverage")
    for (league, season), share in coverage[coverage < 0.95].items():
        note("matches", f"league {league} {season}: share of expected matches present",
             round(share, 3), "flagged via league_match_coverage")
    side = slots.groupby(["match", "season", "league_id", "side"]).agg(
        missing_ids=(PLAYER, lambda s: s.isna().sum()),
        missing_y=("pitch_y", lambda s: s.isna().sum())).reset_index()
    known = slots.dropna(subset=[PLAYER])
    twice = known[known.duplicated(["match", PLAYER], keep=False)].groupby(["match", PLAYER]).side.nunique()
    for check, value, action in [
            ("lineup slots without a player id", int(side.missing_ids.sum()),
             "starts undercounted; flagged via club_lineup_coverage"),
            ("team lineups with all 11 ids missing", (side.missing_ids == 11).sum(), ""),
            ("matches with both lineups fully missing",
             (side.groupby("match").missing_ids.min() == 11).sum(), ""),
            ("player listed for both teams in one match (away slot is the error)", twice.gt(1).sum(),
             "home entry kept"),
            ("player listed twice in one lineup", twice.eq(1).sum(), "counted once"),
            ("slots with pitch Y = 0 (placeholder, X = 0)", (slots.pitch_y == 0).sum(),
             "treated as missing position"),
            ("slots with a player id but no pitch Y", (slots[PLAYER].notna() & slots.pitch_y.isna()).sum(),
             "position from other starts, else missing")]:
        note("lineups", check, value, action)
    table = side.groupby(["league_id", "season"]).agg(
        team_lineups=("side", "size"), mean_missing_ids=("missing_ids", "mean"),
        share_lineups_fully_missing=("missing_ids", lambda s: (s == 11).mean()),
        share_slots_without_y=("missing_y", lambda s: s.sum() / (11 * len(s))))
    table = table.join(m.groupby(["league_id", "season"]).size().rename("matches")).join(coverage)
    table = table.reset_index()
    table.insert(1, "league", table.league_id.map(pd.read_sql("SELECT id, name FROM League", db)
                                                  .set_index("id").name))
    table.round(4).to_csv(DATA / "lineup_coverage.csv", index=False)
    return coverage


def season_features(m, slots, year, coverage):
    """Starts, position and main-club features for everyone who started in season year-1/year."""
    season = f"{year-1}/{year}"
    played = m[(m.season == season) & (m.date < pd.Timestamp(year, 8, 1))]
    points = pd.concat([pd.DataFrame({
        "team": played[f"{s}_team_api_id"], "league_id": played.league_id,
        "points": 3 * (played[f"{s}_team_goal"] > played[f"{o}_team_goal"])
        + (played[f"{s}_team_goal"] == played[f"{o}_team_goal"])})
        for s, o in [("home", "away"), ("away", "home")]])
    lineup = slots[slots.match.isin(played.id)]
    lineup = lineup.assign(away=lineup.side.eq("away")).sort_values(["match", "away", "slot"])
    filled = lineup.groupby(["team", "league_id"])[PLAYER].count().rename("filled_slots")
    # Home rows come first, so a player wrongly listed for both teams keeps the home entry.
    starts = lineup.dropna(subset=[PLAYER]).drop_duplicates(["match", PLAYER]).copy()
    starts[PLAYER] = starts[PLAYER].astype(int)
    starts["position"] = pd.cut(starts.pitch_y.where(starts.pitch_y > 0), POSITION_BINS,
                                labels=POSITIONS)
    player = starts.groupby(PLAYER).size().rename("starts").to_frame()
    modes = (starts.dropna(subset=["position"]).value_counts([PLAYER, "position"]).rename("count")
             .reset_index().sort_values([PLAYER, "count", "position"], ascending=[True, False, True]))
    player["position"] = modes.drop_duplicates(PLAYER).set_index(PLAYER).position
    # A transferred player belongs to the club he started for most often (ties: lower id).
    main = (starts.groupby([PLAYER, "team", "league_id"]).size().rename("club_starts").reset_index()
            .sort_values([PLAYER, "club_starts", "team"], ascending=[True, False, True])
            .drop_duplicates(PLAYER))
    clubs = points.groupby(["team", "league_id"]).points.agg(club_matches="size", club_ppg="mean")
    main = main.join(clubs, on=["team", "league_id"]).join(filled, on=["team", "league_id"])
    main["start_share"] = main.club_starts / main.club_matches
    main["club_lineup_coverage"] = main.pop("filled_slots") / (11 * main.club_matches)
    main["league_match_coverage"] = main.league_id.map(coverage.xs(season, level="season"))
    return player.join(main.set_index(PLAYER))


def build_table(db):
    ratings, people = clean_ratings(db), clean_players(db)
    m, slots = load_lineups(db)
    coverage = audit_matches(db, m, slots)
    leagues = pd.read_sql("SELECT id, name FROM League", db).set_index("id").name
    last_rating = ratings.date.max()

    def snapshot(start, stop):   # latest rating per player inside [start, stop)
        w = ratings[(ratings.date >= start) & (ratings.date < stop)]
        return w.sort_values("date").drop_duplicates(PLAYER, keep="last").set_index(PLAYER)

    panels, years, candidates = [], [], []
    for year in range(2009, 2016):
        cutoff = pd.Timestamp(year, 8, 1)
        end = cutoff + pd.DateOffset(years=1)
        p = season_features(m, slots, year, coverage).join(people)
        p["league"] = p.league_id.map(leagues)
        b = p.birthday.dt
        p["age"] = year - b.year - ((b.month > 8) | ((b.month == 8) & (b.day > 1))).astype(int)
        p["age_years"] = (cutoff - p.birthday).dt.days / 365.25
        p = p.join(snapshot(cutoff - pd.DateOffset(years=1), cutoff).rename(columns={"date": "rating_date"}))
        p["potential_gap"] = p.potential - p.overall_rating
        team = p.groupby("team").overall_rating
        peers = team.transform("count") - 1
        p["rating_vs_teammates"] = p.overall_rating - (
            (team.transform("sum") - p.overall_rating) / peers.where(peers > 0))
        p = p.join(snapshot(cutoff, end)[["date", "overall_rating"]]
                   .set_axis(["outcome_date", "next_rating"], axis=1))
        now, paired = p.overall_rating.notna(), p.overall_rating.notna() & p.next_rating.notna()
        candidates.append(p[now].assign(year=year, included=paired[now]))
        complete = last_rating >= end - pd.Timedelta(days=1)
        u = p[paired].copy()
        years.append([year, len(p), now.sum(), len(u), complete, last_rating.date(), now.sum() - len(u)])
        u["year"] = year
        u["split"] = ("incomplete" if not complete else "holdout" if year >= 2014
                      else "validation" if year >= 2011 else "training")
        u["change"] = u.next_rating - u.overall_rating
        u["annual_median_change"] = u.change.median()
        u[TARGET] = (u.change >= u.annual_median_change + 3).astype(int)
        u["horizon_days"] = (u.outcome_date - u.rating_date).dt.days   # sensitivity only
        u["position_missing"] = u.position.isna().astype(int)
        panels.append(u.reset_index())
    cov = pd.DataFrame(years, columns=["year", "candidates", "current_rating", "paired_ratings",
                                       "full_window", "last_source_rating", "no_followup"])
    cov["no_followup_rate"] = (cov.no_followup / cov.current_rating).round(4)
    cov.to_csv(DATA / "coverage.csv", index=False)
    data = pd.concat(panels, ignore_index=True)
    data = data[[c for c in data.columns if c not in QUALITY_COLUMNS] + QUALITY_COLUMNS]
    assert not data.duplicated(["year", PLAYER]).any()
    assert data.age.between(14, 50).all() and data.start_share.between(0, 1).all()
    assert data.club_lineup_coverage.between(0, 1).all()
    selection_bias(pd.concat(candidates))
    print(cov.to_string(index=False))
    return data


def panel_checks(data):
    young, known = cohort(data, "young"), data[data.split != "incomplete"]
    for check, value, action in [
            ("rows (2009-2015) / players", f"{len(data)} / {data[PLAYER].nunique()}", ""),
            (f"young rows (completed age <= {YOUTH_MAX_AGE}) / players",
             f"{len(young)} / {young[PLAYER].nunique()}", ""),
            ("rows missing the 7 late-added skills", data.volleys.isna().sum(), "kept; all in 2009"),
            ("rows with potential < overall", (data.potential < data.overall_rating).sum(), "kept"),
            ("rows without a position", data.position.isna().sum(), "flagged via position_missing"),
            ("rows with |change| >= 10 / >= 15",
             f"{(data.change.abs() >= 10).sum()} / {(data.change.abs() >= 15).sum()}",
             "kept (EA re-ratings)"),
            ("horizon_days min / median / max", f"{data.horizon_days.min()} / "
             f"{data.horizon_days.median():.0f} / {data.horizon_days.max()}", ""),
            ("rows 2009-2014 with horizon < 180 days, by year",
             str(known[known.horizon_days < 180].year.value_counts().sort_index().to_dict()),
             "kept; horizon_days added"),
            ("breakout rate 2009-2013, horizon < 180 days vs >= 180 days",
             "{:.3f} vs {:.3f}".format(*known[known.year < 2014].groupby(
                 known.horizon_days >= 180)[TARGET].mean()), "sensitivity check in evaluation"),
            ("rows with club_lineup_coverage < 0.9", (data.club_lineup_coverage < 0.9).sum(), "kept; flagged"),
            ("rows with league_match_coverage < 0.95", (data.league_match_coverage < 0.95).sum(),
             "kept; flagged")]:
        note("panel", check, value, action)


def selection_bias(c):
    """Starters with a current rating: those with vs without a follow-up rating."""
    c = c.assign(young=(c.age <= YOUTH_MAX_AGE).astype(int))
    cols = ["age", "overall_rating", "potential_gap", "starts", "start_share", "young"]
    summary = c.groupby("included")[cols].mean().T.set_axis(["excluded_mean", "included_mean"], axis=1)
    summary["excluded_median"] = c[~c.included][cols].median()
    summary["included_median"] = c[c.included][cols].median()
    rates = [pd.DataFrame({"variable": ["all"], "group": ["all"], "candidates": [len(c)],
                           "included": [c.included.sum()]})]
    for field in ["year", "league"]:
        g = c.groupby(field).included.agg(candidates="size", included="sum")
        rates.append(g.reset_index().rename(columns={field: "group"}).assign(variable=field))
    rates = pd.concat(rates)[["variable", "group", "candidates", "included"]]
    rates["excluded_rate"] = 1 - rates.included / rates.candidates
    with open(DATA / "selection_bias.csv", "w", encoding="utf-8", newline="") as f:
        f.write("# Part 1: means and medians, excluded vs included starters\n")
        summary.round(4).rename_axis("variable").to_csv(f)
        f.write("# Part 2: exclusion rate by year and league\n")
        rates.round(4).to_csv(f, index=False)


def baselines(data):
    """EA potential-gap ranking vs a random watchlist on the validation years."""
    rows = []
    for name in ["all", "young"]:
        for _, r in metrics_by_year(cohort(data[data.year.isin(VALIDATION_YEARS)], name),
                                    "potential_gap").iterrows():
            rows.append({"cohort": name, "strategy": "EA potential gap", **r})
            rows.append({"cohort": name, "strategy": "random (expected)", **r,
                         "hits": r.watchlist * r.prevalence, "precision_at_10pct": r.prevalence,
                         "recall_at_10pct": r.watchlist / r.players, "average_precision": r.prevalence})
    out = pd.DataFrame(rows).astype({"year": int, "players": int, "breakouts": int, "watchlist": int})
    save(out, A_TABLES, "baseline")
    print(out.round(3).to_string(index=False))


def main():
    DATA.mkdir(parents=True, exist_ok=True)
    with connect() as db:
        data = build_table(db)
    panel_checks(data)
    pd.DataFrame(audit).to_csv(DATA / "cleaning_audit.csv", index=False)
    target = DATA / "cleaned_player_years.csv.gz"
    data.to_csv(f"{target}.tmp", index=False, float_format="%.7g", compression="gzip")
    os.replace(f"{target}.tmp", target)   # readers never see a half-written file
    dev = data[data.year == 2011]
    dev.sample(20, random_state=12)[[
        "year", PLAYER, "player_name", "age_years", "overall_rating", "potential_gap", "starts",
        "position", "next_rating", "change", "annual_median_change", TARGET]].to_csv(
        DATA / "cleaned_preview.csv", index=False, float_format="%.4f")
    baselines(data)


if __name__ == "__main__":
    main()
