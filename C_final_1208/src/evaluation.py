"""Step 4 - evaluation: test-year metrics, player bootstrap, scouting simulation, explanations,
sensitivity checks and an EA-independent check.

Run: python -m src.evaluation   (~7 min; writes tables/ and figures/)
Models are only refitted on 2009-2013 (or earlier years inside the folds); 2014 is only scored.
"""
from __future__ import annotations

import json
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import average_precision_score

from src.common import (FIGURES, TABLES, COHORTS, COLORS, DEV_YEARS, PLAYER, PREDICTIONS, SEED,
                        SKILLS, TARGET, TEST_YEAR, VALIDATION_YEARS, YOUTH_MAX_AGE, budget_metrics,
                        cohort, connect, load_ratings, metrics_by_year, rank_order, save,
                        season_starts, set_style, spike_mask, watchlist_size)
from src.features import FEATURE_SETS, load_feature_table
from src.modelling import build_pipeline

EA = ("ea_gap", "ea_gap_only")
KEY = ["cohort", "train_scope", "model", "feature_set"]
METRICS = ["precision_at_10pct", "recall_at_10pct", "average_precision", "hits"]
SETS = list(FEATURE_SETS)
NAMES = {"all": "All players", "young": "Young players"}


# ----------------------------------------------------------------------------- inputs
def load_inputs():
    """(feature table 2009-2014, validation predictions, test predictions, selected models)."""
    table = load_feature_table()
    table = table[table.year.isin(DEV_YEARS + [TEST_YEAR])].reset_index(drop=True)
    val = pd.read_csv(PREDICTIONS / "validation_predictions.csv.gz")
    test = pd.read_csv(PREDICTIONS / "test_predictions.csv.gz")
    raw = json.loads((TABLES / "selected_models.json").read_text())["cohorts"]
    return table, val, test, {c: raw[c]["primary"] for c in COHORTS}


def fit(sel, train, label=TARGET):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cols = FEATURE_SETS[sel["feature_set"]]
        return build_pipeline(sel["model"], sel["params"], sel["feature_set"]).fit(
            train[cols], train[label].to_numpy())


def score(pipe, rows, feature_set):
    return pipe.predict_proba(rows[FEATURE_SETS[feature_set]])[:, 1]


def train_rows(table, years, scope):
    return cohort(table[table.year.isin(years)], scope)


def key(scope, model, feature_set):
    return f"{model}|{feature_set}|{scope}"


def sel_key(sel):
    return key(sel["train_scope"], sel["model"], sel["feature_set"])


# ----------------------------------------------------------------------------- 1. metrics
def year_metrics(preds):
    """Metrics per year for every prediction group, plus the random (prevalence) expectation."""
    rows = [metrics_by_year(g.drop_duplicates(["year", PLAYER]), "score").assign(**dict(zip(KEY, k)))
            for k, g in preds.groupby(KEY)]
    for (c, y), g in preds.drop_duplicates(["cohort", "year", PLAYER]).groupby(["cohort", "year"]):
        n, pos, k = len(g), int(g[TARGET].sum()), watchlist_size(len(g))
        rows.append(pd.DataFrame([{"cohort": c, "train_scope": "-", "model": "random",
                                   "feature_set": "-", "year": y, "players": n, "breakouts": pos,
                                   "watchlist": k, "hits": k * pos / n, "precision_at_10pct": pos / n,
                                   "recall_at_10pct": k / n, "average_precision": pos / n,
                                   "prevalence": pos / n}]))
    out = pd.concat(rows, ignore_index=True)
    return out[KEY + [c for c in out if c not in KEY]]


# ----------------------------------------------------------------------------- 2. bootstrap
def player_resamples(players, n_boot, seed=SEED):
    """Row indices of player-clustered bootstrap samples (players drawn with replacement,
    each bringing all of his rows across years)."""
    uniq, inv = np.unique(players, return_inverse=True)
    order, counts = np.argsort(inv, kind="stable"), np.bincount(inv)
    starts = np.r_[0, np.cumsum(counts)[:-1]]
    rng = np.random.default_rng(seed)
    for _ in range(n_boot):
        d = rng.integers(0, len(uniq), len(uniq))
        c, s = counts[d], starts[d]
        yield order[np.repeat(s, c) + np.arange(c.sum()) - np.repeat(np.cumsum(c) - c, c)]


def wide_scores(preds, c):
    """Rows (year, player, label), score matrix and column keys for one cohort."""
    p = preds[preds.cohort == c]
    is_ea = (p.model == EA[0]) & (p.feature_set == EA[1])
    if (is_ea & (p.train_scope == c)).any():          # the EA score ignores the training scope
        p = p[~is_ea | (p.train_scope == c)]
    p = p.assign(key=[key(*t) for t in zip(p.train_scope, p.model, p.feature_set)])
    p = p.drop_duplicates(["year", PLAYER, "key"])
    wide = p.pivot(index=["year", PLAYER], columns="key", values="score")
    rows = wide.index.to_frame(index=False)
    rows[TARGET] = p.drop_duplicates(["year", PLAYER]).set_index(["year", PLAYER])[TARGET] \
        .reindex(wide.index).to_numpy().astype(int)
    return rows, wide.to_numpy(dtype=float), list(wide.columns)


def ea_key(preds, c):
    p = preds[(preds.cohort == c) & (preds.model == EA[0])]
    return key(c if (p.train_scope == c).any() else p.train_scope.iloc[0], *EA)


def sample_metrics(rows, S, idx):
    """Per-year precision, recall, AP and hits for every score column (+ random), averaged."""
    y_all, ids_all, years = rows[TARGET].to_numpy(), rows[PLAYER].to_numpy(), rows.year.to_numpy()[idx]
    res = []
    for yv in np.unique(years):
        sub = idx[years == yv]
        y, ids, k = y_all[sub], ids_all[sub], watchlist_size(len(sub))
        pos, out = y.sum(), np.full((S.shape[1] + 1, 4), np.nan)
        for j in range(S.shape[1]):
            hits = y[rank_order(S[sub, j], ids)[:k]].sum()
            ap = average_precision_score(y, np.nan_to_num(S[sub, j], nan=-np.inf)) if pos else np.nan
            out[j] = [hits / k, hits / pos if pos else np.nan, ap, hits]
        out[-1] = [pos / len(y), k / len(y), pos / len(y), k * pos / len(y)]   # random list
        res.append(out)
    return np.mean(res, axis=0)


def cluster_bootstrap(preds, c, n_boot=1000, keys=None):
    """(point [models+1, 4], draws [n_boot, models+1, 4], keys + random). Same draws for every
    model, so differences between models are paired."""
    rows, S, cols = wide_scores(preds, c)
    if keys is not None:
        S, cols = S[:, [cols.index(k) for k in keys]], list(keys)
    point = sample_metrics(rows, S, np.arange(len(rows)))
    draws = np.array([sample_metrics(rows, S, idx) for idx in player_resamples(rows[PLAYER], n_boot)])
    return point, draws, cols + ["random|-|-"]


def ci(d):
    return np.nanpercentile(d, 2.5), np.nanpercentile(d, 97.5)


def bootstrap_tables(val, test, n_boot=1000):
    """CIs for every prediction group and paired differences vs the EA list and random."""
    recs, diffs = [], []
    for period, preds in [("test_2014", test), ("validation_2011_2013", val)]:
        for c in COHORTS:
            point, draws, cols = cluster_bootstrap(preds, c, n_boot)
            refs = [("ea_gap", cols.index(ea_key(preds, c))), ("random", len(cols) - 1)]
            for j, col in enumerate(cols):
                model, fs, scope = col.split("|")
                tag = dict(period=period, cohort=c, train_scope=scope, model=model, feature_set=fs)
                for m, metric in enumerate(METRICS):
                    lo, hi = ci(draws[:, j, m])
                    recs.append({**tag, "metric": metric, "estimate": point[j, m], "ci_low": lo,
                                 "ci_high": hi, "se": np.nanstd(draws[:, j, m])})
                    for name, r in refs:
                        if r != j:
                            dd = draws[:, j, m] - draws[:, r, m]
                            lo, hi = ci(dd)
                            diffs.append({**tag, "versus": name, "metric": metric,
                                          "difference": point[j, m] - point[r, m], "ci_low": lo,
                                          "ci_high": hi, "share_reps_not_better": float(np.mean(dd <= 0))})
    return pd.DataFrame(recs), pd.DataFrame(diffs)


def feature_set_contrasts(val, test, selection, n_boot=1000):
    """Paired contrasts between feature sets of the selected model family: can a model without
    EA's potential beat EA's list, and what do potential and match history add?"""
    pairs = [("rating_no_potential", "ea_gap_only"), ("rating_only", "rating_no_potential"),
             ("combined", "rating_only"), ("combined", "rating_no_potential")]
    out = []
    for period, preds in [("test_2014", test), ("validation_2011_2013", val)]:
        for c in COHORTS:
            s = selection[c]
            keys = {fs: ea_key(preds, c) if fs == EA[1] else key(s["train_scope"], s["model"], fs)
                    for fs in SETS}
            point, draws, cols = cluster_bootstrap(preds, c, n_boot, list(keys.values()))
            ix = {fs: cols.index(k) for fs, k in keys.items()}
            for a, b in pairs:
                for m, metric in enumerate(METRICS):
                    dd = draws[:, ix[a], m] - draws[:, ix[b], m]
                    lo, hi = ci(dd)
                    out.append({"period": period, "cohort": c, "model": s["model"],
                                "contrast": f"{a} - {b}", "metric": metric,
                                "difference": point[ix[a], m] - point[ix[b], m], "ci_low": lo,
                                "ci_high": hi, "share_reps_not_better": float(np.mean(dd <= 0))})
    return pd.DataFrame(out)


# ----------------------------------------------------------------------------- 3. scouting
def watchlist(table, test, sel, c):
    """2014 cohort players with model and EA ranks and watchlist flags."""
    p = test[test.cohort == c]
    m = p[(p.model == sel["model"]) & (p.feature_set == sel["feature_set"])
          & (p.train_scope == sel["train_scope"])].drop_duplicates(PLAYER)
    e = p[(p.model == EA[0]) & (p.feature_set == EA[1])].drop_duplicates(PLAYER)
    cols = [PLAYER, "player_name", "age", "age_years", "overall_rating", "potential",
            "potential_gap", "league", "position", "starts", "start_share", "change", TARGET]
    w = table[table.year == TEST_YEAR][cols] \
        .merge(m[[PLAYER, "score"]].rename(columns={"score": "model_score"}), on=PLAYER) \
        .merge(e[[PLAYER, "score"]].rename(columns={"score": "ea_score"}), on=PLAYER)
    k = watchlist_size(len(w))
    for name in ["model", "ea"]:
        rank = np.empty(len(w), int)
        rank[rank_order(w[f"{name}_score"], w[PLAYER])] = np.arange(1, len(w) + 1)
        w[f"{name}_rank"], w[f"in_{name}"] = rank, rank <= k
    w.insert(0, "cohort", c)
    w["k"] = k
    return w.sort_values("model_rank").reset_index(drop=True)


def scouting_summary(w, hit_draws):
    """Hits and misses of the model list, the EA list, their overlap and a random list."""
    c, k, n, pos = w.cohort.iloc[0], int(w.k.iloc[0]), len(w), int(w[TARGET].sum())
    lo, hi = stats.hypergeom(n, pos, k).ppf([0.025, 0.975])
    rec = []
    for strategy, mask, note in [
            ("model watchlist", w.in_model, ""), ("EA potential-gap watchlist", w.in_ea, ""),
            ("on both lists", w.in_model & w.in_ea, ""), ("model only", w.in_model & ~w.in_ea, ""),
            ("EA only", w.in_ea & ~w.in_model, ""),
            ("union of both lists", w.in_model | w.in_ea, "uses up to 2x the budget")]:
        hits, size = int(w.loc[mask, TARGET].sum()), int(mask.sum())
        rec.append({"strategy": strategy, "players_listed": size, "hits": hits, "misses": size - hits,
                    "precision": hits / size, "share_of_all_breakouts": hits / pos, "note": note})
    rec.append({"strategy": "random watchlist (expected)", "players_listed": k, "hits": k * pos / n,
                "misses": k - k * pos / n, "precision": pos / n, "share_of_all_breakouts": k / n,
                "note": f"95% of random lists hit {int(lo)}-{int(hi)} breakouts"})
    out = pd.DataFrame(rec).assign(cohort=c, cohort_size=n, breakouts_in_cohort=pos, budget=k)
    for strategy, draws in hit_draws.items():
        out.loc[out.strategy == strategy, ["hits_ci_low", "hits_ci_high"]] = np.percentile(draws, [2.5, 97.5])
    out.loc[out.strategy.str.startswith("random"), ["hits_ci_low", "hits_ci_high"]] = [lo, hi]
    return out


def scouting_profile(w):
    """Who each list picks: age, rating, gap, playing time, position and league mix."""
    rows = []
    for name, mask in [("model watchlist", w.in_model), ("EA watchlist", w.in_ea),
                       ("whole cohort", w.index == w.index), ("actual breakouts", w[TARGET] == 1)]:
        g = w[mask]
        top = g.league.value_counts(normalize=True).head(3)
        rows.append({"cohort": w.cohort.iloc[0], "group": name, "players": len(g),
                     "breakout_rate": g[TARGET].mean(), "mean_age": g.age_years.mean(),
                     "share_age_21_or_less": (g.age <= 21).mean(),
                     "mean_overall_rating": g.overall_rating.mean(),
                     "mean_potential_gap": g.potential_gap.mean(),
                     "mean_prev_season_starts": g.starts.mean(),
                     **{f"share_{p.lower()}": (g.position == p).mean()
                        for p in ["Goalkeeper", "Defender", "Midfielder", "Forward"]},
                     "top_leagues": "; ".join(f"{i} {v:.0%}" for i, v in top.items())})
    return pd.DataFrame(rows)


def league_mix(w):
    return pd.concat([w[mask].groupby("league").agg(players=(PLAYER, "size"), breakouts=(TARGET, "sum"))
                      .assign(list=name).reset_index()
                      for name, mask in [("model", w.in_model), ("EA", w.in_ea), ("cohort", w.index == w.index)]]
                     ).assign(cohort=w.cohort.iloc[0])


def precision_curve(w):
    """Precision at budgets of 1-30%, to show the 10% choice is not cherry-picked."""
    y = w[TARGET].to_numpy()
    om, oe = rank_order(w.model_score, w[PLAYER]), rank_order(w.ea_score, w[PLAYER])
    return pd.DataFrame([{"cohort": w.cohort.iloc[0], "budget_fraction": round(f, 2),
                          "k": int(len(w) * f), "model": y[om[:int(len(w) * f)]].mean(),
                          "ea_gap": y[oe[:int(len(w) * f)]].mean(), "random": y.mean()}
                         for f in np.arange(0.01, 0.31, 0.01)])


# ----------------------------------------------------------------------------- 4. explanations
GROUPS = {
    "EA potential (potential, gap, change)": ["potential", "potential_gap", "potential_change_prev_year"],
    "overall rating": ["overall_rating"], "33 skill ratings": SKILLS,
    "rating history": ["rating_change_prev_year", "has_prev_rating", "rating_change_in_window"],
    "age": ["age_years"],
    "body and EA tags": ["height_cm", "weight_kg", "left_footed", "attacking_work_rate_ord",
                         "defensive_work_rate_ord"],
    "league": ["league"],
    "playing time": ["starts", "start_share", "prev_starts", "starts_change", "has_prev_season"],
    "team context": ["club_ppg", "team_mean_rating", "rating_vs_teammates", "team_rating_rank_pct"],
    "position": ["position"], "line-up data coverage": ["club_lineup_coverage"],
}


def permutation_drop(pipe, X, y, ids, groups, n_repeats, seed=SEED):
    """Mean drop in AP and Precision@10% when each group of columns is shuffled jointly
    (joint shuffling keeps correlated columns, like the 33 skills, together)."""
    rng = np.random.default_rng(seed)
    p10 = lambda s: y[rank_order(s, ids)[:watchlist_size(len(y))]].mean()
    s0 = pipe.predict_proba(X)[:, 1]
    base_ap, base_p = average_precision_score(y, s0), p10(s0)
    rows = []
    for name, cols in groups.items():
        ap, pp = [], []
        for _ in range(n_repeats):
            perm, Xp = rng.permutation(len(X)), X.copy()
            for col in cols:
                Xp[col] = X[col].to_numpy()[perm]
            s = pipe.predict_proba(Xp)[:, 1]
            ap.append(base_ap - average_precision_score(y, s))
            pp.append(base_p - p10(s))
        rows.append({"group": name, "columns": len(cols), "ap_drop": np.mean(ap), "p10_drop": np.mean(pp)})
    return pd.DataFrame(rows)


def permutation_importance(table, selection, n_repeats=10):
    """Primary: the rolling validation folds (fit on years < v, shuffle year v). Secondary and
    descriptive: the 2009-2013 refit scored on 2014. Returns (by group, by column) summaries."""
    parts = {"group": [], "column": []}
    for c in COHORTS:
        s = selection[c]
        cols = FEATURE_SETS[s["feature_set"]]
        lookup, groups = {x: g for g, m in GROUPS.items() for x in m}, {}
        for x in cols:                       # groups in the order their first column appears
            groups.setdefault(lookup.get(x, x), []).append(x)
        folds = [(f"validation {v}", list(range(2009, v)), v) for v in VALIDATION_YEARS]
        for name, years, ev_year in folds + [("test 2014 (descriptive)", DEV_YEARS, TEST_YEAR)]:
            pipe = fit(s, train_rows(table, years, s["train_scope"]))
            ev = cohort(table[table.year == ev_year], c)
            args = (pipe, ev[cols], ev[TARGET].to_numpy(), ev[PLAYER].to_numpy())
            for kind, g, n in [("group", groups, n_repeats), ("column", {x: [x] for x in cols}, max(3, n_repeats // 2))]:
                parts[kind].append(permutation_drop(*args, g, n).assign(cohort=c, data=name))
    out = []
    for kind in ["group", "column"]:
        imp = pd.concat(parts[kind])
        val = imp[imp.data.str.startswith("validation")]
        a = val.groupby(["cohort", "group"]).agg(
            ap_drop_validation=("ap_drop", "mean"), ap_drop_fold_min=("ap_drop", "min"),
            ap_drop_fold_max=("ap_drop", "max"), p10_drop_validation=("p10_drop", "mean"))
        t = imp[imp.data.str.startswith("test")].set_index(["cohort", "group"])
        a["ap_drop_test"], a["p10_drop_test"] = t.ap_drop, t.p10_drop
        out.append(a.reset_index().sort_values(["cohort", "ap_drop_validation"], ascending=[True, False]))
    return out


def logistic_setting(selection, c):
    """The selected model if it is a logistic regression, else the best validated logistic
    regression for the same cohort, training scope and feature set."""
    s = selection[c]
    if s["model"] == "logreg":
        return dict(s, source="selected model")
    best = pd.read_csv(TABLES / "best_settings.csv")
    r = best[(best.cohort == c) & (best.model == "logreg") & (best.feature_set == s["feature_set"])
             & (best.train_scope == s["train_scope"])].sort_values(["mean_p10", "mean_ap"], ascending=False).iloc[0]
    return {"model": "logreg", "params": json.loads(r.setting), "feature_set": r.feature_set,
            "train_scope": r.train_scope, "source": f"best validated logreg (mean P@10% "
            f"{r.mean_p10:.3f}); selected model is {s['model']}"}


def pca_components(pipe):
    prep = pipe[0]
    if "skills_pca" not in prep.named_transformers_:
        return None, None
    cols = next(t[2] for t in prep.transformers_ if t[0] == "skills_pca")
    return prep.named_transformers_["skills_pca"][-1].components_, list(cols)


def logistic_coefficients(table, selection, n_boot=200):
    """Standardised coefficients (log-odds per 1 SD) fitted on 2009-2013, with player-bootstrap
    CIs. PCA components get signs aligned to the full fit in every replicate."""
    rows, loads = [], []
    for c in COHORTS:
        lr = logistic_setting(selection, c)
        train = train_rows(table, DEV_YEARS, lr["train_scope"])
        pipe = fit(lr, train)
        names, coef = list(pipe[:-1].get_feature_names_out()), pipe[-1].coef_[0]
        comps, skill_cols = pca_components(pipe)
        for i, comp in enumerate(comps if comps is not None else []):
            top = pd.Series(comp, index=skill_cols)
            for r, (sk, v) in enumerate(top.reindex(top.abs().sort_values(ascending=False).index).items()):
                loads.append({"cohort": c, "component": f"pca{i}", "skill": sk, "loading": v, "rank": r + 1})
        boots = []
        for idx in player_resamples(train[PLAYER].to_numpy(), n_boot):
            p = fit(lr, train.iloc[idx])
            b = pd.Series(p[-1].coef_[0], index=p[:-1].get_feature_names_out())
            pc, _ = pca_components(p)
            for i in range(len(comps) if comps is not None and pc is not None else 0):
                if np.dot(comps[i], pc[i]) < 0:
                    b[f"pca{i}"] *= -1
            boots.append(b.reindex(names).to_numpy())
        boots = np.array(boots)
        for j, nm in enumerate(names):
            lo, hi = ci(boots[:, j])
            rows.append({"cohort": c, "feature_set": lr["feature_set"], "train_scope": lr["train_scope"],
                         "setting": json.dumps(lr["params"]), "source": lr["source"],
                         "term": nm.split("__", 1)[-1], "coefficient_per_sd": coef[j],
                         "odds_ratio": np.exp(coef[j]), "ci_low": lo, "ci_high": hi})
    out = pd.DataFrame(rows)
    out["ci_excludes_zero"] = (out.ci_low > 0) | (out.ci_high < 0)
    out = out.assign(abs=out.coefficient_per_sd.abs()).sort_values(
        ["cohort", "abs"], ascending=[True, False]).drop(columns="abs")
    return out, pd.DataFrame(loads)


# ----------------------------------------------------------------------------- 5. sensitivity
SCENARIOS = {   # name: (label column, row filter, description, short label)
    "base": ("breakout", None, "Main label: last rating in window, threshold median + 3", "Main analysis"),
    "max_rating_label": ("breakout_max", None, "Label uses the MAX rating in the follow-up window "
                         "(median recomputed)", "Label: max rating in window"),
    "horizon_ge_180d": ("breakout", "horizon_days >= 180", "Drop rows whose paired ratings are "
                        "< 180 days apart (train and eval)", "Drop horizon < 180 days"),
    "threshold_plus2": ("breakout_t2", None, "Breakout = change >= median + 2", "Threshold median + 2"),
    "threshold_plus4": ("breakout_t4", None, "Breakout = change >= median + 4", "Threshold median + 4"),
}


def alternative_labels(table):
    """Max-rating label, threshold variants and a re-check of the main label."""
    with connect() as db:
        r = load_ratings(db)
    r = r[~spike_mask(r)]
    parts = []
    for year, g in table.groupby("year"):
        cut = pd.Timestamp(year, 8, 1)
        w = r[(r.date >= cut) & (r.date < cut + pd.DateOffset(years=1))]
        x = g[["year", PLAYER, "overall_rating", "change", "annual_median_change", TARGET, "horizon_days"]]
        x = x.join(w.groupby(PLAYER).overall_rating.max().rename("max_next"), on=PLAYER)
        x["last_next_check"] = x[PLAYER].map(w.drop_duplicates(PLAYER, keep="last").set_index(PLAYER).overall_rating)
        parts.append(x)
    x = pd.concat(parts, ignore_index=True)
    x["label_recheck_ok"] = (x.last_next_check - x.overall_rating) == x.change
    x["change_max"] = x.max_next - x.overall_rating
    x["median_change_max"] = x.groupby("year").change_max.transform("median")
    x["breakout_max"] = (x.change_max >= x.median_change_max + 3).astype(int)
    x["breakout_t2"] = (x.change >= x.annual_median_change + 2).astype(int)
    x["breakout_t4"] = (x.change >= x.annual_median_change + 4).astype(int)
    return x


def sensitivity(table, selection, n_boot=1000):
    """Re-run the selected pipeline (rolling validation + 2014 test) under each scenario."""
    labels = alternative_labels(table)
    tab = table.merge(labels[["year", PLAYER, "breakout_max", "breakout_t2", "breakout_t4"]],
                      on=["year", PLAYER], how="left")
    years, summary = [], []
    for scen, (label, query, desc, _) in SCENARIOS.items():
        t = tab.query(query) if query else tab
        for c in COHORTS:
            s = selection[c]
            for year, tr_years in [(v, list(range(2009, v))) for v in VALIDATION_YEARS] + [(TEST_YEAR, DEV_YEARS)]:
                ev = cohort(t[t.year == year], c)
                s_model = score(fit(s, train_rows(t, tr_years, s["train_scope"]), label), ev, s["feature_set"])
                for name, sc in [("selected", s_model), ("ea_gap", ev.potential_gap.to_numpy())]:
                    years.append({"scenario": scen, "cohort": c, "strategy": name, "year": year,
                                  **budget_metrics(ev[label], sc, ev[PLAYER])})
            # paired bootstrap of the 2014 difference, model minus EA
            long = pd.concat([ev[["year", PLAYER]].assign(**{TARGET: ev[label].to_numpy()}, cohort=c,
                                                          train_scope=c, model=m, feature_set=f, score=sc)
                              for m, f, sc in [("selected", "-", s_model), (*EA, ev.potential_gap.to_numpy())]])
            point, draws, cols = cluster_bootstrap(long, c, n_boot)
            i_s, i_e = cols.index(key(c, "selected", "-")), cols.index(key(c, *EA))
            g = pd.DataFrame([r for r in years if r["scenario"] == scen and r["cohort"] == c])
            val, te = g[g.year.isin(VALIDATION_YEARS)], g[g.year == TEST_YEAR].set_index("strategy")
            vm = val.groupby("strategy")[["precision_at_10pct", "average_precision"]].mean()
            lo, hi = ci(draws[:, i_s, 0] - draws[:, i_e, 0])
            summary.append({
                "scenario": scen, "description": desc, "cohort": c, "rows_2009_2014": len(cohort(t, c)),
                "prevalence_test": te.loc["selected", "prevalence"],
                "val_precision_selected": vm.loc["selected", "precision_at_10pct"],
                "val_precision_ea": vm.loc["ea_gap", "precision_at_10pct"],
                "val_ap_selected": vm.loc["selected", "average_precision"],
                "val_ap_ea": vm.loc["ea_gap", "average_precision"],
                "test_precision_selected": te.loc["selected", "precision_at_10pct"],
                "test_precision_ea": te.loc["ea_gap", "precision_at_10pct"],
                "test_diff": point[i_s, 0] - point[i_e, 0], "test_diff_ci_low": lo, "test_diff_ci_high": hi,
                "test_ap_selected": te.loc["selected", "average_precision"],
                "test_ap_ea": te.loc["ea_gap", "average_precision"]})
    s = pd.DataFrame(summary)
    s["model_beats_ea_validation"] = s.val_precision_selected > s.val_precision_ea
    s["model_beats_ea_test"] = s.test_diff > 0
    stats_ = labels.groupby("year").agg(
        rows=(PLAYER, "size"), breakout=(TARGET, "mean"), breakout_max=("breakout_max", "mean"),
        median_change_max=("median_change_max", "first"), breakout_t2=("breakout_t2", "mean"),
        breakout_t4=("breakout_t4", "mean"), share_horizon_lt_180=("horizon_days", lambda h: (h < 180).mean()),
        label_recheck_ok=("label_recheck_ok", "mean"))
    stats_["max_label_flips"] = (labels[TARGET] != labels.breakout_max).groupby(labels.year).sum()
    return s, pd.DataFrame(years), stats_.reset_index()


# ----------------------------------------------------------------------------- 6. EA-independent check
def next_season_check(watchlists, n_boot=1000):
    """Do watchlisted 2014 players start more 2014/15 league matches? Raw and adjusted (OLS on
    previous starts, age, rating) differences with bootstrap CIs; leavers count as 0 starts."""
    with connect() as db:
        nxt = season_starts(db, f"{TEST_YEAR}/{TEST_YEAR + 1}").starts.rename("next_starts").reset_index()
    rng, rows, detail = np.random.default_rng(SEED), [], []
    for c, w in watchlists.items():
        d = w.merge(nxt, on=PLAYER, how="left").fillna({"next_starts": 0})
        d["start_change"], d["any_start"] = d.next_starts - d.starts, d.next_starts > 0
        detail.append(d)
        yv = d.next_starts.to_numpy(float)
        for lst, label in [("in_model", "model"), ("in_ea", "EA potential gap")]:
            flag = d[lst].to_numpy()
            Z = np.column_stack([np.ones(len(d)), flag.astype(float), d.starts, d.age_years, d.overall_rating])
            adj = lambda i: np.linalg.lstsq(Z[i], yv[i], rcond=None)[0][1]
            raw_b, adj_b = [], []
            for _ in range(n_boot):
                i = rng.integers(0, len(d), len(d))
                raw_b.append(yv[i][flag[i]].mean() - yv[i][~flag[i]].mean())
                adj_b.append(adj(i))
            on, off = d[flag], d[~flag]
            rows.append({"cohort": c, "list": label, "listed": len(on),
                         "mean_next_starts_listed": on.next_starts.mean(),
                         "mean_next_starts_others": off.next_starts.mean(),
                         "share_any_start_listed": on.any_start.mean(),
                         "share_any_start_others": off.any_start.mean(),
                         "mean_change_listed": on.start_change.mean(), "mean_change_others": off.start_change.mean(),
                         "raw_difference": on.next_starts.mean() - off.next_starts.mean(),
                         "raw_ci_low": np.percentile(raw_b, 2.5), "raw_ci_high": np.percentile(raw_b, 97.5),
                         "adjusted_difference": adj(np.arange(len(d))),
                         "adjusted_ci_low": np.percentile(adj_b, 2.5), "adjusted_ci_high": np.percentile(adj_b, 97.5)})
    return pd.DataFrame(rows), pd.concat(detail, ignore_index=True)


# ----------------------------------------------------------------------------- figures
def xerr(point, lo, hi):
    """Error-bar lengths, never negative (a bootstrap interval need not contain the estimate)."""
    return np.clip([np.asarray(point) - lo, np.asarray(hi) - point], 0, None)


def two_panels(title, h=3.6, **kw):
    import matplotlib.pyplot as plt
    set_style()
    fig, axes = plt.subplots(1, 2, figsize=(11, h), **kw)
    fig.suptitle(title, x=0.01, ha="left", fontsize=12)
    for ax, c in zip(axes, COHORTS):
        ax.set_title(NAMES[c])
    return fig, axes


def plot_year_precision(ym, selection):
    fig, axes = two_panels("Precision@10% by cohort year (2011-13 rolling validation, 2014 test)", 3.4, sharey=True)
    for ax, c in zip(axes, COHORTS):
        s, m = selection[c], ym[ym.cohort == c]
        for lab, mask, col in [("Our model", (m.model == s["model"]) & (m.feature_set == s["feature_set"])
                                & (m.train_scope == s["train_scope"]), COLORS["model"]),
                               ("EA potential gap", m.model == EA[0], COLORS["ea"]),
                               ("Random (prevalence)", m.model == "random", COLORS["random"])]:
            d = m[mask].drop_duplicates("year").sort_values("year")
            ax.plot(d.year, 100 * d.precision_at_10pct, marker="o", color=col, lw=2, label=lab,
                    ls="--" if lab.startswith("Random") else "-")
        ax.axvspan(TEST_YEAR - 0.5, TEST_YEAR + 0.5, color=COLORS["grid"], alpha=0.6)
        ax.set(xticks=VALIDATION_YEARS + [TEST_YEAR], ylim=(0, 100))
    axes[0].set_ylabel("Precision@10% (%)")
    axes[0].legend(fontsize=8.5)
    fig.tight_layout()
    return fig


def plot_headline(summ):
    strategies = [("model watchlist", "Our model", COLORS["model"]),
                  ("EA potential-gap watchlist", "EA potential gap", COLORS["ea"]),
                  ("random watchlist (expected)", "Random list", COLORS["random"])]
    fig, axes = two_panels("Watching 10% of the 2014 cohort: how many breakouts does each list catch?", 3.8)
    for ax, c in zip(axes, COHORTS):
        s = summ[summ.cohort == c].set_index("strategy")
        k = int(s.budget.iloc[0])
        for i, (strat, lab, col) in enumerate(strategies):
            r = s.loc[strat]
            ax.barh(2 - i, r.hits, color=col, height=0.6)
            ax.errorbar(r.hits, 2 - i, xerr=xerr([r.hits], r.hits_ci_low, r.hits_ci_high),
                        fmt="none", ecolor=COLORS["ink"], capsize=4, lw=1.2)
            ax.text(r.hits_ci_high + k * 0.015, 2 - i, f"{r.hits:.0f} hits ({r.precision:.0%})", va="center")
        ax.set_yticks([2, 1, 0], [x[1] for x in strategies])
        ax.set(xlim=(0, k * 0.85), xlabel=f"Breakouts found among the {k} watched players")
        ax.set_title(f"{NAMES[c]}{f' (age ≤ {YOUTH_MAX_AGE})' if c == 'young' else ''}: "
                     f"{int(s.cohort_size.iloc[0]):,} players, {int(s.breakouts_in_cohort.iloc[0])} breakouts")
        ax.grid(axis="y", visible=False)
    fig.text(0.01, -0.02, "Error bars: 95% player-bootstrap interval (random: 95% range of random lists). "
             "Hit = rating rose ≥ median + 3 by 1 Aug 2015.", fontsize=8.5, color=COLORS["ink2"])
    fig.tight_layout()
    return fig


def plot_overlap(summ):
    from matplotlib.patches import Patch
    fig, axes = two_panels("How much do the two 2014 watchlists overlap, and where do the extra hits come from?", 3.2)
    for ax, c in zip(axes, COHORTS):
        s = summ[summ.cohort == c].set_index("strategy")
        both = s.loc["on both lists"]
        for y, own, col in [(1, s.loc["model only"], COLORS["model"]), (0, s.loc["EA only"], COLORS["ea"])]:
            left = 0
            for val, colr, alpha in [(both.hits, COLORS["both"], 1), (both.misses, COLORS["both"], .35),
                                     (own.hits, col, 1), (own.misses, col, .35)]:
                ax.barh(y, val, left=left, color=colr, alpha=alpha, height=0.55, edgecolor="white", lw=2)
                if val >= 12:
                    ax.text(left + val / 2, y, f"{val:.0f}", ha="center", va="center", fontsize=9)
                left += val
        ax.set_yticks([1, 0], ["Our model", "EA potential gap"])
        ax.set(xlabel="Players on the watchlist")
        ax.set_title(f"{NAMES[c]}: {both.players_listed:.0f} of {s.budget.iloc[0]} picks shared")
        ax.grid(axis="y", visible=False)
    fig.legend(handles=[Patch(color=COLORS["both"], label="on both lists: breakout"),
                        Patch(color=COLORS["both"], alpha=.35, label="on both lists: no breakout"),
                        Patch(color=COLORS["model"], label="list-specific pick: breakout"),
                        Patch(color=COLORS["model"], alpha=.35, label="list-specific pick: no breakout")],
               loc="lower center", ncol=4, fontsize=8.5, bbox_to_anchor=(0.5, -0.08))
    fig.tight_layout()
    return fig


def plot_budget_curve(curves):
    fig, axes = two_panels("2014 test year: precision at other scouting budgets (dotted line = the 10% plan)", sharey=True)
    for ax, c in zip(axes, COHORTS):
        d = curves[curves.cohort == c]
        for col, lab, colr in [("model", "Our model", COLORS["model"]), ("ea_gap", "EA potential gap", COLORS["ea"]),
                               ("random", "Random", COLORS["random"])]:
            ax.plot(100 * d.budget_fraction, 100 * d[col], color=colr, lw=2, label=lab, ls="--" if col == "random" else "-")
        ax.axvline(10, color=COLORS["ink2"], lw=0.8, ls=":")
        ax.set(xlabel="Share of the cohort watched (%)")
    axes[0].set_ylabel("Precision: % of watched who break out")
    axes[0].legend()
    fig.tight_layout()
    return fig


def plot_feature_sets(boot, selection):
    labels = {"ea_gap_only": "EA gap only (baseline)", "rating_no_potential": "Ratings, no potential",
              "rating_only": "Ratings + potential", "combined": "Ratings + potential + matches"}
    fig, axes = two_panels("Does the model need EA's potential? Precision by feature set", sharex=True)
    for ax, c in zip(axes, COHORTS):
        s = selection[c]
        d = boot[(boot.cohort == c) & (boot.metric == "precision_at_10pct")]
        for off, period, colr, lab in [(0.17, "validation_2011_2013", COLORS["ink2"], "2011-13 validation (mean)"),
                                       (-0.17, "test_2014", COLORS["model"], "2014 test")]:
            for i, fs in enumerate(SETS):
                is_ea = fs == EA[1]
                r = d[(d.period == period) & (d.feature_set == fs) & (d.model == (EA[0] if is_ea else s["model"]))
                      & ((d.train_scope == s["train_scope"]) | is_ea)].iloc[0]
                ax.errorbar(100 * r.estimate, len(SETS) - 1 - i + off, fmt="o", color=colr, ms=7, capsize=3,
                            xerr=100 * xerr([r.estimate], r.ci_low, r.ci_high),
                            label=lab if i == 0 else None)
        prev = d[(d.period == "test_2014") & (d.model == "random")].estimate.iloc[0]
        ax.axvline(100 * prev, color=COLORS["random"], ls="--", lw=1)
        ax.text(100 * prev, -0.45, " random list, 2014", fontsize=8.5, va="center")
        ax.set_yticks(range(len(SETS))[::-1], [labels[f] for f in SETS])
        ax.set(xlim=(0, 85), xlabel="Precision@10% (%), 95% player-bootstrap CI")
        ax.set_title(f"{NAMES[c]} - {s['model']}")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=2, fontsize=9, bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout()
    return fig


def plot_importance(imp, top=12):
    fig, axes = two_panels("Permutation importance of feature groups (bars: mean of 2011-13 folds, whiskers: fold range)", 4.2)
    for ax, c in zip(axes, COHORTS):
        d = imp[imp.cohort == c].head(top).iloc[::-1]
        y = np.arange(len(d))
        ax.barh(y, d.ap_drop_validation, color=COLORS["model"], height=0.6)
        ax.errorbar(d.ap_drop_validation, y, fmt="none", ecolor=COLORS["ink2"], capsize=3, lw=1,
                    xerr=xerr(d.ap_drop_validation, d.ap_drop_fold_min, d.ap_drop_fold_max))
        ax.scatter(d.ap_drop_test, y, color=COLORS["ea"], zorder=3, s=22, label="2014 test (descriptive)")
        ax.set_yticks(y, d.group, fontsize=9)
        ax.axvline(0, color=COLORS["ink2"], lw=0.8)
        ax.set(xlabel="Drop in average precision when shuffled")
    axes[0].legend(fontsize=8.5, loc="lower right")
    fig.tight_layout()
    return fig


def plot_coefficients(coefs, top=15):
    fig, axes = two_panels("Standardised logistic coefficients, fitted on 2009-2013 (95% player bootstrap)", 5)
    for ax, c in zip(axes, COHORTS):
        d = coefs[(coefs.cohort == c) & ~coefs.term.str.startswith(("league_", "position_"))].head(top).iloc[::-1]
        y = np.arange(len(d))
        ax.errorbar(d.coefficient_per_sd, y, fmt="none", ecolor=COLORS["ink2"], capsize=2, lw=1,
                    xerr=xerr(d.coefficient_per_sd, d.ci_low, d.ci_high))
        ax.scatter(d.coefficient_per_sd, y, s=30, zorder=3,
                   c=np.where(d.coefficient_per_sd > 0, COLORS["model"], COLORS["ea"]))
        ax.axvline(0, color=COLORS["ink2"], lw=0.8)
        ax.set_yticks(y, d.term, fontsize=8.5)
        ax.set(xlabel="Log-odds change per 1 SD, other features fixed (league/position: see table)")
        ax.set_title(f"{NAMES[c]} ({d.feature_set.iloc[0]})")
    fig.tight_layout()
    return fig


def plot_sensitivity(sens):
    fig, axes = two_panels("Sensitivity: does the model's edge over EA survive other label choices?", 3.4, sharex=True)
    for ax, c in zip(axes, COHORTS):
        d = sens[sens.cohort == c].set_index("scenario").reindex(list(SCENARIOS))
        y = np.arange(len(d))[::-1]
        ax.errorbar(100 * d.test_diff, y, fmt="o", color=COLORS["model"], capsize=3, label="2014 test (95% CI)",
                    xerr=100 * xerr(d.test_diff, d.test_diff_ci_low, d.test_diff_ci_high))
        ax.scatter(100 * (d.val_precision_selected - d.val_precision_ea), y, marker="D", color=COLORS["ink2"],
                   s=25, zorder=3, label="2011-13 validation mean")
        ax.axvline(0, color=COLORS["ink2"], lw=0.8)
        ax.set_yticks(y, [v[3] for v in SCENARIOS.values()], fontsize=9)
        ax.set(xlabel="Precision@10%: model minus EA gap (percentage points)")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=2, fontsize=9, bbox_to_anchor=(0.5, -0.07))
    fig.tight_layout()
    return fig


# ----------------------------------------------------------------------------- run everything
def main(n_boot=1000, n_repeats=10, coef_boot=200):
    import matplotlib.pyplot as plt
    table, val, test, selection = load_inputs()
    r = {"selection": selection}
    r["year_metrics"] = pd.concat([year_metrics(val).assign(period="validation"),
                                   year_metrics(test).assign(period="test")], ignore_index=True)
    r["bootstrap"], r["differences"] = bootstrap_tables(val, test, n_boot)
    r["feature_set_contrasts"] = feature_set_contrasts(val, test, selection, n_boot)
    r["watchlists"] = {c: watchlist(table, test, selection[c], c) for c in COHORTS}
    summ, prof, lg, curves = [], [], [], []
    for c, w in r["watchlists"].items():
        keys = [sel_key(selection[c]), ea_key(test, c)]
        _, draws, _ = cluster_bootstrap(test, c, n_boot, keys)
        summ.append(scouting_summary(w, {"model watchlist": draws[:, 0, 3],
                                         "EA potential-gap watchlist": draws[:, 1, 3]}))
        prof.append(scouting_profile(w)), lg.append(league_mix(w)), curves.append(precision_curve(w))
    r["scouting"], r["profile"] = pd.concat(summ, ignore_index=True), pd.concat(prof, ignore_index=True)
    r["leagues"], r["curves"] = pd.concat(lg, ignore_index=True), pd.concat(curves, ignore_index=True)
    r["importance_groups"], r["importance_columns"] = permutation_importance(table, selection, n_repeats)
    r["coefficients"], r["pca_loadings"] = logistic_coefficients(table, selection, coef_boot)
    r["sensitivity"], r["sensitivity_years"], r["label_stats"] = sensitivity(table, selection, n_boot)
    r["next_starts"], r["next_detail"] = next_season_check(r["watchlists"], n_boot)
    for name in ["year_metrics:metrics_by_year", "bootstrap:bootstrap_metrics", "differences:bootstrap_differences",
                 "feature_set_contrasts", "scouting:scouting_summary_2014", "profile:scouting_profile_2014",
                 "leagues:scouting_leagues_2014", "curves:precision_by_budget_2014",
                 "importance_groups:permutation_importance_groups", "importance_columns:permutation_importance_columns",
                 "coefficients:logistic_coefficients", "pca_loadings:logistic_pca_loadings",
                 "sensitivity:sensitivity_summary", "sensitivity_years:sensitivity_by_year",
                 "label_stats:sensitivity_label_stats", "next_starts:next_season_starts_2014"]:
        k, _, f = name.partition(":")
        save(r[k], TABLES, f or k, fmt=None)
    save(pd.concat(r["watchlists"].values()), TABLES, "watchlists_2014", fmt=None)
    for fig, name in [(plot_year_precision(r["year_metrics"], selection), "precision_by_year"),
                      (plot_headline(r["scouting"]), "headline_scouting_2014"),
                      (plot_overlap(r["scouting"]), "watchlist_overlap_2014"),
                      (plot_budget_curve(r["curves"]), "precision_by_budget_2014"),
                      (plot_feature_sets(r["bootstrap"], selection), "feature_set_comparison"),
                      (plot_importance(r["importance_groups"]), "permutation_importance_groups"),
                      (plot_coefficients(r["coefficients"]), "logistic_coefficients"),
                      (plot_sensitivity(r["sensitivity"]), "sensitivity")]:
        save(fig, FIGURES, name)
        plt.close(fig)
    return r


if __name__ == "__main__":
    main()
