# Decisions · Workstream 1 (data cleaning review)

Scope: review `process_data.py` and the cleaned table `checkin/cleaned_player_years.csv.gz`.
Every count below comes from `python process_data.py _local/database.sqlite` and is also in
`checkin/cleaning_audit.csv`, `checkin/lineup_coverage.csv`, `checkin/selection_bias.csv`, and
`checkin/coverage.csv`.

**Status: FINAL (2026-11-23).** `checkin/cleaned_player_years.csv.gz` is final for this review:
25,502 rows × 63 columns, 4,391 breakouts in 2009–2015 (was 4,393). The md5 of the uncompressed CSV is
`1697b523d35b08609da620b8a7323971`. Downstream workstreams can re-run on it.

**Coordinator rule (adopted):** the cleaned table carries cleaning and audit columns only (flags,
coverage, `horizon_days`). Raw feature or attribute columns (work rates, height/weight, foot, history) belong to
WS2 in `src/features.py`. For a short time an intermediate regeneration of `checkin/` carried
`rating_age_days`, `height_cm`, `weight_kg`, `preferred_foot` and string `attacking_/defensive_work_rate`. They
broke WS2/WS3 pipelines ("could not convert 'medium' to float"), and the final table no longer has them
(see D01-12).

**Net change to the cleaned table:** 25,503 → **25,502 rows** (−1), 59 → **63 columns** (4 appended, no
column renamed or removed, original order kept). **1 label flipped** (1 → 0). `position` changed
for 464 rows. Validation baselines change only in 2013 (see D01-5). The young cohort is unchanged
(9,687 rows, 4,724 players).

**Why exactly one row was dropped:** the removed row is `year=2009, player_api_id=41308` (Eric Hassli).
In the old table its current rating was 62 (dated 2008-08-30) and its next rating 71 (2009-08-30). That
change of +9 made it a breakout (label 1). The 62 sits between 73 (2007-08-30) and 71 (2009-08-30), so
it is one of the 21 isolated spikes removed in D01-5. It was his only rating in the 2009 current window
[2008-08-01, 2009-08-01), and without a current rating the row cannot be labelled. So one artefact
breakout leaves with the row (2009 breakouts 854 → 853). Among retained rows exactly one label changed:
`year=2013, player_api_id=316695` (Jude Winchester). Its current rating moves from the spike 44 to 53, and
the label goes from 1 to 0 (2013 breakouts 431 → 430). Total breakouts 4,393 → 4,391. No other row's label
changed, and 2014 (test) is untouched: 3,629 rows, 489 breakouts.

### D01-1: What do pitch Y values mean, and how should they map to positions?
- Evidence: in the full lineup table Y takes values 0, 1, 3, 5–11 (no 2 or 4). The most common sorted
  formations are (1,3,3,3,3,6,6,8,8,8,11) = 4-2-3-1, (1,3,3,3,3,7,7,7,7,10,10) = 4-4-2,
  (1,3,3,3,3,7,7,7,10,10,10) = 4-3-3, and (1,3,3,3,3,5,5,5,8,10,10) / (…5,7,7,9,11,11) = diamonds.
  So Y=5/6 is a holding midfielder, Y=8 is the line behind a lone striker, and Y=9 is the tip of a
  diamond or a second striker, always at central X (4–6). Players seen at Y=5 play
  midfield (Y 6–8) in 75% of their other starts and defence in 19%. Players seen at Y=9 play midfield
  (Y 5–8) in 67% of other starts and attack (Y 10–11) in 28%.
- Options considered: keep (0,1] GK, (1,5] DEF, (5,8] MID, (8,11] FWD; move Y=5 to midfield only;
  move Y=5 and Y=9 to midfield.
- Decision: GK = Y 1; DEF = Y 2–4 (only 3 occurs); MID = Y 5–9; FWD = Y 10–11. Y=0 → missing.
- Why: Y=5 is never a back-line value in any formation. Y=9 players mostly play midfield elsewhere.
- Impact: 233 rows move Defender → Midfielder and 231 move Forward → Midfielder. The 2009–2013
  rates by position stay close together (DEF 16.9%, FWD 17.8%, GK 17.6%, MID 18.2%). Labels unchanged.
- Impact if reversed: those 464 rows go back to their old groups. The effect on models is small.
- Needs team confirmation: yes (Y=9 as midfielder is a judgment call).

### D01-2: How to handle Y=0?
- Evidence: 22 slots, always with X=0, only in slots 1–2 of Netherlands and Portugal 2008/09 matches whose
  other slots have no coordinates at all. It is a placeholder.
- Decision: treat it as a missing position (the old bins already excluded it implicitly; now explicit).
- Impact if reversed: none in practice (22 slots, all from 2008/09).
- Needs team confirmation: no.

### D01-3: Lineup missingness: drop, impute or flag?
- Evidence: 29,257 of 571,538 slots have no player ID. All 1,516 fully missing team lineups (758 matches
  with both sides missing) are in 2008/09, in Belgium (68% of team-matches), Netherlands (67%),
  Switzerland (58%), Poland (57%) and Portugal (43%). Partial gaps persist in Poland until 2012/13
  (mean 2.1–5.7 missing IDs per lineup in 2009/10–2012/13). Poland has **no** pitch coordinates from
  2008/09 to 2012/13. That covers 801 of the 808 position-missing rows; the other 7 are Dutch, 2009.
- Options considered: drop affected club-seasons; rescale starts; keep and flag.
- Decision: keep every row. Add `club_lineup_coverage` (filled lineup slots of the primary club ÷
  11 × its matches) and `position_missing` (0/1). `starts` and `start_share` stay as observed (undercounted
  where coverage is low). Downstream code may use `start_share / club_lineup_coverage` as a corrected share.
- Why: dropping would remove most 2009 rows in five leagues and the whole of Poland's position data.
  The flag lets models and sensitivity checks see the problem. Both columns are known on 1 August
  (they describe the season just finished), so they are not leaky.
- Bias: `club_lineup_coverage` < 0.9 for 1,682 rows (971 in 2009; 726 Polish). In those rows starts are
  undercounted and some real starters are missing from the cohort. The "Unknown" position group's 32.4%
  breakout rate is a Polish league effect, not a position effect (Poland's rate is 32.5%).
- Impact if reversed: dropping rows with coverage < 0.9 removes 1,682 rows, mostly 2009 (training only).
- Needs team confirmation: yes (whether models should use the flags or the corrected share).

### D01-4: Truncated league-seasons
- Evidence: observed matches ÷ a full season's matches (double round robin; Scotland 38 and Switzerland 36
  games per team). Belgium 2013/14 has only **12 matches, 4 clubs, 29 Mar–3 May 2014** (play-offs only):
  coverage 0.05. Italy 2011/12 has 358/380 (0.94). Switzerland 2011/12 has 162/180 (0.90, Xamax expelled
  mid-season). Every other league-season is complete. Belgium 2009/10 has 210 matches for 15 clubs
  (Mouscron's results annulled), which counts as complete.
- Decision: keep, add `league_match_coverage` (per league-season, measured against the league's usual number of clubs
  when a season shows far fewer). 716 rows have coverage below 0.95.
- Why: this cannot be repaired from the database. It matters for the **2014 test year**, where Belgium
  contributes 69 rows instead of about 290 (2009–2015 range 238–298), and those 69 come from 4 clubs.
- Impact if reversed: none on labels. The test cohort's league mix differs from validation either way.
- Needs team confirmation: yes (report 2014 results with and without Belgium as a sensitivity check?).

### D01-5: Rating anomalies: isolated spikes
- Evidence: some rating histories contain one record that jumps and returns, e.g. Chris Humphrey 64,
  51, 64, 51, 64 (2014) and Jude Winchester 53, 53, 44, 54 (2013). The rule "at least 8 points from both
  neighbours, and the neighbours agree within 2" finds 21 such records, mostly from 2007.
- Options considered: keep all; flag rows using a spike; remove spike records before choosing snapshots.
- Decision: remove the 21 records from the ratings before choosing snapshots.
- Why: they are entry errors. A real rating change does not reverse within one update. The rule is strict,
  so genuine re-ratings (e.g. Nicky Kuiper 53 → 35 → 67, whose neighbours disagree) are kept.
- Impact (before → after): rows 25,503 → 25,502 (Eric Hassli 2009, a +9 "breakout" built on the spike,
  lost his only current-window rating, 73 → **62** → 71). Labels: 1 flip (Winchester 2013, current rating 44 → 53, breakout 1 → 0). Next rating
  changed for 2 rows (Bradley Jones 2009, Chris Humphrey 2013; neither label changes).
  rating_vs_teammates changed for 36 rows. Baselines: 2013 all players, EA 36.4% → **36.1%** (138 → 137
  hits), random 11.4% → **11.3%**; 2013 young, EA 50.7% → **50.0%**, random 21.0% → **20.9%**. 2011–2012
  unchanged. 2009 breakouts 854 → 853, 2013 431 → 430.
- Impact if reversed: restores the numbers above.
- Needs team confirmation: yes (it changes a quoted baseline).

### D01-6: Other rating checks
- Same-date duplicates among valid ratings: 0. Each of the 836 dropped rows is fully empty and has the
  same player and date as a valid record, so dropping them loses nothing. The only fully duplicated row
  (excluding id) is one of these empty rows.
- 7 skills (volleys, curve, agility, balance, jumping, vision, sliding_tackle) are missing in 1,877 valid
  records dated 2007-02 to 2010-02 (older FIFA editions lacked them). In the cleaned table that is 199
  rows, all in 2009 (training only). **Decision:** keep them missing, so the median imputer inside the
  pipeline fills them in. No flag: the gap is confined to one training year.
- potential < overall: 1,071 records, 130 cleaned rows (gap down to −16). **Decision:** keep. It is EA's own
  value (EA expects a decline), and `potential_gap` is simply negative.
- Extreme one-year changes: |change| ≥ 10 in 385 rows, ≥ 15 in 67 (range −16 to +32). Big jumps cluster
  in 2009 for young, low-rated players (e.g. Seydou Doumbia 51 → 77), when EA re-rated many players
  between editions. **Decision:** keep, no flag (they are EA's recorded values; `change` already lets
  anyone filter them).
- Work rates contain stray codes (attacking 4,286 records, defensive 6,823: 'None', 'norm', '_0', digits).
  They are not columns in the cleaned table. `src/features.py` maps them to missing.
- Needs team confirmation: no.

### D01-7: Variable horizon between the two ratings
- Evidence: `horizon_days` ranges 7–696 (median 365). In 2009–2013 the 243 rows under 180 days
  break out 5.3% of the time against 18.4% for the rest. Short horizons occur only in 2013 (243) and
  2014 (268), when EA started frequent updates: a player rated in July and last rated in September
  gets a 2-month "year". The median horizon by year is 365 (2009–2011), 450 (2012), 336 (2013),
  357 (2014).
- Decision: keep the label unchanged. Add `horizon_days` for sensitivity analysis only.
- Why: the label definition was agreed. A minimum-horizon filter would change rows mainly in the test year.
- **`horizon_days` is derived from `outcome_date`, so it is leaky; never use it as a feature** (see Requests).
- Needs team confirmation: yes (whether to report a "horizon ≥ 180 days" sensitivity result; workstream 4
  already plans one).

### D01-8: Selection bias from missing follow-up ratings
- Evidence: of 28,810 starters with a current rating, 3,308 (11.48%) have no follow-up rating.
  Compared with included starters, excluded ones are older (mean 27.6 vs 26.1), lower rated (64.8 vs 70.3),
  have smaller gaps (3.5 vs 4.8) and fewer starts (10.7 vs 16.9; start share 0.32 vs 0.47), and are less
  often young (30% vs 38%). The exclusion rate is 2.9% in England, 4–6% in Germany, Italy, Spain and France, 16% in
  the Netherlands and 19–23% in Poland, Belgium, Scotland, Switzerland and Portugal. By year it is
  9.4–13.0%, with no jump in 2014 (12.2%).
- Decision: keep the exclusion (no outcome exists), document it, write `checkin/selection_bias.csv`.
- Why: these players probably left EA's licensed leagues or retired, and few of them would be breakouts.
  So the measured prevalence and precision apply to players who stay in EA's coverage, and
  small-league rates are the most affected.
- Needs team confirmation: no (the report must state it as a limitation).

### D01-9: Match, team and player tables
- Matches: 25,979 rows, no duplicate IDs or (league, season, date, home, away) duplicates, no home = away,
  all dates inside 1 Jul–30 Jun of the named season (so the "July fixtures" case never occurs), goals
  0–10 with none missing (the 10–0 is a real 2010 Eredivisie result), all team IDs in Team, and no team in
  two leagues in one season. Repeat fixtures in one season occur only in Scotland and Switzerland, where
  they are part of the format.
- Lineup duplicates: 13 matches list a home player in an away slot (usually the home goalkeeper copied into
  away slot 11 with striker or missing coordinates), and 1 lists a player twice in one lineup.
  **Decision:** keep the home entry, count each player once per match (made explicit in code).
- Team table: 3 clubs have two `team_api_id`s sharing one FIFA ID (Mouscron, Polonia Bytom, Widzew Łódź). Not
  merged. Only `team_api_id` is used, within a season.
- Players: 11,060 unique `player_api_id`s. 212 extra rows share a name, but no name + birthday pair repeats
  (IDs, not names, are keys). No missing values. Birthdays 1967-01-23 to 1999-04-24, ages in the cleaned table
  between 14 and 50 (asserted). Height is stored in cm converted from whole inches (157.5–208.3 cm);
  weight is in **pounds** (117–243 lb = 53–110 kg). BMI 17.6–30.9, all plausible (e.g. Tim Wiese, 243 lb).
  9 players have rating records under two FIFA IDs; only `player_api_id` is used.
  **Decision:** no player rows changed. Height and weight are static (one value per player), so they
  slightly look ahead. Unit conversion is done in `src/features.py`.
- Needs team confirmation: no.

### D01-10: Is the 2014 follow-up window complete, and is 2015 incomplete?
- Evidence: the last rating is 2016-07-07. The 2014 window [2014-08-01, 2015-08-01) contains ratings up
  to 2015-07-31 with the usual monthly density (e.g. 681 in June and 1,317 in July 2015). Its exclusion rate
  (12.2%) matches 2011–2013. The 2015 window ends 2016-08-01 but data stop on 2016-07-07, with only 9
  ratings in July 2016.
- Decision: 2014 = `holdout` (complete), 2015 = `incomplete` (unchanged logic; never used).
- Needs team confirmation: no.

### D01-11: pandas deprecation and robustness
- `set_index(verify_integrity=True)` (deprecated in pandas 3 via `Pandas4Warning`) is replaced with an explicit
  `is_unique` check. The script now runs with `python -W error` (every warning an error) on pandas 3.0.5.
- The cleaned CSV is written to a temporary file and then renamed, so readers never see a partial file.
- Not tested on pandas 2.2 (it is not installed here). The code uses only APIs available in 2.2.
- Needs team confirmation: no.

### D01-12: Columns added and columns deliberately not added
- Added (appended after the original 59): `horizon_days` (leaky; sensitivity only), `position_missing`,
  `club_lineup_coverage`, `league_match_coverage` (all known on 1 August).
- Not added: `rating_age_days`, `height_cm`, `weight_kg`, work rates and footedness. `src/features.py`
  already engineers these, and its `build_feature_table` skips any engineered column that already
  exists in the cleaned table. A string `attacking_work_rate` from here would silently replace its numeric
  encoding.
- Needs team confirmation: no.

## Requests to coordinator

1. `src/common.py`: add `"horizon_days"` to `LEAKY_COLUMNS` (it is `outcome_date − rating_date`).
2. `src/common.py` (optional): list the three new data-quality columns somewhere (e.g. a `QUALITY_FLAGS`
   list) so modelling can choose to include or exclude them deliberately. They are not in any `BASE_*` list.
3. Numbers quoted elsewhere that changed: 2013 EA baseline 36.4% → 36.1% (all) and 50.7% → 50.0% (young).
4. `README.md` discrepancies (not edited by me):
   - "25,503 player-years" → **25,502**; "3,318–3,871 per year" → **3,317–3,871**.
   - "Remove 836 empty rating rows" → also "remove 21 isolated rating spikes (entry errors)".
   - Validation text says "reserve 2014–2015 for final testing … Confirm coverage through 1 August 2016
     before including the 2015 cohort". The plan is **2014 only**. 2015 is dropped because ratings end on
     2016-07-07.
   - Licence: README says "non-commercial use". Kaggle's licence field reads "Database: Open Database,
     Contents: © Original Authors" (ODbL for the database). The author asks for no commercial use, and FIFA data are EA
     Sports' property. Suggested citation: Mathien, H. (2016), *European Soccer Database*, Kaggle, v10
     (2016-10-23), https://www.kaggle.com/datasets/hugomathien/soccer. Sources: football-data.mx-api.enetscores.com
     (matches, lineups), football-data.co.uk (odds), sofifa.com (EA FIFA ratings).
   - "Reproduce" section: `checkin/` now also holds `cleaning_audit.csv`, `lineup_coverage.csv` and
     `selection_bias.csv`. The commands are POSIX-only (`source .venv/bin/activate`); the Windows equivalent is
     `.venv\Scripts\activate`.
   - Checked and correct: 25,979 matches, 11 leagues, 11,060 players, 183,978 rating records, 33 skills,
     313 MB, 29,149 candidates, 8,903 players, 11.48%, 9,687 / 4,724 young, breakout rate 11–26% by year.

## Open issues

- Not verified on pandas 2.2 (no such environment installed). Needs a run on a pandas 2.2 install.
- Whether models should use `club_lineup_coverage` / `league_match_coverage` / `position_missing` as
  features or only for sensitivity checks (modelling's call).
- The 2014 test cohort lacks most Belgian players (D01-4). The report should say so, and workstream 4 may
  add a "2014 without Belgium" sensitivity row.
- Real starters missing from 2009 lineups (five leagues) are absent from the cohort; this cannot be recovered.
