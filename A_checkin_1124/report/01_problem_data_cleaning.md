# 1. Problem and dataset

## Problem and why it matters

Clubs cannot scout every player. A scouting department that can watch only one player in ten needs to
know whom to watch. We ask whether data available on 1 August of a year can pick out the players whose
FIFA overall rating will rise unusually fast over the next year, the "breakouts". A player breaks out if
the player's latest overall rating in the 12 months after 1 August is at least 3 points above the latest rating
in the 12 months before, after subtracting that year's median change across all players. That median is 0 in every year
2009–2014, so in practice a breakout is a rise of 3 or more points. The action is a watchlist of the top
10% (rounded down) of each year's cohort. We judge it by Precision@10%, the share of the watchlist who
break out. We study all players and, separately, young players (completed age ≤ 24 on 1 August), the
group scouts care about most.

## One data point

One row is one player in one prediction year, for a player who started at least one league match in
the season just finished. Example: on 1 August 2011 Mark Davies had started 9 matches. The latest overall
rating was 72 and EA's potential was 78. The latest rating in the following 12 months was 75. The
change of +3 is at least the median (0) plus 3, so the row is a breakout. Everything except the later
rating is known on 1 August.

## How many data points

The source holds 25,979 matches in 11 top European leagues (2008/09–2015/16), 11,060 players and
183,978 dated rating records. From 29,149 starter-years (2009–2015) we obtain 25,502 rows for 8,903
players that have both a current and a follow-up rating. That is 3,317–3,871 rows per year, with a
breakout rate between 11.3% and 25.7% by year. The young subset has 9,687 rows (4,724 players). 2009–2010
are used only for training, 2011–2013 for rolling validation (18,002 rows in 2009–2013), and 2014
(3,629 rows) is the final test, used once. 2015 (3,871 rows) is dropped because the ratings end on
7 July 2016, before its follow-up window closes.

## Seemingly useful features

- From the current rating snapshot: overall rating, EA's potential, and the gap between them. The EA
  baseline ranks players by this gap. Also 33 skill ratings and the date of the snapshot.
- From the player: age.
- From last season's lineups: number of starts, share of the club's matches started, the most common
  pitch position, club points per game, and the rating compared with teammates.
- Context: league.

Because the label is also an EA rating, we test models without EA's potential and compare rating-only
with rating-plus-match-history feature sets.

## Data source

Mathien, H. (2016). *European Soccer Database* (version 10, 23 Oct 2016). Kaggle.
https://www.kaggle.com/datasets/hugomathien/soccer. SQLite file, 313 MB. Licence: "Database: Open
Database, Contents: © Original Authors" (Open Database Licence for the database). The author asks for
no commercial use. The data come from football-data.mx-api.enetscores.com (matches, lineups and
formations) and sofifa.com (EA Sports FIFA player ratings; FIFA data are EA Sports' property). We use
it for non-commercial coursework only.

# 2. Data cleaning and preparation

All steps are in `src/cleaning.py`. Every check and count is written to `data/cleaning_audit.csv`.

**Missing values.** 836 rating records (0.5%) are completely empty. Each sits on the same player and
date as a valid record, so we drop them. Seven skills (volleys, curve, agility, balance, jumping,
vision, sliding tackle) did not exist in older FIFA editions. They are missing in 1,877 records dated
before March 2010, which is 199 rows of our 2009 cohort. We leave them missing, and the median imputer
fills them inside each training fold. Lineups are the main gap: 29,257 of 571,538 lineup slots have no
player ID. All 1,516 fully missing team lineups are from 2008/09, in Belgium, the Netherlands, Poland,
Portugal and Switzerland (43–68% of their team-matches). Poland also has no pitch coordinates from
2008/09 to 2012/13, so 801 of the 808 rows without a position are Polish. Their high breakout rate
(32%) is a league effect, not a position effect. We keep these rows. Instead we add
`club_lineup_coverage` (share of the club's lineup slots filled, below 0.9 for 1,682 rows, mostly 2009)
and `position_missing`, because dropping them would remove most of 2009 in five leagues.

**Duplicates.** There are no duplicate match IDs and no duplicate (league, season, date, home, away)
fixtures. Player IDs are unique. 212 extra rows share a name, but never with the same birthday, so we
join on IDs, never names. Valid ratings have no duplicate player-date pairs. In 13 matches a home player
also appears in an away slot (usually the home goalkeeper copied into slot 11). We keep the home entry
and count each player once per match. The cleaned table is asserted to have one row per (year, player).

**Inconsistent entries.** Pitch Y coordinates were binned wrongly at first. Formations show that Y=5–9
are midfield lines: 5/6 holding, 7 central, 8 behind a lone striker, 9 the tip of a diamond. Only
Y=10–11 are strikers, and Y=0 (always with X=0) is a placeholder. Correcting the bins moved 464 rows into
Midfielder. EA work-rate fields contain stray codes ('None', 'norm', '_0', digits; about 4,300 and
6,800 records), which the feature code sets to missing. Belgium 2013/14 contains only 12 play-off
matches between 4 clubs, and Italy and Switzerland 2011/12 are 94% and 90% complete. We add
`league_match_coverage`. As a result the 2014 test year has 69 Belgian rows instead of about 290.

**Data types and units.** Dates are parsed as dates, IDs as integers, and ratings are checked to be
numeric and within 0–100. Height is stored in centimetres converted from whole inches (157–208 cm), and
weight is in pounds (117–243 lb, converted to kg for features). Age is measured in completed years on
1 August.

**Unusual values.** Goals range from 0 to 10, which is plausible. BMI ranges from 17.6 to 30.9, also
plausible, so no player outliers are removed. 21 rating records are isolated spikes: at least 8 points
from both neighbours, which agree with each other (for example 53, 53, 44, 54). We remove them as entry
errors. This drops 1 row and flips 1 label, and the 2013 EA baseline moves from 36.4% to 36.1%. We keep
1,071 records where potential is below overall (EA's own judgment) and 385 rows whose one-year change
is 10 points or more (−16 to +32), because these are EA re-ratings, not errors.

**Preparation and known biases.** For each 1 August we take the latest rating in the previous 12
months and the latest in the next 12 months. A rating on 1 August itself counts as follow-up. Of
28,810 starters with a current rating, 11.48% have no follow-up rating and cannot be labelled. They are
older (27.6 vs 26.1), lower rated (64.8 vs 70.3), started fewer matches (10.7 vs 16.9) and come mostly
from smaller leagues (19–23% excluded in Portugal, Switzerland, Scotland, Belgium and Poland, 3% in
England). Our results therefore describe players who stay in EA's database. The gap between the two
ratings varies from 7 to 696 days (median 365). The 243 validation-period rows with gaps under 180
days (all from 2013) break out only 5.3% of the time, against 18.4% otherwise. We keep the agreed label
and store `horizon_days` for a sensitivity check. It uses the outcome date, so it is never a feature.
The 2014 follow-up window is complete (ratings run to 31 July 2015).
