# Slides plan

About 11 slides, 2–3 per person, everyone presents. Figures are in `figures/`.

1. **The problem.** Scouts can follow 1 player in 10. Who should they pick? A breakout = FIFA rating
   up by the median + 3 within a year.
2. **The data.** 11 leagues, 2008/09–2015/16, 183,978 ratings. One row = one player in one year
   (25,502 rows). Example: Mark Davies, 72 → 75.
3. **Cleaning.** What we fixed: empty records, typing errors in ratings, positions from formations,
   missing line-ups. Who we lose: 11.5% without a later rating.
4. **What goes with breakouts.** Age and current rating. EA's potential gap is mostly age.
   (`02_breakout_rate_by_group.png`, `03_potential_gap_vs_age.png`)
5. **Features.** Four feature sets: EA gap / ratings without potential / with potential / + match history.
6. **How we test.** Train on earlier years, validate on 2011–2013, test once on 2014.
   (`precision_by_year.png`)
7. **Model comparison.** Logistic regression, random forest, gradient boosting vs EA's gap and random.
   (`validation_precision_by_year.png`)
8. **The 2014 list.** 169 vs 134 vs 49 breakouts. (`headline_scouting_2014.png`)
9. **Does it just copy EA?** No: without potential it still wins; match history adds the new
   information. (`feature_set_comparison.png`)
10. **What the model looks for.** Regular starters at good clubs who are rated below their
    teammates. (`permutation_importance_groups.png`, `logistic_coefficients.png`)
11. **Robustness, limits and advice for scouts.** (`sensitivity.png`)
