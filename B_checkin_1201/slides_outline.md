# Slide outline (draft for check-in 2)

Every member presents. Suggested split: slides 1–3, 4–5, 6–8, 9–11, one person each.

1. **The scouting problem.** A club can watch 1 player in 10. Whom? Breakout = EA rating rises by at
   least median + 3 within a year. Action: a ranked watchlist every 1 August.
2. **The data.** European Soccer Database: 11 leagues, 2008/09–2015/16, 25,979 matches, 183,978 dated
   ratings. One row = one player in one year (25,502 rows). Mark Davies example.
3. **Cleaning.** What we checked and fixed (empty records, spikes, formations → positions, missing
   line-ups, truncated seasons); 11.48% unlabelled starters as selection bias.
   Figure: `A_checkin_1124/figures/02_breakout_rate_by_group.png`.
4. **Exploration.** Age and current rating dominate; EA's potential gap is mostly age.
   Figure: `A_checkin_1124/figures/03_potential_gap_vs_age.png`.
5. **Features.** Four feature sets (EA gap / ratings without potential / + potential / + match history);
   new features; preprocessing fitted inside each fold.
6. **Validation design.** Rolling years 2011–2013, test once on 2014, player bootstrap.
   Figure: `B_checkin_1201/figures/precision_by_year.png`.
7. **Model comparison.** Logistic regression, random forest, gradient boosting vs EA gap and random.
   Figure: `B_checkin_1201/figures/validation_precision_by_year.png`.
8. **Headline: the 2014 watchlist.** 169 vs 134 vs 49 breakouts.
   Figure: `B_checkin_1201/figures/headline_scouting_2014.png`.
9. **Does it just copy EA?** No-potential model still beats EA; match history adds the new information.
   Figure: `B_checkin_1201/figures/feature_set_comparison.png`.
10. **What drives the ranking.** Regular starters at successful clubs rated below their teammates.
    Figures: `permutation_importance_groups.png`, `logistic_coefficients.png`.
11. **Robustness and limits; what a scout should do.** Sensitivity checks hold; EA subjectivity,
    selection bias, one test year. Figure: `B_checkin_1201/figures/sensitivity.png`.
