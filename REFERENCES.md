# References

Claims and formulas in this repository cite a key from `references/refs.bib` or carry the label **own derivation** / **design choice**. References were checked against the linked publisher, conference, author, or repository pages where available.

## Verified ball-drive references

The RBFA Study 30 document is Louis Stalpaert's KU Leuven thesis, *CounterRisk: Predicting counter-attacks, even if they don't occur*, academic year 2021–2022 [@stalpaert2022]. Drive metrics come from Table 4.4 and ECE from Table 4.7; their locations and derived gains are retained in `data/benchmarks/ball_drive/literature_comparison.csv`.

The pressing-intensity reference is **Joris Bekkers**, *Pressing Intensity: An Intuitive Measure for Pressing in Soccer*, arXiv:2501.04712, revised June 2025 [@bekkers2025]. The dribbling-covariate reference is Schepers et al., arXiv:2506.22503 [@schepers2025dribble]. Equation attribution distinguishes these sources from PRESS's own competing-risk formulation.

## Notes

SkillCorner's article describes pressure attributes and an overall pressure metric [@skillcorner-pressure]. It does not publish a hazard-model specification; this repository makes no claim that SkillCorner uses a hazard model.

Spearman's paper calls its control model the **Potential Pitch Control Field (PPCF)** [@spearman2018beyond]. In this code, the lowercase feature name `rppcf` is a project variable for the transition-weighted pass-success aggregate, `sum_k p_k * pass_success_k`; it is not presented as a standard name from Spearman's paper. **own derivation**
