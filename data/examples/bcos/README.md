# BCOS real-match descriptive example

Brisbane FC–Perth Glory, match1925299,21December2024. Published material consists
of aggregate spatial/time/support/numerical tables, metadata and a rendered
one-frame model map. Raw tracking and checkpoint-level feature tables remain
in the local input/research repositories.

The same1928canonical outfield/no-shot AS73 states supply all summaries. They
are not a uniform-time sample or all control frames. Production carry dispersion
is used, not a mixture of held-out fold models. Spatial cells average the scalar
area at carrier origin; temporal thirds classify carrier origin, not target
locations. Cell means are descriptive; bootstrap spread concerns aggregation
stability and does not quantify BCOS solver or predictive error.

Rebuild aggregate plots with `python scripts/build_bcos_vignette_figures.py`.
The notebook recomputes the real frame when local processed data are available;
otherwise it displays the frozen rendered figure and its verified summary.
