# Research pipeline

Start from the repository root with `python -m pipeline.run --help`.
The default prints a plan; `--execute` starts a fresh complete run.

See [pipeline.qmd](../pipeline.qmd) for the dependency order, input/output
contracts, match splits, per-task stages and refactoring checks.

The library modules contain the scientific implementations. Numbered scripts
are command-line entry points. Outputs belong in an explicit external run
folder; they are never installed automatically into the inference package.
