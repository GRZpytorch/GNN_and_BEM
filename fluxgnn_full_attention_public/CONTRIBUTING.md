# Contributing

Contributions that improve numerical robustness, graph construction, model
evaluation, reproducibility, or documentation are welcome.

## Development guidelines

1. Keep each change focused on one technical objective.
2. Preserve the documented physical sign conventions and boundary definitions.
3. Add validation when introducing new array shapes, graph formats, or problem cases.
4. Run `python -m compileall .` before opening a pull request.
5. Do not commit generated datasets, checkpoints, or evaluation figures.

Changes to the BEM formulation or neural architecture should be documented in
the pull request because they may invalidate existing checkpoints or numerical
comparisons.
