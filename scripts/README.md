# Manual project scripts

Run these modules from the repository root with `python -m` so imports and relative report paths resolve consistently. They are separate from the maintained `tests/` suite and the MLOps CI workflow.

- `audits/`: source and submission checks.
- `benchmarks/`: local performance and research measurements.
- `validation/`: manual smoke and UI checks that require local models or services.

For example, run `python -m scripts.validation.smoke_test --speech --evidence`. The automated CI checks remain in `tests/`.
