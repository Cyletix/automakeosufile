# Parameter Adjustment Workflow

1. Read the root `README.md`.
2. Use `audio/Scattered Rose.wav` and `audio/Scattered Rose.osu` as the default benchmark pair unless the user says otherwise.
3. Run the unified optimizer:

```bash
python -m algorithm.optimizer --audio-file "audio/Scattered Rose.wav" --reference-osu "audio/Scattered Rose.osu" --rounds 2
```

4. Judge changes by reference statistics, not by file creation alone.
5. Keep optimization artifacts inside `output/optimization_experiments/`.
6. Do not create extra optimizer scripts in `temp/`.
7. After meaningful tuning, update the benchmark section in `README.md` and `docs/parameter_optimization_summary.md`.
