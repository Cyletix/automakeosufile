# Repository Constraints

1. Before any task, read the root `README.md`.
2. Production code lives in `algorithm/`. Do not create a parallel core module.
3. `archive/` is reference-only. Do not revive old workflows there.
4. `temp/` may hold disposable scratch work only. Do not place core optimizer logic, parsers, or permanent tests in `temp/`.
5. Do not create new folders in the project root.
6. Default outputs must stay under `output/optimization_experiments/` unless the user explicitly asks for another existing path.
7. Optimization should overwrite stable workspace files such as:
   - `current_candidate.osu`
   - `best_candidate.osu`
   - `optimization_report.json`
8. Avoid popup windows. Save plots to `picture/`.
9. If copying files into the osu! songs directory fails, treat it as a warning, not a fatal error.
