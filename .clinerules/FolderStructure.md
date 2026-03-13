# Folder Structure

- `algorithm/`
  - current production implementation
- `archive/`
  - old versions and historical experiments
- `audio/`
  - input audio and reference `.osu` files
- `docs/`
  - detailed architecture and benchmark records
- `output/optimization_experiments/`
  - default output workspace
- `picture/`
  - saved visualizations
- `temp/`
  - temporary scratch files only

Current reference benchmark:

- audio: `audio/Scattered Rose.wav`
- target map: `audio/Scattered Rose.osu`

Any structural change must keep `algorithm/` as the single source of truth.
