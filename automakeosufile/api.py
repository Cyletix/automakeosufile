from copy import deepcopy

from algorithm import Config, run_pipeline
from algorithm.optimizer import BeatmapOptimizer


def generate_beatmap(
    audio_path,
    *,
    columns=7,
    output_dir=None,
    process_seconds=None,
    output_filename=None,
    version_label=None,
    config=None,
    copy_audio=True,
):
    runtime_config = deepcopy(config) if config is not None else Config()
    runtime_config.DEFAULT_COLUMNS = columns
    runtime_config.DURATION = process_seconds
    runtime_config.OUTPUT_DIR = output_dir
    runtime_config.COPY_AUDIO_TO_OUTPUT_DIR = copy_audio

    return run_pipeline(
        audio_path,
        config=runtime_config,
        output_dir=output_dir,
        copy_audio=copy_audio,
        output_filename=output_filename,
        version_label=version_label,
    )


def optimize_beatmap(
    audio_path,
    reference_osu_path,
    *,
    columns=7,
    workspace_dir=None,
    rounds=4,
    target_similarity=0.9,
    process_seconds=None,
    config=None,
):
    runtime_config = deepcopy(config) if config is not None else Config()
    runtime_config.DEFAULT_COLUMNS = columns
    runtime_config.DURATION = process_seconds
    runtime_config.OUTPUT_DIR = workspace_dir
    runtime_config.COPY_AUDIO_TO_OUTPUT_DIR = False

    optimizer = BeatmapOptimizer(
        audio_path=audio_path,
        reference_osu_path=reference_osu_path,
        columns=columns,
        workspace_dir=workspace_dir,
        base_config=runtime_config,
    )
    return optimizer.optimize(max_rounds=rounds, target_similarity=target_similarity)
