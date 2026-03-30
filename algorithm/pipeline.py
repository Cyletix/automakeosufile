# -*- coding: utf-8 -*-
"""
Shared in-process beatmap generation pipeline used by CLI and optimizer.
"""

from .audio_processing import AudioProcessor
from .beatmap_generator import BeatmapGenerator
from .config import Config
from .feature_extraction import FeatureExtractor


def run_pipeline(
    audio_file,
    config=None,
    output_dir=None,
    iteration=None,
    copy_audio=None,
    output_filename=None,
    version_label=None,
):
    config = config or Config()

    if output_dir is not None:
        config.OUTPUT_DIR = output_dir

    audio_processor = AudioProcessor(config)
    audio_data = audio_processor.process_audio(audio_file)

    feature_extractor = FeatureExtractor(config)
    features = feature_extractor.extract_features(audio_data, audio_data["note_events"])

    beatmap_generator = BeatmapGenerator(config)
    output_path = beatmap_generator.generate_beatmap(
        audio_file,
        features,
        output_dir=config.OUTPUT_DIR,
        iteration=iteration,
        output_filename=output_filename,
        copy_audio=copy_audio,
        version_label=version_label,
    )

    return {
        "audio_data": audio_data,
        "features": features,
        "output_path": output_path,
    }
