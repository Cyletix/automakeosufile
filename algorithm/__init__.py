"""
AutoMakeosuFile - 自动生成osu!mania谱面的Python包
"""

from .config import Config
from .audio_processing import AudioProcessor
from .feature_extraction import FeatureExtractor
from .beatmap_generator import BeatmapGenerator
from .auto_optimization import AutoOptimizer
from .osu_parser import OsuFileParser, parse_osu_file
from .evolutionary_optimizer import EvolutionaryOptimizer
from .utils import (
    add_timestamp_to_filename,
    save_to_temp_with_timestamp,
    save_to_picture_with_timestamp,
    copy_to_osu_songs_dir,
    cleanup_temp_files,
)

__version__ = "2.0.0"
__author__ = "AutoMakeosuFile"
__all__ = [
    "Config",
    "AudioProcessor",
    "FeatureExtractor",
    "BeatmapGenerator",
    "AutoOptimizer",
    "OsuFileParser",
    "parse_osu_file",
    "EvolutionaryOptimizer",
    "add_timestamp_to_filename",
    "save_to_temp_with_timestamp",
    "save_to_picture_with_timestamp",
    "copy_to_osu_songs_dir",
    "cleanup_temp_files",
]
