"""
AutoMakeosuFile - 自动生成osu!mania谱面的Python包
"""

from .config import Config
from .audio_processing import AudioProcessor
from .feature_extraction import FeatureExtractor
from .beatmap_generator import BeatmapGenerator

__version__ = "2.0.0"
__author__ = "AutoMakeosuFile"
__all__ = ["Config", "AudioProcessor", "FeatureExtractor", "BeatmapGenerator"]
