import os
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple


DEFAULT_TEST_SONG_DIR = r"D:\OneDrive\Project\CyletixMusicGame\songs\Scattered Rose"
DEFAULT_TEST_AUDIO_FILE = os.path.join(DEFAULT_TEST_SONG_DIR, "Scattered Rose.mp3")
DEFAULT_TEST_REFERENCE_OSU_FILE = os.path.join(
    DEFAULT_TEST_SONG_DIR, "Scattered Rose.osu"
)
DEFAULT_EXPORT_SUBDIR = "automakeosu_generated"


@dataclass
class Config:
    """Runtime configuration for beatmap generation and optimization."""

    SAMPLE_RATE: int = 22050
    DURATION: Optional[float] = None
    MONO: bool = True

    N_FFT: int = 2048
    HOP_LENGTH: int = 512

    N_MELS: int = 128
    FMIN: int = 20
    FMAX: int = 8000

    ADAPTIVE_THRESHOLD_BLOCK_SIZE: int = 15
    ADAPTIVE_THRESHOLD_C: float = -8.0
    MORPH_KERNEL_SIZE: int = 2

    MIN_NOTE_DURATION_MS: int = 5
    MAX_NOTE_DURATION_MS: int = 2000
    NOTE_GAP_MS: int = 10

    BEAT_DIVISORS: List[int] = field(default_factory=lambda: [1, 2, 3, 4, 6, 8, 12, 16])
    MAX_ALIGN_ERROR_MS: int = 100

    MIN_COLUMN_GAP_MS: int = 1
    MAX_SAME_COLUMN_INTERVAL_MS: int = 10
    PHYSICAL_CORRECTION_STRICTNESS: float = 0.05

    DEFAULT_COLUMNS: int = 7
    COLUMN_MAPPING: Dict[int, List[int]] = field(
        default_factory=lambda: {
            4: [0, 3, 7, 10],
            6: [0, 2, 4, 5, 7, 9],
            7: list(range(7)),
            8: list(range(8)),
        }
    )

    MAX_NOTES_PER_BEAT: int = 16
    ENERGY_SMOOTHING_WINDOW: int = 20
    ONSET_SMOOTHING_WINDOW: int = 5
    DENSITY_FILTER_RATIO: float = 0.5
    DENSITY_NPS_SCALE: float = 1.4
    DENSITY_MAPPING: List[Tuple[float, float]] = field(
        default_factory=lambda: [
            (0.0, 20.0),
            (0.2, 22.0),
            (0.4, 24.0),
            (0.6, 26.0),
            (0.8, 28.0),
        ]
    )
    SNAP_RESTRICTIONS: List[Tuple[float, List[int]]] = field(
        default_factory=lambda: [
            (0.0, [1, 2, 3, 4, 6, 8, 12, 16]),
            (0.2, [1, 2, 3, 4, 6, 8, 12, 16]),
            (0.4, [1, 2, 3, 4, 6, 8, 12, 16]),
            (0.6, [1, 2, 3, 4, 6, 8, 12, 16]),
            (0.8, [1, 2, 3, 4, 6, 8, 12, 16]),
        ]
    )

    GRID_WEIGHT_DIVISORS: List[int] = field(default_factory=lambda: [1, 2, 4, 8])
    GRID_WEIGHT_ENERGY_WEIGHT: float = 0.65
    GRID_WEIGHT_RECURRENCE_WEIGHT: float = 0.35
    GRID_WEIGHT_PRIORS: Dict[int, float] = field(
        default_factory=lambda: {
            1: 1.15,
            2: 1.0,
            3: 0.92,
            4: 0.74,
            6: 0.60,
            8: 0.38,
            12: 0.28,
            16: 0.18,
        }
    )

    TRANSIENT_LAYER_DIVISORS: List[int] = field(default_factory=lambda: [2, 4, 8])
    TRANSIENT_LAYER_PRIORS: Dict[int, float] = field(
        default_factory=lambda: {
            2: 1.00,
            4: 0.70,
            8: 0.24,
        }
    )
    TRANSIENT_DIFF_WEIGHTS: Dict[int, float] = field(
        default_factory=lambda: {
            2: 0.95,
            4: 0.70,
            8: 0.28,
        }
    )
    TRANSIENT_GRADIENT_WEIGHT: float = 0.55
    TRANSIENT_PEAK_WEIGHT: float = 0.45
    TRANSIENT_POWER: float = 1.35
    ONSET_LOW_BAND_RATIO: Tuple[float, float] = (0.0, 0.18)
    ONSET_MID_BAND_RATIO: Tuple[float, float] = (0.18, 0.58)
    ONSET_HIGH_BAND_RATIO: Tuple[float, float] = (0.58, 1.0)
    ONSET_COMBINED_WEIGHTS: Tuple[float, float, float] = (0.40, 0.35, 0.25)

    BEAT_FAMILY_BINARY_DIVISORS: List[int] = field(default_factory=lambda: [1, 2, 4, 8, 16])
    BEAT_FAMILY_TRIPLET_DIVISORS: List[int] = field(default_factory=lambda: [3, 6, 12])
    BEAT_FAMILY_WINDOW_BEATS: int = 8
    BEAT_FAMILY_SWITCH_PENALTY: float = 0.16
    BEAT_FAMILY_STAY_BONUS: float = 0.02
    BEAT_FAMILY_GLOBAL_DOMINANCE_THRESHOLD: float = 1.18
    BEAT_FAMILY_GLOBAL_SUPPRESS_MULTIPLIER: float = 0.10
    BEAT_FAMILY_WINDOW_SUPPRESS_MULTIPLIER: float = 0.18
    BEAT_FAMILY_WINDOW_MATCH_BOOST: float = 1.10
    BEAT_FAMILY_NEUTRAL_MARGIN: float = 0.08
    BEAT_FAMILY_BINARY_PRIOR: float = 1.00
    BEAT_FAMILY_TRIPLET_PRIOR: float = 0.68

    SECTION_STATE_DIVISORS: List[int] = field(default_factory=lambda: [1, 2, 4, 8])
    SECTION_STATE_WINDOW_BEATS: int = 8
    SECTION_STATE_ALLOW_COARSER_WEIGHT: float = 0.72
    SECTION_STATE_FINE_DIVISOR_PENALTY: float = 0.28
    SECTION_STATE_SWITCH_PENALTY: float = 0.22
    SECTION_STATE_SWITCH_DISTANCE_SCALE: float = 0.18
    SECTION_STATE_STAY_BONUS: float = 0.03
    SECTION_STATE_SUPPRESS_FINESCALE_MULTIPLIER: float = 0.08
    SECTION_STATE_LOCAL_ENERGY_WEIGHT: float = 0.62
    SECTION_STATE_LOCAL_REPETITION_WEIGHT: float = 0.38
    SECTION_STATE_ANCHOR_RATIO_THRESHOLD: float = 0.58
    SECTION_STATE_ANCHOR_BOOST: float = 1.28
    SECTION_STATE_COARSE_SKELETON_BOOST: float = 1.18
    SECTION_STATE_COARSE_PROTECTION_MARGIN: float = 0.12
    SECTION_STATE_FINE_COMPETITION_PENALTY: float = 0.72
    SECTION_STATE_PRIORS: Dict[int, float] = field(
        default_factory=lambda: {
            1: 1.08,
            2: 1.00,
            4: 0.76,
            8: 0.18,
        }
    )

    BAR_PATTERN_BEATS: int = 4
    BAR_PATTERN_SLOTS_PER_BEAT: int = 2
    BAR_PATTERN_ONSET_WEIGHT: float = 0.62
    BAR_PATTERN_OCCUPANCY_WEIGHT: float = 0.38
    BAR_PATTERN_SLOT_THRESHOLD: float = 0.56
    BAR_PATTERN_NEIGHBOR_KEEP_THRESHOLD: float = 0.42
    BAR_PATTERN_INHERIT_BONUS: float = 0.12
    BAR_PATTERN_MIN_ACTIVE_SLOTS: int = 2
    BAR_PATTERN_SNAP_TOLERANCE_RATIO: float = 0.55
    BAR_PATTERN_ENABLE_TRIPLET_REMAP: bool = True
    BAR_PATTERN_ENABLE_FINE_BINARY_REMAP: bool = True
    BAR_PATTERN_SECTION_MIN_BARS: int = 4
    BAR_PATTERN_SECTION_NOVELTY_RADIUS: int = 2
    BAR_PATTERN_SECTION_THRESHOLD_STD: float = 0.50
    BAR_PATTERN_SECTION_PATTERN_NOVELTY_WEIGHT: float = 0.75
    BAR_PATTERN_SECTION_AUDIO_NOVELTY_WEIGHT: float = 0.25
    BAR_PATTERN_PROTOTYPE_MEDIUM_SECTION_BARS: int = 8
    BAR_PATTERN_PROTOTYPE_LARGE_SECTION_BARS: int = 16
    BAR_PATTERN_PROTOTYPE_MAX_COUNT: int = 3
    BAR_PATTERN_PROTOTYPE_KEEP_RATIO: float = 0.22
    BAR_PATTERN_PROTOTYPE_EXTRA_RATIO: float = 0.82
    BAR_PATTERN_PROTOTYPE_MAX_EXTRA_SLOTS: int = 1
    BAR_PATTERN_PROTOTYPE_FORCE_COPY: bool = True
    BAR_PATTERN_PROTOTYPE_TEMPLATE_PENALTY: float = 0.08
    BAR_PATTERN_PROTOTYPE_TEMPLATE_SLACK: int = 1
    BAR_PATTERN_PROTOTYPE_TEMPLATE_TIE_MARGIN_RATIO: float = 0.18

    HOLD_NOTE_MIN_DURATION: int = 180
    HOLD_NOTE_MAX_DURATION: int = 800
    HOLD_NOTE_TARGET_PERCENTAGE: float = 15.0

    COLUMN_BALANCE_TARGET_STD: float = 2.0
    COLUMN_REBALANCE_THRESHOLD: float = 0.08

    ENABLE_TIMING_GRID_FILTER: bool = True
    ENABLE_COLUMN_BALANCE_FILTER: bool = True
    ENABLE_SILENCE_ENERGY_FILTER: bool = True

    SILENCE_WINDOW_MS: int = 500
    SILENCE_LEADING_MARGIN_MS: int = 80
    SILENCE_ONSET_ABS_THRESHOLD: float = 0.10
    SILENCE_ONSET_REL_THRESHOLD: float = 1.30
    SILENCE_ABS_THRESHOLD: float = 0.055
    SILENCE_REL_THRESHOLD: float = 0.92

    TIMING_FILTER_HOLD_MIN_DIVISOR: int = 8
    COLUMN_BALANCE_WINDOW_MS: int = 2000
    COLUMN_BALANCE_MAX_SHARE: float = 0.45

    OUTPUT_DIR: Optional[str] = None
    EXPORT_SUBDIR: str = DEFAULT_EXPORT_SUBDIR
    COPY_AUDIO_TO_OUTPUT_DIR: bool = True

    def clone(self, **overrides: Any) -> "Config":
        data = self.to_dict()
        data.update(overrides)
        return Config(**data)

    def apply_overrides(self, overrides: Dict[str, Any]) -> None:
        for key, value in overrides.items():
            if not hasattr(self, key):
                raise AttributeError(f"Unknown config field: {key}")
            setattr(self, key, value)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
