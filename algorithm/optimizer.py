"""
Unified reference-driven optimizer for the beatmap generation pipeline.
"""

import argparse
import json
import os
import shutil
import time
from copy import deepcopy

from .config import Config, DEFAULT_TEST_AUDIO_FILE, DEFAULT_TEST_REFERENCE_OSU_FILE
from .osu_parser import parse_osu_file
from .pipeline import run_pipeline
from .utils import resolve_output_dir


DEFAULT_PARAMETER_GRID = {
    "DENSITY_FILTER_RATIO": [0.20, 0.30, 0.40, 0.50, 0.60, 0.70],
    "DENSITY_NPS_SCALE": [0.80, 1.00, 1.20, 1.40, 1.60],
    "MAX_NOTES_PER_BEAT": [6, 8, 10, 12, 14, 16],
    "MIN_COLUMN_GAP_MS": [0, 1, 5, 10],
    "MAX_SAME_COLUMN_INTERVAL_MS": [5, 10, 20, 30, 40, 50],
    "HOLD_NOTE_MIN_DURATION": [180, 240, 300, 360, 420],
    "HOLD_NOTE_TARGET_PERCENTAGE": [8.0, 12.0, 15.0, 18.0, 22.0],
    "COLUMN_REBALANCE_THRESHOLD": [0.08, 0.12, 0.16, 0.20, 0.24],
    "PHYSICAL_CORRECTION_STRICTNESS": [0.05, 0.10, 0.20, 0.40, 0.70],
}


class BeatmapEvaluator:
    def __init__(self, reference_osu_path):
        self.reference_osu_path = reference_osu_path
        self.reference_stats = parse_osu_file(reference_osu_path)

    def compare(self, candidate_stats):
        target = self.reference_stats
        metric_scores = {
            "total_notes": 1.0
            - min(
                abs(candidate_stats["total_notes"] - target["total_notes"])
                / max(target["total_notes"], 1),
                1.0,
            ),
            "nps": 1.0
            - min(abs(candidate_stats["nps"] - target["nps"]) / max(target["nps"], 1.0), 1.0),
            "hold_notes_percentage": 1.0
            - min(
                abs(
                    candidate_stats["hold_notes_percentage"]
                    - target["hold_notes_percentage"]
                )
                / 100.0,
                1.0,
            ),
            "column_balance_std": 1.0
            - min(
                abs(candidate_stats["column_balance_std"] - target["column_balance_std"])
                / max(target["column_balance_std"], 1.0),
                1.0,
            ),
            "mean_hold_duration": 1.0
            - min(
                abs(candidate_stats["mean_hold_duration"] - target["mean_hold_duration"])
                / max(target["mean_hold_duration"], 1.0),
                1.0,
            ),
        }

        weights = {
            "total_notes": 0.30,
            "nps": 0.25,
            "hold_notes_percentage": 0.20,
            "column_balance_std": 0.15,
            "mean_hold_duration": 0.10,
        }
        similarity = sum(metric_scores[name] * weights[name] for name in metric_scores)

        deltas = {
            "total_notes": candidate_stats["total_notes"] - target["total_notes"],
            "nps": round(candidate_stats["nps"] - target["nps"], 4),
            "hold_notes_percentage": round(
                candidate_stats["hold_notes_percentage"] - target["hold_notes_percentage"],
                4,
            ),
            "column_balance_std": round(
                candidate_stats["column_balance_std"] - target["column_balance_std"], 4
            ),
            "mean_hold_duration": round(
                candidate_stats["mean_hold_duration"] - target["mean_hold_duration"], 4
            ),
        }

        return {
            "similarity": round(similarity, 6),
            "metric_scores": metric_scores,
            "deltas": deltas,
            "reference": target,
        }


class BeatmapOptimizer:
    def __init__(
        self,
        audio_path=DEFAULT_TEST_AUDIO_FILE,
        reference_osu_path=DEFAULT_TEST_REFERENCE_OSU_FILE,
        columns=7,
        workspace_dir=None,
        base_config=None,
        parameter_grid=None,
    ):
        self.audio_path = audio_path
        self.columns = columns
        self.workspace_dir = resolve_output_dir(
            audio_path,
            output_dir=workspace_dir,
            export_subdir=os.path.join("automakeosu_generated", "optimization"),
        )
        self.current_osu_path = os.path.join(self.workspace_dir, "current_candidate.osu")
        self.best_osu_path = os.path.join(self.workspace_dir, "best_candidate.osu")
        self.report_path = os.path.join(self.workspace_dir, "optimization_report.json")
        self.reference_cache_path = os.path.join(self.workspace_dir, "reference_stats.json")
        self.parameter_grid = parameter_grid or DEFAULT_PARAMETER_GRID
        self.evaluator = BeatmapEvaluator(reference_osu_path)
        self.base_config = base_config or Config(DEFAULT_COLUMNS=columns, DURATION=None)
        self.base_config.DEFAULT_COLUMNS = columns
        self.base_config.OUTPUT_DIR = self.workspace_dir
        self.base_config.COPY_AUDIO_TO_OUTPUT_DIR = False
        self.history = []
        self.best_result = None

        os.makedirs(self.workspace_dir, exist_ok=True)
        self._write_reference_cache()

    def _write_reference_cache(self):
        try:
            with open(self.reference_cache_path, "w", encoding="utf-8") as handle:
                json.dump(self.evaluator.reference_stats, handle, indent=2)
        except OSError:
            pass

    def _closest_index(self, values, current_value):
        distances = [abs(value - current_value) for value in values]
        return distances.index(min(distances))

    def _neighbor_candidates(self, config):
        candidates = []
        for name, values in self.parameter_grid.items():
            current_value = getattr(config, name)
            index = self._closest_index(values, current_value)
            for neighbor_index in {index - 1, index + 1}:
                if 0 <= neighbor_index < len(values):
                    candidate_value = values[neighbor_index]
                    candidate_config = config.clone()
                    setattr(candidate_config, name, candidate_value)
                    candidates.append((name, candidate_value, candidate_config))
        return candidates

    def evaluate(self, config, label):
        started_at = time.time()
        result = run_pipeline(
            self.audio_path,
            config=config,
            output_dir=self.workspace_dir,
            copy_audio=False,
            output_filename=os.path.basename(self.current_osu_path),
            version_label=f"Optimization {label}",
        )
        stats = parse_osu_file(result["output_path"])
        comparison = self.evaluator.compare(stats)
        elapsed = round(time.time() - started_at, 2)

        evaluation = {
            "label": label,
            "similarity": comparison["similarity"],
            "config": config.to_dict(),
            "stats": stats,
            "comparison": comparison,
            "elapsed_seconds": elapsed,
            "output_path": result["output_path"],
        }
        self.history.append(evaluation)

        if self.best_result is None or evaluation["similarity"] > self.best_result["similarity"]:
            self.best_result = deepcopy(evaluation)
            shutil.copy2(result["output_path"], self.best_osu_path)

        self._write_report()
        print(
            f"[{label}] similarity={evaluation['similarity']:.4f} "
            f"notes={stats['total_notes']} nps={stats['nps']:.2f} "
            f"hold={stats['hold_notes_percentage']:.2f}% "
            f"col_std={stats['column_balance_std']:.2f}"
        )
        return evaluation

    def _write_report(self):
        payload = {
            "audio_path": self.audio_path,
            "reference_osu_path": self.evaluator.reference_osu_path,
            "workspace_dir": self.workspace_dir,
            "best_result": self.best_result,
            "history": self.history,
        }
        with open(self.report_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)

    def optimize(self, max_rounds=4, target_similarity=0.9):
        current_config = self.base_config.clone()
        current_result = self.evaluate(current_config, "baseline")

        for round_index in range(1, max_rounds + 1):
            if current_result["similarity"] >= target_similarity:
                break

            best_round_result = current_result
            best_round_config = current_config

            print(f"\n开始第 {round_index} 轮参数搜索...")
            for name, candidate_value, candidate_config in self._neighbor_candidates(current_config):
                label = f"round{round_index}_{name}_{candidate_value}"
                candidate_result = self.evaluate(candidate_config, label)

                if candidate_result["similarity"] > best_round_result["similarity"]:
                    best_round_result = candidate_result
                    best_round_config = candidate_config

            if best_round_result["similarity"] <= current_result["similarity"]:
                print("未找到更优参数，停止优化。")
                break

            current_config = best_round_config
            current_result = best_round_result
            print(
                f"第 {round_index} 轮完成，最佳相似度提升到 {current_result['similarity']:.4f}"
            )

        return self.best_result


def main():
    parser = argparse.ArgumentParser(description="参考谱面驱动的参数优化器")
    parser.add_argument(
        "--audio-file",
        default=DEFAULT_TEST_AUDIO_FILE,
        help="用于生成谱面的目标音频",
    )
    parser.add_argument(
        "--reference-osu",
        default=DEFAULT_TEST_REFERENCE_OSU_FILE,
        help="参考谱面文件",
    )
    parser.add_argument("--columns", type=int, default=7, help="键数")
    parser.add_argument("--rounds", type=int, default=4, help="最大搜索轮数")
    parser.add_argument(
        "--target-similarity",
        type=float,
        default=0.9,
        help="目标统计相似度",
    )
    parser.add_argument(
        "--process-seconds",
        type=float,
        default=None,
        help="仅用于快速试验；留空表示全曲优化",
    )
    parser.add_argument(
        "--workspace-dir",
        default=None,
        help="优化工作区目录",
    )

    args = parser.parse_args()

    base_config = Config(
        DEFAULT_COLUMNS=args.columns,
        DURATION=args.process_seconds,
        OUTPUT_DIR=args.workspace_dir,
        COPY_AUDIO_TO_OUTPUT_DIR=False,
    )

    optimizer = BeatmapOptimizer(
        audio_path=args.audio_file,
        reference_osu_path=args.reference_osu,
        columns=args.columns,
        workspace_dir=args.workspace_dir,
        base_config=base_config,
    )

    best_result = optimizer.optimize(
        max_rounds=args.rounds,
        target_similarity=args.target_similarity,
    )

    print("\n优化完成")
    print(f"最佳相似度: {best_result['similarity']:.4f}")
    print(f"最佳谱面: {optimizer.best_osu_path}")
    print(f"报告: {optimizer.report_path}")


if __name__ == "__main__":
    main()
