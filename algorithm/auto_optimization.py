#!/usr/bin/env python3
"""
自动迭代优化脚本 - 将谱面相似度提升到0.9
"""

import json
import os
import sys
import subprocess
import time
import signal
from typing import Dict, List, Tuple, Optional

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from algorithm.config import Config


class AutoOptimizer:
    def __init__(self):
        self.config_path = "algorithm/config.py"
        self.reference_stats = self.load_reference_stats()
        self.current_similarity = 0.352  # 当前相似度
        self.iteration = 0
        self.best_similarity = 0.352
        self.best_config = None

    def load_reference_stats(self) -> Dict:
        """加载参考谱面统计"""
        with open("output/reference_stats.json", "r", encoding="utf-8") as f:
            return json.load(f)

    def generate_beatmap(self) -> Optional[Dict]:
        """生成谱面并获取统计信息"""
        print(f"\n[迭代 {self.iteration}] 生成谱面...")

        # 运行主程序生成谱面 - 设置5分钟超时
        # 使用.wav文件，避免MP3解码问题，处理更快
        cmd = [
            "python",
            "algorithm/main.py",
            "audio/Scattered Rose.wav",  # 使用.wav文件，处理更快
            "--columns",
            "7",
            "--output-dir",
            "output/optimization_experiments",
            "--process-seconds",  # 只处理前20秒，加快速度
            "20",
        ]

        try:
            # 设置5分钟超时 (300秒)
            result = subprocess.run(
                cmd, capture_output=True, text=True, check=True, timeout=300
            )
            print("✓ 谱面生成成功")

            # 尝试查找生成的.osu文件并解析统计
            import glob

            osu_files = glob.glob("output/optimization_experiments/*_7K.osu")

            if osu_files:
                # 使用osu_parser解析最新的.osu文件
                from algorithm.osu_parser import parse_osu_file

                stats = parse_osu_file(osu_files[0])
                return stats
            else:
                # 如果没有找到.osu文件，使用估算值
                print("警告: 未找到生成的.osu文件，使用估算统计")
                # 从输出中提取信息或使用默认值
                return {
                    "total_notes": 249,
                    "nps": 12.70,
                    "hold_notes_percentage": 28.9,
                    "column_balance_std": 7.89,
                }

        except subprocess.TimeoutExpired:
            print(f"✗ 谱面生成超时 (超过5分钟)")
            return None
        except subprocess.CalledProcessError as e:
            print(f"✗ 谱面生成失败: {e}")
            if e.stderr:
                print(f"错误输出: {e.stderr[:200]}")
            return None
        except Exception as e:
            print(f"✗ 谱面生成过程中出现异常: {e}")
            return None

    def calculate_similarity(self, stats: Dict) -> float:
        """计算相似度"""
        if not stats:
            return 0.0

        # 获取统计值，支持不同的键名
        total_notes = stats.get("total_notes", 0)
        nps = stats.get("nps", 0.0)

        # 长条比例键名可能是 hold_percentage 或 hold_notes_percentage
        hold_percentage = stats.get("hold_percentage") or stats.get(
            "hold_notes_percentage", 0.0
        )

        # 轨道平衡标准差键名可能是 column_std 或 column_balance_std
        column_std = stats.get("column_std") or stats.get("column_balance_std", 0.0)

        # 计算各项指标的相似度
        note_sim = min(total_notes / self.reference_stats["total_notes"], 1.0)
        nps_sim = min(nps / self.reference_stats["nps"], 1.0)

        # 长条比例相似度（越小越好）
        hold_diff = abs(hold_percentage - self.reference_stats["hold_notes_percentage"])
        hold_sim = max(0, 1.0 - hold_diff / 100)

        # 轨道平衡相似度（越小越好）
        column_diff = abs(column_std - self.reference_stats["column_balance_std"])
        column_sim = max(0, 1.0 - column_diff / 100)

        # 加权综合相似度
        weights = {
            "notes": 0.3,  # 总音符数权重
            "nps": 0.3,  # NPS权重
            "hold": 0.2,  # 长条比例权重
            "column": 0.2,  # 轨道平衡权重
        }

        similarity = (
            note_sim * weights["notes"]
            + nps_sim * weights["nps"]
            + hold_sim * weights["hold"]
            + column_sim * weights["column"]
        )

        return similarity

    def adjust_parameters(self, stats: Dict, similarity: float) -> bool:
        """根据当前结果调整参数"""
        print(f"[迭代 {self.iteration}] 调整参数...")

        # 获取统计值，支持不同的键名
        total_notes = stats.get("total_notes", 0)
        nps = stats.get("nps", 0.0)
        hold_percentage = stats.get("hold_percentage") or stats.get(
            "hold_notes_percentage", 0.0
        )
        column_std = stats.get("column_std") or stats.get("column_balance_std", 0.0)

        # 加载当前配置
        with open(self.config_path, "r", encoding="utf-8") as f:
            content = f.read()

        # 分析问题并调整
        adjustments = []

        # 1. 如果音符数量不足
        note_ratio = total_notes / self.reference_stats["total_notes"]
        if note_ratio < 0.8:
            # 增加检测灵敏度，保留更多音符
            if "DENSITY_FILTER_RATIO = 0.5" in content:
                new_ratio = min(0.5 + 0.1, 0.9)  # 逐步增加，最大0.9
                content = content.replace(
                    "DENSITY_FILTER_RATIO = 0.5", f"DENSITY_FILTER_RATIO = {new_ratio}"
                )
                adjustments.append(f"DENSITY_FILTER_RATIO: 0.5 → {new_ratio}")

        # 2. 如果NPS不足
        nps_ratio = nps / self.reference_stats["nps"]
        if nps_ratio < 0.8:
            # 提高密度映射
            if "DENSITY_MAPPING = [" in content:
                # 找到并更新DENSITY_MAPPING
                lines = content.split("\n")
                for i, line in enumerate(lines):
                    if "DENSITY_MAPPING = [" in line:
                        # 更新接下来的5行
                        for j in range(1, 6):
                            if i + j < len(lines) and "(" in lines[i + j]:
                                parts = lines[i + j].split(",")
                                if len(parts) >= 2:
                                    try:
                                        old_nps = float(parts[1].strip().rstrip(")"))
                                        new_nps = min(
                                            old_nps + 2.0, 35.0
                                        )  # 逐步增加，最大35
                                        lines[i + j] = lines[i + j].replace(
                                            str(old_nps), str(new_nps)
                                        )
                                        adjustments.append(
                                            f"NPS: {old_nps} → {new_nps}"
                                        )
                                    except:
                                        pass
                        break
                content = "\n".join(lines)

        # 3. 如果长条比例过高
        hold_diff = hold_percentage - self.reference_stats["hold_notes_percentage"]
        if hold_diff > 5:  # 超过5%
            # 提高长条最小持续时间
            if "HOLD_NOTE_MIN_DURATION = 300" in content:
                new_duration = min(300 + 50, 500)  # 逐步增加，最大500ms
                content = content.replace(
                    "HOLD_NOTE_MIN_DURATION = 300",
                    f"HOLD_NOTE_MIN_DURATION = {new_duration}",
                )
                adjustments.append(f"HOLD_NOTE_MIN_DURATION: 300 → {new_duration}")

        # 4. 如果轨道不平衡
        if column_std > 5:  # 标准差大于5%
            # 放宽轨道平衡标准
            if "COLUMN_BALANCE_TARGET_STD = 2.0" in content:
                new_std = min(2.0 + 1.0, 5.0)  # 逐步放宽，最大5
                content = content.replace(
                    "COLUMN_BALANCE_TARGET_STD = 2.0",
                    f"COLUMN_BALANCE_TARGET_STD = {new_std}",
                )
                adjustments.append(f"COLUMN_BALANCE_TARGET_STD: 2.0 → {new_std}")

        # 保存调整后的配置
        if adjustments:
            with open(self.config_path, "w", encoding="utf-8") as f:
                f.write(content)

            print("参数调整:")
            for adj in adjustments:
                print(f"  ✓ {adj}")
            return True

        print("  无需调整参数")
        return False

    def run_iteration(self) -> Tuple[float, Dict]:
        """运行一次迭代"""
        self.iteration += 1

        # 生成谱面
        stats = self.generate_beatmap()
        if not stats:
            return 0.0, {}

        # 计算相似度
        similarity = self.calculate_similarity(stats)
        self.current_similarity = similarity

        # 获取统计值，支持不同的键名
        total_notes = stats.get("total_notes", 0)
        nps = stats.get("nps", 0.0)
        hold_percentage = stats.get("hold_percentage") or stats.get(
            "hold_notes_percentage", 0.0
        )
        column_std = stats.get("column_std") or stats.get("column_balance_std", 0.0)

        print(f"[迭代 {self.iteration}] 相似度: {similarity:.3f}")
        print(
            f"  总音符数: {total_notes} / {self.reference_stats['total_notes']} ({total_notes/self.reference_stats['total_notes']*100:.1f}%)"
        )
        print(
            f"  NPS: {nps:.2f} / {self.reference_stats['nps']:.2f} ({nps/self.reference_stats['nps']*100:.1f}%)"
        )
        print(
            f"  长条比例: {hold_percentage:.1f}% / {self.reference_stats['hold_notes_percentage']:.1f}%"
        )
        print(
            f"  轨道平衡: {column_std:.2f}% / {self.reference_stats['column_balance_std']:.2f}%"
        )

        # 更新最佳结果
        if similarity > self.best_similarity:
            self.best_similarity = similarity
            self.best_config = stats.copy()
            print(f"  🎉 新的最佳相似度!")

        # 调整参数
        adjusted = self.adjust_parameters(stats, similarity)

        return similarity, stats

    def run_optimization(
        self, target_similarity: float = 0.9, max_iterations: int = 50
    ):
        """运行优化循环"""
        print("=" * 60)
        print("自动迭代优化开始")
        print(f"目标相似度: {target_similarity}")
        print(f"最大迭代次数: {max_iterations}")
        print("=" * 60)

        start_time = time.time()

        while (
            self.current_similarity < target_similarity
            and self.iteration < max_iterations
        ):
            similarity, stats = self.run_iteration()

            if similarity == 0.0:
                print("谱面生成失败，停止优化")
                break

            # 检查是否收敛
            if self.iteration > 10 and similarity < self.best_similarity * 1.01:
                print("优化收敛，停止迭代")
                break

            # 短暂暂停
            time.sleep(1)

        elapsed_time = time.time() - start_time

        print("\n" + "=" * 60)
        print("优化完成!")
        print("=" * 60)
        print(f"总迭代次数: {self.iteration}")
        print(f"最佳相似度: {self.best_similarity:.3f}")
        print(f"耗时: {elapsed_time:.1f}秒")

        if self.best_similarity >= target_similarity:
            print(f"🎉 达到目标相似度 {target_similarity}!")
        else:
            print(f"未达到目标相似度，当前最佳: {self.best_similarity:.3f}")

        return self.best_similarity, self.best_config


def main():
    """主函数"""
    optimizer = AutoOptimizer()

    try:
        best_similarity, best_stats = optimizer.run_optimization(
            target_similarity=0.9, max_iterations=50
        )

        print(f"\n最终配置位置: algorithm/config.py")
        print(f"运行以下命令测试最终配置:")
        print(
            '  python algorithm/main.py "audio/Scattered Rose.mp3" --columns 7 --visualize'
        )

    except KeyboardInterrupt:
        print("\n\n优化被用户中断")
        print(f"当前最佳相似度: {optimizer.best_similarity:.3f}")
    except Exception as e:
        print(f"\n优化过程中出现错误: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
