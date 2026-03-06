#!/usr/bin/env python3
"""
进化算法优化器 - 使用遗传算法优化谱面生成参数
"""

import os
import sys
import json
import random
import re
import subprocess
import time
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import shutil

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from algorithm.config import Config

# 导入.osu文件解析器
from algorithm.osu_parser import parse_osu_file


class EvolutionaryOptimizer:
    def __init__(self, population_size: int = 10, mutation_rate: float = 0.2):
        """
        初始化进化算法优化器

        参数:
            population_size: 种群大小
            mutation_rate: 变异率
        """
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.generation = 0
        self.best_fitness = 0
        self.best_individual = None
        self.best_stats = None

        # 加载参考谱面统计
        self.reference_stats = self.load_reference_stats()

        # 初始化种群
        self.population = self.initialize_population()

        # 创建输出目录
        self.output_dir = "output/optimization_experiments"
        os.makedirs(self.output_dir, exist_ok=True)

    def load_reference_stats(self) -> Dict:
        """加载参考谱面统计"""
        with open("output/reference_stats.json", "r", encoding="utf-8") as f:
            return json.load(f)

    def initialize_population(self) -> List[Dict]:
        """初始化种群"""
        population = []

        # 基础配置
        base_config = {
            # 二值化参数
            "ADAPTIVE_THRESHOLD_BLOCK_SIZE": 15,
            "ADAPTIVE_THRESHOLD_C": -8,
            "MORPH_KERNEL_SIZE": 2,
            # 音符检测参数
            "MIN_NOTE_DURATION_MS": 5,
            "MAX_NOTE_DURATION_MS": 2000,
            "NOTE_GAP_MS": 10,
            # 密度控制参数
            "DENSITY_FILTER_RATIO": 0.5,
            "MAX_NOTES_PER_BEAT": 16,
            "MAX_SAME_COLUMN_INTERVAL_MS": 50,
            # 长条参数
            "HOLD_NOTE_MIN_DURATION": 300,
            "HOLD_NOTE_MAX_DURATION": 800,
            "HOLD_NOTE_TARGET_PERCENTAGE": 15,
            # 轨道平衡参数
            "COLUMN_BALANCE_TARGET_STD": 2.0,
            "COLUMN_REBALANCE_THRESHOLD": 0.2,
            # 物理手感参数
            "PHYSICAL_CORRECTION_STRICTNESS": 0.5,
        }

        # 创建变异个体
        for i in range(self.population_size):
            individual = base_config.copy()

            # 随机变异
            if random.random() < 0.5:
                individual["ADAPTIVE_THRESHOLD_BLOCK_SIZE"] = random.randint(5, 25)
                individual["ADAPTIVE_THRESHOLD_C"] = random.uniform(-15, 0)
                individual["MORPH_KERNEL_SIZE"] = random.randint(1, 5)

            if random.random() < 0.5:
                individual["DENSITY_FILTER_RATIO"] = random.uniform(0.1, 0.9)
                individual["MAX_NOTES_PER_BEAT"] = random.randint(8, 32)
                individual["MAX_SAME_COLUMN_INTERVAL_MS"] = random.randint(20, 100)

            if random.random() < 0.5:
                individual["HOLD_NOTE_MIN_DURATION"] = random.randint(100, 500)
                individual["HOLD_NOTE_MAX_DURATION"] = random.randint(500, 1500)
                individual["HOLD_NOTE_TARGET_PERCENTAGE"] = random.uniform(5, 30)

            if random.random() < 0.5:
                individual["COLUMN_BALANCE_TARGET_STD"] = random.uniform(1.0, 5.0)
                individual["COLUMN_REBALANCE_THRESHOLD"] = random.uniform(0.1, 0.5)
                individual["PHYSICAL_CORRECTION_STRICTNESS"] = random.uniform(0.1, 1.0)

            population.append(individual)

        return population

    def update_config_file(self, individual: Dict, iteration: int):
        """更新config.py文件"""
        config_path = "algorithm/config.py"

        with open(config_path, "r", encoding="utf-8") as f:
            content = f.read()

        # 更新参数
        for param_name, param_value in individual.items():
            if isinstance(param_value, float):
                search_str = f"{param_name} = "
                pattern = rf"{search_str}[0-9\.\-]+"
                replacement = f"{search_str}{param_value}"
                content = re.sub(pattern, replacement, content)
            elif isinstance(param_value, int):
                search_str = f"{param_name} = "
                pattern = rf"{search_str}[0-9\-]+"
                replacement = f"{search_str}{param_value}"
                content = re.sub(pattern, replacement, content)

        # 保存更新后的配置
        with open(config_path, "w", encoding="utf-8") as f:
            f.write(content)

    def generate_beatmap(
        self, individual: Dict, iteration: int
    ) -> Tuple[Optional[str], Optional[Dict]]:
        """生成谱面并返回统计信息"""
        # 更新配置文件
        self.update_config_file(individual, iteration)

        # 运行主程序生成谱面
        cmd = [
            "python",
            "algorithm/main.py",
            "audio/Scattered Rose.mp3",
            "--columns",
            "7",
            "--visualize",
            "--output-dir",
            self.output_dir,
            "--iteration",
            str(iteration),
        ]

        print(f"  生成谱面 (迭代 {iteration})...")

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)

            # 查找生成的.osu文件
            osu_files = []
            for file in os.listdir(self.output_dir):
                if file.endswith(f"_iter{iteration}_7K.osu"):
                    osu_files.append(os.path.join(self.output_dir, file))

            if not osu_files:
                print(f"  警告: 未找到生成的.osu文件")
                return None, None

            # 解析第一个找到的.osu文件
            osu_file = osu_files[0]
            stats = parse_osu_file(osu_file)

            print(f"  ✓ 谱面生成成功: {os.path.basename(osu_file)}")
            return osu_file, stats

        except subprocess.CalledProcessError as e:
            print(f"  ✗ 谱面生成失败: {e}")
            return None, None

    def calculate_fitness(self, stats: Dict) -> float:
        """计算适应度（与参考谱面的相似度）"""
        if not stats:
            return 0.0

        # 计算各项指标的相似度
        metrics = {}

        # 1. 总音符数相似度
        target_notes = self.reference_stats["total_notes"]
        actual_notes = stats["total_notes"]
        notes_similarity = 1.0 - min(
            abs(actual_notes - target_notes) / target_notes, 1.0
        )
        metrics["notes"] = notes_similarity

        # 2. NPS相似度
        target_nps = self.reference_stats["nps"]
        actual_nps = stats["nps"]
        nps_similarity = 1.0 - min(abs(actual_nps - target_nps) / target_nps, 1.0)
        metrics["nps"] = nps_similarity

        # 3. 长条比例相似度
        target_hold = self.reference_stats["hold_notes_percentage"]
        actual_hold = stats["hold_notes_percentage"]
        hold_similarity = 1.0 - min(abs(actual_hold - target_hold) / 100, 1.0)
        metrics["hold"] = hold_similarity

        # 4. 轨道平衡相似度
        target_std = self.reference_stats["column_balance_std"]
        actual_std = stats["column_balance_std"]
        std_similarity = 1.0 - min(abs(actual_std - target_std) / 5.0, 1.0)
        metrics["std"] = std_similarity

        # 加权综合适应度
        weights = {"notes": 0.3, "nps": 0.3, "hold": 0.2, "std": 0.2}

        fitness = sum(metrics[metric] * weights[metric] for metric in metrics)

        # 惩罚过大的偏差
        if any(similarity < 0.5 for similarity in metrics.values()):
            fitness *= 0.8

        return fitness

    def evaluate_population(self) -> List[Tuple[float, Dict, Dict]]:
        """评估种群中所有个体"""
        evaluations = []

        for i, individual in enumerate(self.population):
            print(f"\n评估个体 {i+1}/{self.population_size}:")

            # 生成谱面
            iteration = self.generation * self.population_size + i + 1
            osu_file, stats = self.generate_beatmap(individual, iteration)

            # 计算适应度
            if stats:
                fitness = self.calculate_fitness(stats)

                # 打印详细统计
                print(f"  适应度: {fitness:.4f}")
                print(
                    f"  总音符数: {stats['total_notes']} (目标: {self.reference_stats['total_notes']})"
                )
                print(
                    f"  NPS: {stats['nps']:.2f} (目标: {self.reference_stats['nps']:.2f})"
                )
                print(
                    f"  长条比例: {stats['hold_notes_percentage']:.1f}% (目标: {self.reference_stats['hold_notes_percentage']:.1f}%)"
                )
                print(
                    f"  轨道平衡标准差: {stats['column_balance_std']:.3f} (目标: {self.reference_stats['column_balance_std']:.3f})"
                )

                evaluations.append((fitness, individual, stats))
            else:
                print(f"  适应度: 0.0 (生成失败)")
                evaluations.append((0.0, individual, {}))

        return evaluations

    def selection(self, evaluations: List[Tuple[float, Dict, Dict]]) -> List[Dict]:
        """选择操作 - 轮盘赌选择"""
        # 按适应度排序
        evaluations.sort(key=lambda x: x[0], reverse=True)

        # 计算适应度总和
        total_fitness = sum(fitness for fitness, _, _ in evaluations)

        # 轮盘赌选择
        selected = []
        for _ in range(self.population_size):
            r = random.uniform(0, total_fitness)
            cumulative = 0
            for fitness, individual, _ in evaluations:
                cumulative += fitness
                if cumulative >= r:
                    selected.append(individual.copy())
                    break

        return selected

    def crossover(self, parent1: Dict, parent2: Dict) -> Dict:
        """交叉操作 - 均匀交叉"""
        child = {}

        for param in parent1:
            if random.random() < 0.5:
                child[param] = parent1[param]
            else:
                child[param] = parent2[param]

        return child

    def mutation(self, individual: Dict) -> Dict:
        """变异操作"""
        mutated = individual.copy()

        for param in mutated:
            if random.random() < self.mutation_rate:
                if param == "ADAPTIVE_THRESHOLD_BLOCK_SIZE":
                    mutated[param] = random.randint(5, 25)
                elif param == "ADAPTIVE_THRESHOLD_C":
                    mutated[param] = random.uniform(-15, 0)
                elif param == "MORPH_KERNEL_SIZE":
                    mutated[param] = random.randint(1, 5)
                elif param == "DENSITY_FILTER_RATIO":
                    mutated[param] = random.uniform(0.1, 0.9)
                elif param == "MAX_NOTES_PER_BEAT":
                    mutated[param] = random.randint(8, 32)
                elif param == "MAX_SAME_COLUMN_INTERVAL_MS":
                    mutated[param] = random.randint(20, 100)
                elif param == "HOLD_NOTE_MIN_DURATION":
                    mutated[param] = random.randint(100, 500)
                elif param == "HOLD_NOTE_MAX_DURATION":
                    mutated[param] = random.randint(500, 1500)
                elif param == "HOLD_NOTE_TARGET_PERCENTAGE":
                    mutated[param] = random.uniform(5, 30)
                elif param == "COLUMN_BALANCE_TARGET_STD":
                    mutated[param] = random.uniform(1.0, 5.0)
                elif param == "COLUMN_REBALANCE_THRESHOLD":
                    mutated[param] = random.uniform(0.1, 0.5)
                elif param == "PHYSICAL_CORRECTION_STRICTNESS":
                    mutated[param] = random.uniform(0.1, 1.0)

        return mutated

    def evolve(self):
        """进化一代"""
        print(f"\n{'='*60}")
        print(f"第 {self.generation + 1} 代进化")
        print(f"{'='*60}")

        # 评估当前种群
        evaluations = self.evaluate_population()

        # 更新最佳个体
        best_in_generation = max(evaluations, key=lambda x: x[0])
        best_fitness, best_individual, best_stats = best_in_generation

        if best_fitness > self.best_fitness:
            self.best_fitness = best_fitness
            self.best_individual = best_individual.copy()
            self.best_stats = best_stats.copy() if best_stats else {}
            print(f"\n🎉 新的最佳适应度: {best_fitness:.4f}")

        # 选择
        selected = self.selection(evaluations)

        # 交叉和变异
        new_population = []
        for i in range(0, len(selected), 2):
            if i + 1 < len(selected):
                parent1 = selected[i]
                parent2 = selected[i + 1]

                # 交叉
                child1 = self.crossover(parent1, parent2)
                child2 = self.crossover(parent2, parent1)

                # 变异
                child1 = self.mutation(child1)
                child2 = self.mutation(child2)

                new_population.extend([child1, child2])
            else:
                # 奇数情况，直接复制
                new_population.append(self.mutation(selected[i]))

        # 确保种群大小不变
        self.population = new_population[: self.population_size]

        # 保留最佳个体（精英保留）
        if self.best_individual and random.random() < 0.3:
            self.population[0] = self.best_individual.copy()

        self.generation += 1

        return best_fitness, best_individual, best_stats

    def run_optimization(self, max_generations: int = 20, target_fitness: float = 0.9):
        """运行优化循环"""
        print("=" * 60)
        print("进化算法优化开始")
        print(f"种群大小: {self.population_size}")
        print(f"变异率: {self.mutation_rate}")
        print(f"最大代数: {max_generations}")
        print(f"目标适应度: {target_fitness}")
        print("=" * 60)

        start_time = time.time()
        convergence_count = 0

        for gen in range(max_generations):
            best_fitness, best_individual, best_stats = self.evolve()

            print(f"\n第 {gen + 1} 代结果:")
            print(f"  最佳适应度: {best_fitness:.4f}")
            print(f"  目标适应度: {target_fitness}")

            # 检查是否达到目标
            if best_fitness >= target_fitness:
                print(f"\n🎉 达到目标适应度 {target_fitness}!")
                break

            # 检查收敛
            if gen > 5:
                if abs(best_fitness - self.best_fitness) < 0.01:
                    convergence_count += 1
                else:
                    convergence_count = 0

                if convergence_count >= 3:
                    print(f"\n优化收敛，停止进化")
                    break

            # 短暂暂停
            time.sleep(1)

        elapsed_time = time.time() - start_time

        print("\n" + "=" * 60)
        print("优化完成!")
        print("=" * 60)
        print(f"总代数: {self.generation}")
        print(f"最佳适应度: {self.best_fitness:.4f}")
        print(f"耗时: {elapsed_time:.1f}秒")

        # 保存最佳配置
        self.save_best_config()

        return self.best_fitness, self.best_individual, self.best_stats

    def save_best_config(self):
        """保存最佳配置"""
        if not self.best_individual:
            return

        # 更新配置文件为最佳配置
        self.update_config_file(self.best_individual, 9999)  # 使用特殊迭代编号

        # 保存最佳配置到JSON文件
        best_config_path = os.path.join(self.output_dir, "best_config.json")
        with open(best_config_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "best_fitness": self.best_fitness,
                    "best_individual": self.best_individual,
                    "best_stats": self.best_stats,
                    "generation": self.generation,
                    "reference_stats": self.reference_stats,
                },
                f,
                indent=2,
            )

        print(f"✓ 最佳配置已保存到: {best_config_path}")

        # 使用最佳配置生成最终谱面
        print("\n使用最佳配置生成最终谱面...")
        final_osu_file, final_stats = self.generate_beatmap(self.best_individual, 9999)

        if final_osu_file:
            print(f"✓ 最终谱面已生成: {os.path.basename(final_osu_file)}")

            # 保存最终统计
            final_stats_path = os.path.join(self.output_dir, "final_stats.json")
            with open(final_stats_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "final_stats": final_stats,
                        "reference_stats": self.reference_stats,
                        "similarity_score": self.best_fitness,
                    },
                    f,
                    indent=2,
                )

            print(f"✓ 最终统计已保存到: {final_stats_path}")


def main():
    """主函数"""
    print("=" * 60)
    print("进化算法谱面优化器")
    print("=" * 60)

    # 创建优化器
    optimizer = EvolutionaryOptimizer(
        population_size=8,  # 较小的种群大小以加快测试
        mutation_rate=0.3,  # 较高的变异率以增加探索
    )

    try:
        # 运行优化
        best_fitness, best_individual, best_stats = optimizer.run_optimization(
            max_generations=10,  # 较少的代数以进行测试
            target_fitness=0.85,  # 目标相似度85%
        )

        print(f"\n优化结果:")
        print(f"  最佳适应度: {best_fitness:.4f}")
        print(f"  最佳配置已保存到 output/optimization_experiments/best_config.json")

        if best_fitness >= 0.85:
            print(f"\n🎉 成功达到目标相似度!")
        else:
            print(f"\n⚠️ 未达到目标相似度，但找到了最佳配置")

        print(f"\n运行以下命令测试最终配置:")
        print(
            '  python algorithm/main.py "audio/Scattered Rose.mp3" --columns 7 --visualize'
        )

    except KeyboardInterrupt:
        print("\n\n优化被用户中断")
        print(f"当前最佳适应度: {optimizer.best_fitness:.4f}")
    except Exception as e:
        print(f"\n优化过程中出现错误: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
