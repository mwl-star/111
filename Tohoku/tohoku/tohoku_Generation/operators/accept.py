"""
接受类算子 - 用于解的接受/拒绝决策的原子操作
包含：贪婪接受、Metropolis接受、阈值接受、锦标赛选择等
"""
import numpy as np
from typing import Dict, Any
from .base import AcceptPrimitive
from .op_types import OperatorContext, OperatorResult


class GreedyAccept(AcceptPrimitive):
    """
    贪婪接受
    只接受比原解更优的新解
    """

    def __init__(self):
        super().__init__("greedy_accept", {})

    def execute(self, ctx: OperatorContext) -> OperatorResult:
        new_pop = ctx.population.copy()
        new_fitness = ctx.fitness.copy()

        # 获取原始种群和适应度
        old_pop = ctx.params.get('original_pop', new_pop.copy())
        old_fitness = ctx.params.get('old_fitness', new_fitness.copy())

        # 贪婪接受：只接受更优解
        accepted_count = 0
        for i in range(len(new_pop)):
            if new_fitness[i] < old_fitness[i]:
                accepted_count += 1
            else:
                new_pop[i] = old_pop[i]
                new_fitness[i] = old_fitness[i]

        # 更新最优
        min_idx = np.argmin(new_fitness)
        best_solution = new_pop[min_idx].copy()
        best_fitness = new_fitness[min_idx]

        return OperatorResult(
            population=new_pop,
            fitness=new_fitness,
            best_solution=best_solution,
            best_fitness=best_fitness,
            evaluations_used=0,
            success=accepted_count > 0,
            stats={'accepted_count': accepted_count, 'accept_rate': accepted_count / len(new_pop)}
        )


class MetropolisAccept(AcceptPrimitive):
    """
    Metropolis接受准则（模拟退火）
    以一定概率接受较差的解，概率随温度降低
    """

    def __init__(self, temperature: float = 1.0, cooling_rate: float = 0.99):
        """
        Args:
            temperature: 初始温度
            cooling_rate: 降温系数
        """
        super().__init__("metropolis_accept", {
            'temperature': temperature,
            'cooling_rate': cooling_rate
        })

    def execute(self, ctx: OperatorContext) -> OperatorResult:
        new_pop = ctx.population.copy()
        new_fitness = ctx.fitness.copy()

        # 获取原始种群和适应度
        old_pop = ctx.params.get('original_pop', new_pop.copy())
        old_fitness = ctx.params.get('old_fitness', new_fitness.copy())

        # 获取当前温度（可能从上下文获取）
        T = ctx.params.get('temperature', self.params['temperature'])
        cooling_rate = self.params['cooling_rate']

        accepted_count = 0
        for i in range(len(new_pop)):
            delta = new_fitness[i] - old_fitness[i]

            # 如果更优，直接接受
            if delta < 0:
                accepted_count += 1
            # 否则，以Metropolis概率接受
            elif np.random.rand() < np.exp(-delta / T):
                accepted_count += 1
            else:
                new_pop[i] = old_pop[i]
                new_fitness[i] = old_fitness[i]

        # 降温
        T = T * cooling_rate

        # 更新最优
        min_idx = np.argmin(new_fitness)
        best_solution = new_pop[min_idx].copy()
        best_fitness = new_fitness[min_idx]

        return OperatorResult(
            population=new_pop,
            fitness=new_fitness,
            best_solution=best_solution,
            best_fitness=best_fitness,
            evaluations_used=0,
            success=accepted_count > 0,
            stats={'temperature': T, 'accepted_count': accepted_count},
            extra={'temperature': T}
        )


class ThresholdAccept(AcceptPrimitive):
    """
    阈值接受
    只接受改进量超过阈值的解
    """

    def __init__(self, threshold: float = 0.01, threshold_decay: float = 0.95):
        """
        Args:
            threshold: 接受阈值
            threshold_decay: 阈值衰减系数
        """
        super().__init__("threshold_accept", {
            'threshold': threshold,
            'threshold_decay': threshold_decay
        })

    def execute(self, ctx: OperatorContext) -> OperatorResult:
        new_pop = ctx.population.copy()
        new_fitness = ctx.fitness.copy()

        # 获取原始种群和适应度
        old_pop = ctx.params.get('original_pop', new_pop.copy())
        old_fitness = ctx.params.get('old_fitness', new_fitness.copy())

        # 获取当前阈值
        threshold = ctx.params.get('threshold', self.params['threshold'])
        threshold_decay = self.params['threshold_decay']

        accepted_count = 0
        for i in range(len(new_pop)):
            delta = old_fitness[i] - new_fitness[i]  # 改进量（正值表示改进）

            # 如果改进量超过阈值，接受
            if delta > threshold:
                accepted_count += 1
            else:
                new_pop[i] = old_pop[i]
                new_fitness[i] = old_fitness[i]

        # 阈值衰减
        threshold = threshold * threshold_decay

        # 更新最优
        min_idx = np.argmin(new_fitness)
        best_solution = new_pop[min_idx].copy()
        best_fitness = new_fitness[min_idx]

        return OperatorResult(
            population=new_pop,
            fitness=new_fitness,
            best_solution=best_solution,
            best_fitness=best_fitness,
            evaluations_used=0,
            success=accepted_count > 0,
            stats={'threshold': threshold, 'accepted_count': accepted_count},
            extra={'threshold': threshold}
        )


class TournamentSelection(AcceptPrimitive):
    """
    锦标赛选择
    通过锦标赛竞争选择个体
    """

    def __init__(self, tournament_size: int = 3, selection_pressure: float = 1.0):
        """
        Args:
            tournament_size: 锦标赛大小
            selection_pressure: 选择压力
        """
        super().__init__("tournament_selection", {
            'tournament_size': tournament_size,
            'selection_pressure': selection_pressure
        })

    def execute(self, ctx: OperatorContext) -> OperatorResult:
        pop = ctx.population.copy()
        fitness = ctx.fitness.copy()
        n = len(pop)

        tournament_size = self.params['tournament_size']

        new_pop = np.zeros_like(pop)
        new_fitness = np.zeros(n)

        for i in range(n):
            # 随机选择tournament_size个个体
            candidates = np.random.choice(n, tournament_size, replace=False)

            # 选择最优的
            winner_idx = candidates[np.argmin(fitness[candidates])]
            new_pop[i] = pop[winner_idx]
            new_fitness[i] = fitness[winner_idx]

        # 更新最优
        min_idx = np.argmin(new_fitness)
        best_solution = new_pop[min_idx].copy()
        best_fitness = new_fitness[min_idx]

        return OperatorResult(
            population=new_pop,
            fitness=new_fitness,
            best_solution=best_solution,
            best_fitness=best_fitness,
            evaluations_used=0,
            stats={'tournament_size': tournament_size}
        )


class ElitismAccept(AcceptPrimitive):
    """
    精英保留接受
    保留最优个体，其余按贪婪接受
    """

    def __init__(self, elite_size: int = 1):
        """
        Args:
            elite_size: 精英个体数量
        """
        super().__init__("elitism_accept", {'elite_size': elite_size})

    def execute(self, ctx: OperatorContext) -> OperatorResult:
        new_pop = ctx.population.copy()
        new_fitness = ctx.fitness.copy()

        # 获取原始种群和适应度
        old_pop = ctx.params.get('original_pop', new_pop.copy())
        old_fitness = ctx.params.get('old_fitness', new_fitness.copy())

        elite_size = self.params['elite_size']
        n = len(new_pop)

        # 找出原始种群中的精英
        elite_indices = np.argsort(old_fitness)[:elite_size]

        # 贪婪接受
        for i in range(n):
            if i in elite_indices:
                # 精英位置：保留原始精英或接受更优解
                if new_fitness[i] < old_fitness[i]:
                    pass  # 接受新解
                else:
                    new_pop[i] = old_pop[i]
                    new_fitness[i] = old_fitness[i]
            else:
                # 非精英位置：贪婪接受
                if new_fitness[i] < old_fitness[i]:
                    pass  # 接受新解
                else:
                    new_pop[i] = old_pop[i]
                    new_fitness[i] = old_fitness[i]

        # 更新最优
        min_idx = np.argmin(new_fitness)
        best_solution = new_pop[min_idx].copy()
        best_fitness = new_fitness[min_idx]

        return OperatorResult(
            population=new_pop,
            fitness=new_fitness,
            best_solution=best_solution,
            best_fitness=best_fitness,
            evaluations_used=0,
            stats={'elite_size': elite_size}
        )


class ProbabilisticAccept(AcceptPrimitive):
    """
    概率接受
    根据适应度差异以概率接受新解
    """

    def __init__(self, accept_prob: float = 0.5):
        """
        Args:
            accept_prob: 基础接受概率
        """
        super().__init__("probabilistic_accept", {'accept_prob': accept_prob})

    def execute(self, ctx: OperatorContext) -> OperatorResult:
        new_pop = ctx.population.copy()
        new_fitness = ctx.fitness.copy()

        # 获取原始种群和适应度
        old_pop = ctx.params.get('original_pop', new_pop.copy())
        old_fitness = ctx.params.get('old_fitness', new_fitness.copy())

        base_prob = self.params['accept_prob']

        accepted_count = 0
        for i in range(len(new_pop)):
            delta = old_fitness[i] - new_fitness[i]

            # 计算接受概率
            if delta > 0:  # 改进
                prob = base_prob + (1 - base_prob) * min(delta / old_fitness[i], 1.0)
            else:  # 退化
                prob = base_prob * max(0.1, 1 - abs(delta) / old_fitness[i])

            if np.random.rand() < prob:
                accepted_count += 1
            else:
                new_pop[i] = old_pop[i]
                new_fitness[i] = old_fitness[i]

        # 更新最优
        min_idx = np.argmin(new_fitness)
        best_solution = new_pop[min_idx].copy()
        best_fitness = new_fitness[min_idx]

        return OperatorResult(
            population=new_pop,
            fitness=new_fitness,
            best_solution=best_solution,
            best_fitness=best_fitness,
            evaluations_used=0,
            success=accepted_count > 0,
            stats={'accepted_count': accepted_count}
        )


class ArchiveUpdate(AcceptPrimitive):
    """
    存档更新
    将改进的解存入历史最优存档
    """

    def __init__(self, archive_size: int = 10):
        """
        Args:
            archive_size: 存档大小
        """
        super().__init__("archive_update", {'archive_size': archive_size})

    def execute(self, ctx: OperatorContext) -> OperatorResult:
        archive = ctx.archive.copy() if ctx.archive else []
        archive_size = self.params['archive_size']

        # 添加当前最优解到存档
        if ctx.best_solution is not None:
            archive.append(ctx.best_solution.copy())

            # 保持存档大小
            if len(archive) > archive_size:
                archive = archive[-archive_size:]

        return OperatorResult(
            population=ctx.population.copy(),
            fitness=ctx.fitness.copy(),
            best_solution=ctx.best_solution.copy(),
            best_fitness=ctx.best_fitness,
            evaluations_used=0,
            stats={'archive_size': len(archive)},
            extra={'archive': archive}
        )


# 导出所有接受类算子
__all__ = [
    'GreedyAccept',
    'MetropolisAccept',
    'ThresholdAccept',
    'TournamentSelection',
    'ElitismAccept',
    'ProbabilisticAccept',
    'ArchiveUpdate'
]