"""
重组类算子 - 用于解的融合和交叉的原子操作
包含：算术交叉、SBX交叉、DE交叉、均匀交叉等
"""
import numpy as np
from typing import Dict, Any
from .base import RecombinePrimitive
from .op_types import OperatorContext, OperatorResult


class ArithmeticCrossover(RecombinePrimitive):
    """
    算术交叉
    线性组合两个个体生成新个体
    """

    def __init__(self, cr: float = 0.9, alpha: float = None):
        """
        Args:
            cr: 交叉概率
            alpha: 混合系数 (None表示随机)
        """
        super().__init__("arithmetic_crossover", {'cr': cr, 'alpha': alpha})

    def execute(self, ctx: OperatorContext) -> OperatorResult:
        pop = ctx.population.copy()
        n = len(pop)

        # 获取变异后的种群（如果有）
        mutant = ctx.params.get('mutant', pop.copy())

        cr = self.params['cr']
        alpha = self.params['alpha']

        new_pop = pop.copy()

        for i in range(n):
            if np.random.rand() < cr:
                # 选择另一个个体
                partner_idx = np.random.choice([j for j in range(n) if j != i])

                # 混合系数
                a = alpha if alpha is not None else np.random.rand()

                # 算术交叉
                new_pop[i] = a * pop[i] + (1 - a) * pop[partner_idx]

        new_pop = self._clip_to_bounds(new_pop, ctx.bounds)

        # 评估
        new_fitness = np.array([ctx.objective_func(ind) for ind in new_pop])

        # 更新最优
        min_idx = np.argmin(new_fitness)
        best_solution = ctx.best_solution.copy()
        best_fitness = ctx.best_fitness
        improvement = 0.0

        if new_fitness[min_idx] < best_fitness:
            improvement = best_fitness - new_fitness[min_idx]
            best_solution = new_pop[min_idx].copy()
            best_fitness = new_fitness[min_idx]

        self._update_stats(improvement, n, improvement > 0)

        return OperatorResult(
            population=new_pop,
            fitness=new_fitness,
            best_solution=best_solution,
            best_fitness=best_fitness,
            evaluations_used=n,
            success=improvement > 0,
            stats={'cr': cr}
        )


class SBXCrossover(RecombinePrimitive):
    """
    模拟二进制交叉 (Simulated Binary Crossover)
    模拟二进制编码的单点交叉行为
    """

    def __init__(self, cr: float = 0.9, eta: float = 20.0):
        """
        Args:
            cr: 交叉概率
            eta: 分布指数 (越大子代越接近父代)
        """
        super().__init__("sbx_crossover", {'cr': cr, 'eta': eta})

    def execute(self, ctx: OperatorContext) -> OperatorResult:
        pop = ctx.population.copy()
        n = len(pop)
        dim = pop.shape[1]
        lb, ub = ctx.bounds

        cr = self.params['cr']
        eta = self.params['eta']

        new_pop = pop.copy()

        for i in range(0, n - 1, 2):
            if np.random.rand() < cr:
                parent1, parent2 = pop[i], pop[i + 1]

                for j in range(dim):
                    if np.random.rand() < 0.5:
                        # 计算beta
                        u = np.random.rand()
                        if u <= 0.5:
                            beta = (2 * u) ** (1 / (eta + 1))
                        else:
                            beta = (1 / (2 * (1 - u))) ** (1 / (eta + 1))

                        # 生成子代
                        child1 = 0.5 * ((1 + beta) * parent1[j] + (1 - beta) * parent2[j])
                        child2 = 0.5 * ((1 - beta) * parent1[j] + (1 + beta) * parent2[j])

                        new_pop[i, j] = np.clip(child1, lb, ub)
                        new_pop[i + 1, j] = np.clip(child2, lb, ub)

        # 评估
        new_fitness = np.array([ctx.objective_func(ind) for ind in new_pop])

        # 更新最优
        min_idx = np.argmin(new_fitness)
        best_solution = ctx.best_solution.copy()
        best_fitness = ctx.best_fitness
        improvement = 0.0

        if new_fitness[min_idx] < best_fitness:
            improvement = best_fitness - new_fitness[min_idx]
            best_solution = new_pop[min_idx].copy()
            best_fitness = new_fitness[min_idx]

        self._update_stats(improvement, n, improvement > 0)

        return OperatorResult(
            population=new_pop,
            fitness=new_fitness,
            best_solution=best_solution,
            best_fitness=best_fitness,
            evaluations_used=n,
            success=improvement > 0,
            stats={'eta': eta}
        )


class DECrossover(RecombinePrimitive):
    """
    差分进化交叉
    将变异向量与目标向量混合
    """

    def __init__(self, cr: float = 0.9):
        """
        Args:
            cr: 交叉概率
        """
        super().__init__("de_crossover", {'cr': cr})

    def execute(self, ctx: OperatorContext) -> OperatorResult:
        pop = ctx.population.copy()
        n = len(pop)
        dim = pop.shape[1]

        # 获取变异向量
        mutant = ctx.params.get('mutant', pop.copy())

        cr = self.params['cr']

        trial = np.zeros_like(pop)

        for i in range(n):
            # 确保至少一个维度来自变异向量
            j_rand = np.random.randint(dim)

            for j in range(dim):
                if np.random.rand() < cr or j == j_rand:
                    trial[i, j] = mutant[i, j]
                else:
                    trial[i, j] = pop[i, j]

        trial = self._clip_to_bounds(trial, ctx.bounds)

        # 评估
        trial_fitness = np.array([ctx.objective_func(ind) for ind in trial])

        # 更新最优
        min_idx = np.argmin(trial_fitness)
        best_solution = ctx.best_solution.copy()
        best_fitness = ctx.best_fitness
        improvement = 0.0

        if trial_fitness[min_idx] < best_fitness:
            improvement = best_fitness - trial_fitness[min_idx]
            best_solution = trial[min_idx].copy()
            best_fitness = trial_fitness[min_idx]

        self._update_stats(improvement, n, improvement > 0)

        return OperatorResult(
            population=trial,
            fitness=trial_fitness,
            best_solution=best_solution,
            best_fitness=best_fitness,
            evaluations_used=n,
            success=improvement > 0,
            stats={'cr': cr},
            extra={'original_pop': pop.copy()}
        )


class UniformCrossover(RecombinePrimitive):
    """
    均匀交叉
    每个维度随机选择来自哪个父代
    """

    def __init__(self, cr: float = 0.5):
        """
        Args:
            cr: 选择第一个父代的概率
        """
        super().__init__("uniform_crossover", {'cr': cr})

    def execute(self, ctx: OperatorContext) -> OperatorResult:
        pop = ctx.population.copy()
        n = len(pop)

        cr = self.params['cr']

        new_pop = pop.copy()

        for i in range(0, n - 1, 2):
            parent1, parent2 = pop[i], pop[i + 1]

            # 均匀交叉掩码
            mask = np.random.rand(pop.shape[1]) < cr

            new_pop[i] = np.where(mask, parent1, parent2)
            new_pop[i + 1] = np.where(mask, parent2, parent1)

        new_pop = self._clip_to_bounds(new_pop, ctx.bounds)

        # 评估
        new_fitness = np.array([ctx.objective_func(ind) for ind in new_pop])

        # 更新最优
        min_idx = np.argmin(new_fitness)
        best_solution = ctx.best_solution.copy()
        best_fitness = ctx.best_fitness
        improvement = 0.0

        if new_fitness[min_idx] < best_fitness:
            improvement = best_fitness - new_fitness[min_idx]
            best_solution = new_pop[min_idx].copy()
            best_fitness = new_fitness[min_idx]

        self._update_stats(improvement, n, improvement > 0)

        return OperatorResult(
            population=new_pop,
            fitness=new_fitness,
            best_solution=best_solution,
            best_fitness=best_fitness,
            evaluations_used=n,
            success=improvement > 0,
            stats={'cr': cr}
        )


class BlendCrossover(RecombinePrimitive):
    """
    混合交叉 (BLX-α)
    在父代定义的扩展区间内随机生成子代
    """

    def __init__(self, alpha: float = 0.5, cr: float = 0.9):
        """
        Args:
            alpha: 扩展因子
            cr: 交叉概率
        """
        super().__init__("blend_crossover", {'alpha': alpha, 'cr': cr})

    def execute(self, ctx: OperatorContext) -> OperatorResult:
        pop = ctx.population.copy()
        n = len(pop)
        dim = pop.shape[1]
        lb, ub = ctx.bounds

        alpha = self.params['alpha']
        cr = self.params['cr']

        new_pop = pop.copy()

        for i in range(0, n - 1, 2):
            if np.random.rand() < cr:
                parent1, parent2 = pop[i], pop[i + 1]

                for j in range(dim):
                    x_min = min(parent1[j], parent2[j])
                    x_max = max(parent1[j], parent2[j])
                    range_j = x_max - x_min

                    # 扩展区间
                    low = x_min - alpha * range_j
                    high = x_max + alpha * range_j

                    new_pop[i, j] = np.random.uniform(low, high)
                    new_pop[i + 1, j] = np.random.uniform(low, high)

        new_pop = self._clip_to_bounds(new_pop, ctx.bounds)

        # 评估
        new_fitness = np.array([ctx.objective_func(ind) for ind in new_pop])

        # 更新最优
        min_idx = np.argmin(new_fitness)
        best_solution = ctx.best_solution.copy()
        best_fitness = ctx.best_fitness
        improvement = 0.0

        if new_fitness[min_idx] < best_fitness:
            improvement = best_fitness - new_fitness[min_idx]
            best_solution = new_pop[min_idx].copy()
            best_fitness = new_fitness[min_idx]

        self._update_stats(improvement, n, improvement > 0)

        return OperatorResult(
            population=new_pop,
            fitness=new_fitness,
            best_solution=best_solution,
            best_fitness=best_fitness,
            evaluations_used=n,
            success=improvement > 0,
            stats={'alpha': alpha}
        )


class DifferentialMutation(RecombinePrimitive):
    """
    差分变异
    组合多个个体的差分向量
    """

    def __init__(self, F: float = 0.8, strategy: str = "rand/1"):
        """
        Args:
            F: 缩放因子
            strategy: 变异策略 (rand/1, best/1, rand/2, best/2)
        """
        super().__init__("differential_mutation", {'F': F, 'strategy': strategy})

    def execute(self, ctx: OperatorContext) -> OperatorResult:
        pop = ctx.population.copy()
        n = len(pop)
        F = self.params['F']
        strategy = self.params['strategy']

        mutant = np.zeros_like(pop)

        for i in range(n):
            candidates = [j for j in range(n) if j != i]

            if strategy == "rand/1":
                r1, r2, r3 = np.random.choice(candidates, 3, replace=False)
                mutant[i] = pop[r1] + F * (pop[r2] - pop[r3])

            elif strategy == "best/1":
                r1, r2 = np.random.choice(candidates, 2, replace=False)
                mutant[i] = ctx.best_solution + F * (pop[r1] - pop[r2])

            elif strategy == "rand/2":
                r1, r2, r3, r4, r5 = np.random.choice(candidates, 5, replace=False)
                mutant[i] = pop[r1] + F * (pop[r2] - pop[r3]) + F * (pop[r4] - pop[r5])

            elif strategy == "best/2":
                r1, r2, r3, r4 = np.random.choice(candidates, 4, replace=False)
                mutant[i] = ctx.best_solution + F * (pop[r1] - pop[r2]) + F * (pop[r3] - pop[r4])

        mutant = self._clip_to_bounds(mutant, ctx.bounds)

        return OperatorResult(
            population=mutant,
            fitness=ctx.fitness.copy(),
            best_solution=ctx.best_solution.copy(),
            best_fitness=ctx.best_fitness,
            evaluations_used=0,
            stats={'F': F, 'strategy': strategy},
            extra={'original_pop': pop.copy(), 'mutant': mutant}
        )


# 导出所有重组类算子
__all__ = [
    'ArithmeticCrossover',
    'SBXCrossover',
    'DECrossover',
    'UniformCrossover',
    'BlendCrossover',
    'DifferentialMutation'
]
