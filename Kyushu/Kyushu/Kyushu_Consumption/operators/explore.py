"""
探索类算子 - 用于全局探索的原子操作
包含：高斯扰动、Levy跳跃、DE变异、蝙蝠频率移动、PSO惯性移动等
"""
import numpy as np
import math
from typing import Dict, Any
from .base import ExplorePrimitive
from .op_types import OperatorContext, OperatorResult


class GaussianStep(ExplorePrimitive):
    """
    高斯步长探索
    对每个个体添加高斯分布的随机扰动
    """

    def __init__(self, step_size: float = 0.1, adaptive: bool = False):
        """
        Args:
            step_size: 步长大小（相对于搜索空间）
            adaptive: 是否自适应调整步长
        """
        super().__init__("gaussian_step", {
            'step_size': step_size,
            'adaptive': adaptive
        })

    def execute(self, ctx: OperatorContext) -> OperatorResult:
        new_pop = ctx.population.copy()
        lb, ub = ctx.bounds
        search_range = ub - lb

        # 计算实际步长
        if self.params['adaptive']:
            # 自适应：根据多样性调整步长
            diversity = ctx.get_diversity()
            step_size = self.params['step_size'] * (1 + diversity)
        else:
            step_size = self.params['step_size']

        # 高斯扰动
        for i in range(len(new_pop)):
            step = np.random.normal(0, step_size * search_range, new_pop.shape[1])
            new_pop[i] = new_pop[i] + step

        # 边界处理
        new_pop = self._clip_to_bounds(new_pop, ctx.bounds)

        # 评估新解
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

        # 更新统计
        self._update_stats(improvement, len(new_pop), improvement > 0)

        return OperatorResult(
            population=new_pop,
            fitness=new_fitness,
            best_solution=best_solution,
            best_fitness=best_fitness,
            evaluations_used=len(new_pop),
            success=improvement > 0,
            stats={'step_size': step_size},
            extra={'original_pop': ctx.population.copy()}
        )


class LevyJump(ExplorePrimitive):
    """
    Levy飞行跳跃
    使用Levy分布生成跳跃步长，适合大范围探索
    """

    def __init__(self, scale: float = 1.0, beta: float = 1.5):
        """
        Args:
            scale: 缩放因子
            beta: Levy指数 (1 < beta <= 3)
        """
        super().__init__("levy_jump", {'scale': scale, 'beta': beta})

    def _levy_flight(self, dim: int) -> np.ndarray:
        """生成Levy飞行步长"""
        beta = self.params['beta']
        # 计算 sigma
        numerator = math.gamma(1 + beta) * np.sin(np.pi * beta / 2)
        denominator = math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2)
        sigma = (numerator / denominator) ** (1 / beta)

        u = np.random.normal(0, sigma, dim)
        v = np.random.normal(0, 1, dim)

        step = u / (np.abs(v) ** (1 / beta))
        return step * self.params['scale']

    def execute(self, ctx: OperatorContext) -> OperatorResult:
        new_pop = ctx.population.copy()
        lb, ub = ctx.bounds
        search_range = ub - lb

        for i in range(len(new_pop)):
            step = self._levy_flight(new_pop.shape[1])
            new_pop[i] = new_pop[i] + step * search_range

        # 边界处理
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

        self._update_stats(improvement, len(new_pop), improvement > 0)

        return OperatorResult(
            population=new_pop,
            fitness=new_fitness,
            best_solution=best_solution,
            best_fitness=best_fitness,
            evaluations_used=len(new_pop),
            success=improvement > 0,
            stats={'scale': self.params['scale']}
        )


class DERand1(ExplorePrimitive):
    """
    DE/rand/1 变异
    差分进化经典变异策略: v = x_r1 + F * (x_r2 - x_r3)
    """

    def __init__(self, F: float = 0.8):
        """
        Args:
            F: 缩放因子 (通常 0.4-1.0)
        """
        super().__init__("de_rand_1", {'F': F})

    def execute(self, ctx: OperatorContext) -> OperatorResult:
        pop = ctx.population
        n = len(pop)
        dim = pop.shape[1]

        mutant = np.zeros_like(pop)
        for i in range(n):
            # 随机选择3个不同的个体
            candidates = [j for j in range(n) if j != i]
            idxs = np.random.choice(candidates, 3, replace=False)
            r1, r2, r3 = idxs

            # DE/rand/1: v_i = x_r1 + F * (x_r2 - x_r3)
            F = self.params['F']
            mutant[i] = pop[r1] + F * (pop[r2] - pop[r3])

        # 边界处理
        mutant = self._clip_to_bounds(mutant, ctx.bounds)

        return OperatorResult(
            population=mutant,
            fitness=ctx.fitness.copy(),  # 变异后需要交叉和评估
            best_solution=ctx.best_solution.copy(),
            best_fitness=ctx.best_fitness,
            evaluations_used=0,  # 变异本身不消耗评估
            stats={'F': self.params['F']},
            extra={'original_pop': pop.copy(), 'mutant': mutant}
        )


class DEBest1(ExplorePrimitive):
    """
    DE/best/1 变异
    使用最优解引导: v = x_best + F * (x_r1 - x_r2)
    """

    def __init__(self, F: float = 0.8):
        super().__init__("de_best_1", {'F': F})

    def execute(self, ctx: OperatorContext) -> OperatorResult:
        pop = ctx.population
        n = len(pop)

        mutant = np.zeros_like(pop)
        for i in range(n):
            # 随机选择2个不同的个体
            candidates = [j for j in range(n) if j != i]
            idxs = np.random.choice(candidates, 2, replace=False)
            r1, r2 = idxs

            # DE/best/1: v_i = x_best + F * (x_r1 - x_r2)
            F = self.params['F']
            mutant[i] = ctx.best_solution + F * (pop[r1] - pop[r2])

        mutant = self._clip_to_bounds(mutant, ctx.bounds)

        return OperatorResult(
            population=mutant,
            fitness=ctx.fitness.copy(),
            best_solution=ctx.best_solution.copy(),
            best_fitness=ctx.best_fitness,
            evaluations_used=0,
            stats={'F': self.params['F']},
            extra={'original_pop': pop.copy(), 'mutant': mutant}
        )


class BatFrequencyMove(ExplorePrimitive):
    """
    蝙蝠算法频率移动
    基于频率调整速度和位置
    """

    def __init__(self, f_min: float = 0.0, f_max: float = 2.0):
        """
        Args:
            f_min: 最小频率
            f_max: 最大频率
        """
        super().__init__("bat_frequency_move", {'f_min': f_min, 'f_max': f_max})

    def execute(self, ctx: OperatorContext) -> OperatorResult:
        pop = ctx.population.copy()
        n = len(pop)

        # 获取或初始化速度
        velocity = ctx.params.get('velocity', np.zeros_like(pop))

        f_min = self.params['f_min']
        f_max = self.params['f_max']

        for i in range(n):
            # 频率
            beta = np.random.rand()
            f = f_min + (f_max - f_min) * beta

            # 速度更新
            velocity[i] = velocity[i] + (pop[i] - ctx.best_solution) * f

            # 位置更新
            pop[i] = pop[i] + velocity[i]

        pop = self._clip_to_bounds(pop, ctx.bounds)

        return OperatorResult(
            population=pop,
            fitness=ctx.fitness.copy(),
            best_solution=ctx.best_solution.copy(),
            best_fitness=ctx.best_fitness,
            evaluations_used=0,
            stats={'avg_frequency': (f_min + f_max) / 2},
            extra={'velocity': velocity}
        )


class PSOInertiaMove(ExplorePrimitive):
    """
    PSO惯性移动
    基于惯性权重的速度更新
    """

    def __init__(self, w: float = 0.7):
        """
        Args:
            w: 惯性权重 (通常 0.4-0.9)
        """
        super().__init__("pso_inertia_move", {'w': w})

    def execute(self, ctx: OperatorContext) -> OperatorResult:
        pop = ctx.population.copy()

        # 获取或初始化速度
        velocity = ctx.params.get('velocity', np.zeros_like(pop))

        w = self.params['w']

        # 惯性移动
        velocity = w * velocity
        pop = pop + velocity

        pop = self._clip_to_bounds(pop, ctx.bounds)

        return OperatorResult(
            population=pop,
            fitness=ctx.fitness.copy(),
            best_solution=ctx.best_solution.copy(),
            best_fitness=ctx.best_fitness,
            evaluations_used=0,
            stats={'w': w},
            extra={'velocity': velocity}
        )


class WhaleFall(ExplorePrimitive):
    """
    鲸鱼优化算法下落
    模拟座头鲸的螺旋下落捕食行为
    """

    def __init__(self, b: float = 1.0):
        """
        Args:
            b: 螺旋形状参数
        """
        super().__init__("whale_fall", {'b': b})

    def execute(self, ctx: OperatorContext) -> OperatorResult:
        pop = ctx.population.copy()
        n = len(pop)
        dim = pop.shape[1]

        b = self.params['b']

        for i in range(n):
            # 螺旋下落
            l = np.random.uniform(-1, 1)
            r = np.random.rand()

            # 距离最优解的距离
            D = np.abs(ctx.best_solution - pop[i])

            # 螺旋位置更新
            pop[i] = D * np.exp(b * l) * np.cos(2 * np.pi * l) + ctx.best_solution

        pop = self._clip_to_bounds(pop, ctx.bounds)

        return OperatorResult(
            population=pop,
            fitness=ctx.fitness.copy(),
            best_solution=ctx.best_solution.copy(),
            best_fitness=ctx.best_fitness,
            evaluations_used=0,
            stats={'b': b}
        )


class RandomRestart(ExplorePrimitive):
    """
    随机重启
    对部分个体进行随机重新初始化
    """

    def __init__(self, restart_ratio: float = 0.1):
        """
        Args:
            restart_ratio: 重启比例 (0-1)
        """
        super().__init__("random_restart", {'restart_ratio': restart_ratio})

    def execute(self, ctx: OperatorContext) -> OperatorResult:
        new_pop = ctx.population.copy()
        n = len(new_pop)
        lb, ub = ctx.bounds

        # 选择要重启的个体
        restart_count = max(1, int(n * self.params['restart_ratio']))
        worst_indices = np.argsort(ctx.fitness)[-restart_count:]

        # 随机重新初始化
        for idx in worst_indices:
            new_pop[idx] = np.random.uniform(lb, ub, new_pop.shape[1])

        # 评估重启的个体
        new_fitness = ctx.fitness.copy()
        for idx in worst_indices:
            new_fitness[idx] = ctx.objective_func(new_pop[idx])

        # 更新最优
        min_idx = np.argmin(new_fitness)
        best_solution = ctx.best_solution.copy()
        best_fitness = ctx.best_fitness
        improvement = 0.0

        if new_fitness[min_idx] < best_fitness:
            improvement = best_fitness - new_fitness[min_idx]
            best_solution = new_pop[min_idx].copy()
            best_fitness = new_fitness[min_idx]

        self._update_stats(improvement, restart_count, improvement > 0)

        return OperatorResult(
            population=new_pop,
            fitness=new_fitness,
            best_solution=best_solution,
            best_fitness=best_fitness,
            evaluations_used=restart_count,
            success=improvement > 0,
            stats={'restart_count': restart_count}
        )


# 导出所有探索类算子
__all__ = [
    'GaussianStep',
    'LevyJump',
    'DERand1',
    'DEBest1',
    'BatFrequencyMove',
    'PSOInertiaMove',
    'WhaleFall',
    'RandomRestart'
]
