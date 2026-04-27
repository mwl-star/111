"""
Motif模板库 - 组合primitive形成可复用的短流程
Motif是2-4个primitive组成的可复用优化流程模板
"""
from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple
import numpy as np

from .base import Primitive
from .op_types import OperatorContext, OperatorResult
from .explore import GaussianStep, LevyJump, DERand1, DEBest1, BatFrequencyMove, PSOInertiaMove, WhaleFall, RandomRestart
from .exploit import PbestPull, GbestPull, CoordinateRefine, BatLocalSearch, WolfEncircle, HillClimbing, NelderMeadStep
from .recombine import ArithmeticCrossover, SBXCrossover, DECrossover, UniformCrossover, BlendCrossover, DifferentialMutation
from .accept import GreedyAccept, MetropolisAccept, ThresholdAccept, TournamentSelection, ElitismAccept


@dataclass
class Motif:
    """
    Motif模板：2-4个primitive组成的可复用短流程

    Motif代表一个完整的优化步骤序列，如：
    - PSO风格：个体最优吸引 -> 全局最优吸引 -> 贪婪接受
    - DE风格：变异 -> 交叉 -> 贪婪接受
    - BA风格：频率移动 -> 局部精修 -> Metropolis接受
    """
    name: str                                    # Motif名称
    primitives: List[Primitive]                  # 算子序列
    description: str                             # 描述
    source_algorithm: str = ""                   # 来源算法（如PSO、BA、DE）
    tags: List[str] = field(default_factory=list) # 标签（如explore-heavy, exploit-heavy）

    def execute(self, ctx: OperatorContext) -> OperatorResult:
        """
        顺序执行motif中的primitives

        Args:
            ctx: 初始上下文

        Returns:
            最终执行结果
        """
        result = None
        current_ctx = ctx

        for primitive in self.primitives:
            result = primitive.execute(current_ctx)

            # 更新上下文，传递给下一个算子
            current_ctx = OperatorContext(
                population=result.population,
                fitness=result.fitness,
                best_solution=result.best_solution,
                best_fitness=result.best_fitness,
                bounds=current_ctx.bounds,
                objective_func=current_ctx.objective_func,
                archive=result.extra.get('archive', current_ctx.archive),
                budget_remaining=current_ctx.budget_remaining - result.evaluations_used,
                iteration=current_ctx.iteration,
                params={**current_ctx.params, **result.extra}
            )

        return result

    def get_primitive_names(self) -> List[str]:
        """获取算子名称列表"""
        return [p.name for p in self.primitives]

    def get_primitive_types(self) -> List[str]:
        """获取算子类型列表"""
        return [p.op_type.name for p in self.primitives]

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'name': self.name,
            'primitives': self.get_primitive_names(),
            'description': self.description,
            'source_algorithm': self.source_algorithm,
            'tags': self.tags
        }


# ==================== 预定义Motif库 ====================

MOTIF_LIBRARY: Dict[str, Motif] = {}


def register_motif(motif: Motif):
    """注册Motif到库"""
    MOTIF_LIBRARY[motif.name] = motif


# PSO风格Motifs
register_motif(Motif(
    name='pso_standard',
    primitives=[
        PSOInertiaMove(w=0.7),
        PbestPull(c1=1.5),
        GbestPull(c2=1.5),
        GreedyAccept()
    ],
    description='标准PSO：惯性移动 -> 个体最优吸引 -> 全局最优吸引 -> 贪婪接受',
    source_algorithm='PSO',
    tags=['balance', 'velocity-based']
))

register_motif(Motif(
    name='pso_exploit',
    primitives=[
        GbestPull(c2=2.0),
        CoordinateRefine(step_size=0.01, max_steps=10),
        GreedyAccept()
    ],
    description='PSO精修版：全局最优吸引 -> 坐标精修 -> 贪婪接受',
    source_algorithm='PSO',
    tags=['exploit-heavy', 'local-search']
))

register_motif(Motif(
    name='pso_explore',
    primitives=[
        LevyJump(scale=1.0),
        PSOInertiaMove(w=0.9),
        GbestPull(c2=1.0),
        GreedyAccept()
    ],
    description='PSO探索版：Levy跳跃 -> 高惯性移动 -> 全局吸引 -> 贪婪接受',
    source_algorithm='PSO',
    tags=['explore-heavy', 'global-search']
))


# DE风格Motifs
register_motif(Motif(
    name='de_rand1',
    primitives=[
        DERand1(F=0.8),
        DECrossover(cr=0.9),
        GreedyAccept()
    ],
    description='DE/rand/1：随机变异 -> DE交叉 -> 贪婪接受',
    source_algorithm='DE',
    tags=['balance', 'mutation-based']
))

register_motif(Motif(
    name='de_best1',
    primitives=[
        DEBest1(F=0.8),
        DECrossover(cr=0.9),
        GreedyAccept()
    ],
    description='DE/best/1：最优引导变异 -> DE交叉 -> 贪婪接受',
    source_algorithm='DE',
    tags=['exploit-heavy', 'best-guided']
))

register_motif(Motif(
    name='de_adaptive',
    primitives=[
        DifferentialMutation(F=0.5, strategy='rand/2'),
        BlendCrossover(alpha=0.5, cr=0.9),
        GreedyAccept()
    ],
    description='DE自适应：双差分变异 -> 混合交叉 -> 贪婪接受',
    source_algorithm='DE',
    tags=['adaptive', 'diversity']
))


# BA风格Motifs
register_motif(Motif(
    name='ba_standard',
    primitives=[
        BatFrequencyMove(f_min=0, f_max=2),
        BatLocalSearch(epsilon=0.1, scale_factor=0.1),
        GreedyAccept()
    ],
    description='标准BA：频率移动 -> 局部搜索 -> 贪婪接受',
    source_algorithm='BA',
    tags=['balance', 'frequency-based']
))

register_motif(Motif(
    name='ba_sa',
    primitives=[
        BatFrequencyMove(f_min=0, f_max=2),
        BatLocalSearch(epsilon=0.1),
        MetropolisAccept(temperature=1.0, cooling_rate=0.99)
    ],
    description='BA+SA：频率移动 -> 局部搜索 -> Metropolis接受',
    source_algorithm='BA',
    tags=['adaptive', 'annealing']
))


# GWO风格Motifs
register_motif(Motif(
    name='gwo_standard',
    primitives=[
        WolfEncircle(a=2.0),
        GaussianStep(step_size=0.1),
        GreedyAccept()
    ],
    description='标准GWO：狼群包围 -> 高斯扰动 -> 贪婪接受',
    source_algorithm='GWO',
    tags=['balance', 'leader-guided']
))

register_motif(Motif(
    name='gwo_exploit',
    primitives=[
        WolfEncircle(a=1.0),
        CoordinateRefine(step_size=0.01, max_steps=20),
        GreedyAccept()
    ],
    description='GWO精修版：狼群包围 -> 坐标精修 -> 贪婪接受',
    source_algorithm='GWO',
    tags=['exploit-heavy', 'local-search']
))


# WOA风格Motifs
register_motif(Motif(
    name='woa_standard',
    primitives=[
        WhaleFall(b=1.0),
        GbestPull(c2=1.5),
        GreedyAccept()
    ],
    description='标准WOA：鲸鱼下落 -> 全局吸引 -> 贪婪接受',
    source_algorithm='WOA',
    tags=['balance', 'spiral-based']
))


# 混合探索Motifs
register_motif(Motif(
    name='explore_first',
    primitives=[
        LevyJump(scale=1.5),
        GaussianStep(step_size=0.1),
        GreedyAccept()
    ],
    description='探索优先：Levy跳跃 -> 高斯扰动 -> 贪婪接受',
    source_algorithm='Hybrid',
    tags=['explore-heavy', 'global-search']
))

register_motif(Motif(
    name='explore_restart',
    primitives=[
        RandomRestart(restart_ratio=0.1),
        LevyJump(scale=1.0),
        GreedyAccept()
    ],
    description='探索重启：随机重启 -> Levy跳跃 -> 贪婪接受',
    source_algorithm='Hybrid',
    tags=['explore-heavy', 'restart']
))


# 混合利用Motifs
register_motif(Motif(
    name='exploit_refine',
    primitives=[
        GbestPull(c2=2.0),
        CoordinateRefine(step_size=0.01, max_steps=20),
        HillClimbing(neighborhood_size=0.05, max_attempts=10),
        GreedyAccept()
    ],
    description='精修优先：全局吸引 -> 坐标精修 -> 爬山搜索 -> 贪婪接受',
    source_algorithm='Hybrid',
    tags=['exploit-heavy', 'multi-refine']
))

register_motif(Motif(
    name='exploit_nelder',
    primitives=[
        NelderMeadStep(),
        CoordinateRefine(step_size=0.01, max_steps=10),
        GreedyAccept()
    ],
    description='单纯形精修：Nelder-Mead步骤 -> 坐标精修 -> 贪婪接受',
    source_algorithm='Hybrid',
    tags=['exploit-heavy', 'simplex']
))


# 平衡型Motifs
register_motif(Motif(
    name='balance_de_pso',
    primitives=[
        DERand1(F=0.8),
        ArithmeticCrossover(cr=0.9),
        GbestPull(c2=1.5),
        GreedyAccept()
    ],
    description='DE+PSO混合：DE变异 -> 算术交叉 -> 全局吸引 -> 贪婪接受',
    source_algorithm='Hybrid',
    tags=['balance', 'hybrid']
))

register_motif(Motif(
    name='balance_ba_de',
    primitives=[
        BatFrequencyMove(f_min=0, f_max=2),
        DECrossover(cr=0.9),
        GreedyAccept()
    ],
    description='BA+DE混合：频率移动 -> DE交叉 -> 贪婪接受',
    source_algorithm='Hybrid',
    tags=['balance', 'hybrid']
))


# 自适应Motifs
register_motif(Motif(
    name='adaptive_sa',
    primitives=[
        GaussianStep(step_size=0.1, adaptive=True),
        CoordinateRefine(step_size=0.01, max_steps=10),
        MetropolisAccept(temperature=1.0, cooling_rate=0.99)
    ],
    description='自适应SA：自适应高斯扰动 -> 坐标精修 -> Metropolis接受',
    source_algorithm='Hybrid',
    tags=['adaptive', 'annealing']
))

register_motif(Motif(
    name='adaptive_threshold',
    primitives=[
        GaussianStep(step_size=0.1),
        GbestPull(c2=1.5),
        ThresholdAccept(threshold=0.01, threshold_decay=0.95)
    ],
    description='阈值接受：高斯扰动 -> 全局吸引 -> 阈值接受',
    source_algorithm='Hybrid',
    tags=['adaptive', 'threshold']
))


# 精英保留Motifs
register_motif(Motif(
    name='elitism_standard',
    primitives=[
        DERand1(F=0.8),
        DECrossover(cr=0.9),
        ElitismAccept(elite_size=1)
    ],
    description='精英DE：DE变异 -> DE交叉 -> 精英保留接受',
    source_algorithm='DE',
    tags=['elitism', 'best-preserve']
))


def get_motif(name: str) -> Motif:
    """获取指定名称的Motif"""
    return MOTIF_LIBRARY.get(name)


def get_all_motifs() -> Dict[str, Motif]:
    """获取所有Motif"""
    return MOTIF_LIBRARY.copy()


def get_motif_names() -> List[str]:
    """获取所有Motif名称"""
    return list(MOTIF_LIBRARY.keys())


def get_motifs_by_tag(tag: str) -> List[Motif]:
    """按标签筛选Motif"""
    return [m for m in MOTIF_LIBRARY.values() if tag in m.tags]


def get_motifs_by_algorithm(algorithm: str) -> List[Motif]:
    """按来源算法筛选Motif"""
    return [m for m in MOTIF_LIBRARY.values() if m.source_algorithm == algorithm]


def create_custom_motif(name: str, primitives: List[Primitive], description: str = "") -> Motif:
    """创建自定义Motif"""
    motif = Motif(
        name=name,
        primitives=primitives,
        description=description,
        source_algorithm='Custom',
        tags=['custom']
    )
    register_motif(motif)
    return motif


# 导出
__all__ = [
    'Motif',
    'MOTIF_LIBRARY',
    'register_motif',
    'get_motif',
    'get_all_motifs',
    'get_motif_names',
    'get_motifs_by_tag',
    'get_motifs_by_algorithm',
    'create_custom_motif'
]