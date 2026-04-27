"""
细粒度算子库 - 第二阶段核心模块

将PSO/BA/DE/GWO/WOA等算法拆解为可重组的原子操作(Primitive)，
支持通过Motif模板组合，并通过Typed-DAG进行动态组装。

主要组件：
- types: 算子类型系统
- base: Primitive基类
- explore: 探索类算子
- exploit: 利用类算子
- recombine: 重组类算子
- accept: 接受类算子
- motif: Motif模板库
- graph: Typed-DAG组装器
"""

from .op_types import (
    OperatorType,
    OperatorContext,
    OperatorResult,
    OperatorStats
)

from .base import (
    Primitive,
    InitPrimitive,
    ExplorePrimitive,
    ExploitPrimitive,
    RecombinePrimitive,
    AcceptPrimitive,
    ControlPrimitive
)

from .explore import (
    GaussianStep,
    LevyJump,
    DERand1,
    DEBest1,
    BatFrequencyMove,
    PSOInertiaMove,
    WhaleFall,
    RandomRestart
)

from .exploit import (
    PbestPull,
    GbestPull,
    CoordinateRefine,
    BatLocalSearch,
    WolfEncircle,
    HillClimbing,
    NelderMeadStep
)

from .recombine import (
    ArithmeticCrossover,
    SBXCrossover,
    DECrossover,
    UniformCrossover,
    BlendCrossover,
    DifferentialMutation
)

from .accept import (
    GreedyAccept,
    MetropolisAccept,
    ThresholdAccept,
    TournamentSelection,
    ElitismAccept,
    ProbabilisticAccept,
    ArchiveUpdate
)

from .motif import (
    Motif,
    MOTIF_LIBRARY,
    register_motif,
    get_motif,
    get_all_motifs,
    get_motif_names,
    get_motifs_by_tag,
    get_motifs_by_algorithm,
    create_custom_motif
)

from .graph import (
    NodeType,
    OperatorNode,
    OperatorGraph,
    GraphBuilder,
    repair_graph
)


# 算子统计
def get_operator_stats() -> dict:
    """获取算子库统计信息"""
    return {
        'explore_operators': 8,
        'exploit_operators': 7,
        'recombine_operators': 6,
        'accept_operators': 7,
        'total_primitives': 28,
        'total_motifs': len(MOTIF_LIBRARY)
    }


__all__ = [
    # Types
    'OperatorType',
    'OperatorContext',
    'OperatorResult',
    'OperatorStats',

    # Base
    'Primitive',
    'InitPrimitive',
    'ExplorePrimitive',
    'ExploitPrimitive',
    'RecombinePrimitive',
    'AcceptPrimitive',
    'ControlPrimitive',

    # Explore
    'GaussianStep',
    'LevyJump',
    'DERand1',
    'DEBest1',
    'BatFrequencyMove',
    'PSOInertiaMove',
    'WhaleFall',
    'RandomRestart',

    # Exploit
    'PbestPull',
    'GbestPull',
    'CoordinateRefine',
    'BatLocalSearch',
    'WolfEncircle',
    'HillClimbing',
    'NelderMeadStep',

    # Recombine
    'ArithmeticCrossover',
    'SBXCrossover',
    'DECrossover',
    'UniformCrossover',
    'BlendCrossover',
    'DifferentialMutation',

    # Accept
    'GreedyAccept',
    'MetropolisAccept',
    'ThresholdAccept',
    'TournamentSelection',
    'ElitismAccept',
    'ProbabilisticAccept',
    'ArchiveUpdate',

    # Motif
    'Motif',
    'MOTIF_LIBRARY',
    'register_motif',
    'get_motif',
    'get_all_motifs',
    'get_motif_names',
    'get_motifs_by_tag',
    'get_motifs_by_algorithm',
    'create_custom_motif',

    # Graph
    'NodeType',
    'OperatorNode',
    'OperatorGraph',
    'GraphBuilder',
    'repair_graph',

    # Stats
    'get_operator_stats'
]
