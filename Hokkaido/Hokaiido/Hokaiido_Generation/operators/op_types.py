"""
算子类型系统 - 定义细粒度算子的类型和上下文
"""
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Callable
import numpy as np


class OperatorType(Enum):
    """算子类型枚举"""
    INIT = auto()        # 初始化/重启
    EXPLORE = auto()     # 探索移动
    EXPLOIT = auto()     # 利用移动
    RECOMBINE = auto()   # 重组/融合
    ACCEPT = auto()      # 接受/存档
    CONTROL = auto()     # 调度/终止


@dataclass
class OperatorContext:
    """算子执行上下文 - 包含执行算子所需的所有信息"""
    population: np.ndarray              # 当前种群 (pop_size, dim)
    fitness: np.ndarray                 # 适应度 (pop_size,)
    best_solution: np.ndarray           # 全局最优解 (dim,)
    best_fitness: float                 # 全局最优适应度
    bounds: tuple                       # 搜索边界 (lower, upper)
    objective_func: Callable            # 目标函数
    archive: List[np.ndarray] = field(default_factory=list)  # 历史最优解存档
    budget_remaining: int = 10000       # 剩余评估预算
    iteration: int = 0                  # 当前迭代次数
    params: Dict[str, Any] = field(default_factory=dict)  # 额外参数

    def get_diversity(self) -> float:
        """计算种群多样性"""
        if self.population is None or len(self.population) < 2:
            return 0.0
        return float(np.std(self.population))


@dataclass
class OperatorResult:
    """算子执行结果"""
    population: np.ndarray              # 更新后种群
    fitness: np.ndarray                 # 更新后适应度
    best_solution: np.ndarray           # 更新后全局最优
    best_fitness: float                 # 更新后全局最优适应度
    evaluations_used: int               # 消耗的评估次数
    success: bool = True                # 是否成功改进
    stats: Dict[str, Any] = field(default_factory=dict)  # 统计信息
    extra: Dict[str, Any] = field(default_factory=dict)  # 额外输出（如速度）


@dataclass
class OperatorStats:
    """算子统计信息"""
    name: str
    op_type: OperatorType
    call_count: int = 0
    success_count: int = 0
    total_improvement: float = 0.0
    total_evaluations: int = 0

    @property
    def success_rate(self) -> float:
        """成功率"""
        return self.success_count / max(self.call_count, 1)

    @property
    def avg_improvement(self) -> float:
        """平均改进量"""
        return self.total_improvement / max(self.success_count, 1)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'name': self.name,
            'type': self.op_type.name,
            'call_count': self.call_count,
            'success_count': self.success_count,
            'success_rate': self.success_rate,
            'avg_improvement': self.avg_improvement,
            'total_evaluations': self.total_evaluations
        }
