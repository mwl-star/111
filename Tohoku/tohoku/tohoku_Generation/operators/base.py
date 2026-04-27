"""
Primitive算子基类 - 所有细粒度算子的抽象基类
"""
from abc import ABC, abstractmethod
from typing import Dict, Any
import numpy as np
from .op_types import OperatorType, OperatorContext, OperatorResult, OperatorStats


class Primitive(ABC):
    """
    Primitive算子基类

    每个Primitive代表一个原子操作，可以是从现有算法中拆解出的基本步骤
    如：高斯扰动、Levy跳跃、个体最优吸引、全局最优吸引等
    """

    def __init__(self, name: str, op_type: OperatorType, params: Dict[str, Any] = None):
        """
        初始化算子

        Args:
            name: 算子名称
            op_type: 算子类型 (INIT/EXPLORE/EXPLOIT/RECOMBINE/ACCEPT/CONTROL)
            params: 算子参数
        """
        self.name = name
        self.op_type = op_type
        self.params = params or {}
        self.stats = OperatorStats(name=name, op_type=op_type)

    @abstractmethod
    def execute(self, ctx: OperatorContext) -> OperatorResult:
        """
        执行算子

        Args:
            ctx: 算子执行上下文

        Returns:
            OperatorResult: 执行结果
        """
        pass

    def _clip_to_bounds(self, population: np.ndarray, bounds: tuple) -> np.ndarray:
        """将种群限制在边界内"""
        lb, ub = bounds
        return np.clip(population, lb, ub)

    def _update_stats(self, improvement: float, evaluations: int, success: bool):
        """更新统计信息"""
        self.stats.call_count += 1
        self.stats.total_evaluations += evaluations
        if success:
            self.stats.success_count += 1
            self.stats.total_improvement += improvement

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return self.stats.to_dict()

    def reset_stats(self):
        """重置统计信息"""
        self.stats = OperatorStats(name=self.name, op_type=self.op_type)

    def set_params(self, params: Dict[str, Any]):
        """设置参数"""
        self.params.update(params)

    def __repr__(self) -> str:
        return f"Primitive(name='{self.name}', type={self.op_type.name}, params={self.params})"


class InitPrimitive(Primitive):
    """初始化类算子基类"""

    def __init__(self, name: str, params: Dict[str, Any] = None):
        super().__init__(name, OperatorType.INIT, params)


class ExplorePrimitive(Primitive):
    """探索类算子基类 - 用于全局探索"""

    def __init__(self, name: str, params: Dict[str, Any] = None):
        super().__init__(name, OperatorType.EXPLORE, params)


class ExploitPrimitive(Primitive):
    """利用类算子基类 - 用于局部开发"""

    def __init__(self, name: str, params: Dict[str, Any] = None):
        super().__init__(name, OperatorType.EXPLOIT, params)


class RecombinePrimitive(Primitive):
    """重组类算子基类 - 用于解的融合"""

    def __init__(self, name: str, params: Dict[str, Any] = None):
        super().__init__(name, OperatorType.RECOMBINE, params)


class AcceptPrimitive(Primitive):
    """接受类算子基类 - 用于解的接受/拒绝"""

    def __init__(self, name: str, params: Dict[str, Any] = None):
        super().__init__(name, OperatorType.ACCEPT, params)


class ControlPrimitive(Primitive):
    """控制类算子基类 - 用于流程控制"""

    def __init__(self, name: str, params: Dict[str, Any] = None):
        super().__init__(name, OperatorType.CONTROL, params)