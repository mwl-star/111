"""
Typed-DAG组装器 - 支持算子图的在线组装与合法性检查
实现类型化的有向无环图，用于动态组合算子
"""
from dataclasses import dataclass, field
from typing import List, Dict, Set, Optional, Tuple, Any
from enum import Enum
import numpy as np

from .base import Primitive
from .op_types import OperatorType, OperatorContext, OperatorResult
from .motif import Motif


class NodeType(Enum):
    """节点类型"""
    SOURCE = 'source'      # 输入节点
    SINK = 'sink'          # 输出节点
    OPERATOR = 'operator'  # 算子节点


@dataclass
class OperatorNode:
    """算子图节点"""
    id: int
    node_type: NodeType
    operator: Optional[Primitive] = None
    successors: List[int] = field(default_factory=list)
    predecessors: List[int] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'id': self.id,
            'type': self.node_type.value,
            'operator': self.operator.name if self.operator else None,
            'operator_type': self.operator.op_type.name if self.operator else None,
            'successors': self.successors,
            'predecessors': self.predecessors
        }


class OperatorGraph:
    """
    类型化有向算子图

    支持动态组装算子，并进行合法性检查：
    1. 从source到sink必须可达
    2. 探索/利用节点后必须接接受节点
    3. 至少有一个移动节点
    4. 不能有环
    """

    def __init__(self):
        self.nodes: Dict[int, OperatorNode] = {}
        self.node_counter = 0
        self._add_source_sink()

    def _add_source_sink(self):
        """添加源节点和汇节点"""
        source = OperatorNode(id=0, node_type=NodeType.SOURCE)
        sink = OperatorNode(id=1, node_type=NodeType.SINK)
        self.nodes[0] = source
        self.nodes[1] = sink
        self.node_counter = 2

    def add_node(self, operator: Primitive) -> int:
        """
        添加算子节点

        Args:
            operator: 算子实例

        Returns:
            节点ID
        """
        node = OperatorNode(
            id=self.node_counter,
            node_type=NodeType.OPERATOR,
            operator=operator
        )
        self.nodes[self.node_counter] = node
        self.node_counter += 1
        return node.id

    def add_edge(self, from_id: int, to_id: int) -> bool:
        """
        添加边

        Args:
            from_id: 起始节点ID
            to_id: 目标节点ID

        Returns:
            是否成功
        """
        if from_id not in self.nodes or to_id not in self.nodes:
            return False

        # 检查是否会形成环
        if self._would_create_cycle(from_id, to_id):
            return False

        self.nodes[from_id].successors.append(to_id)
        self.nodes[to_id].predecessors.append(from_id)
        return True

    def remove_node(self, node_id: int) -> bool:
        """删除节点"""
        if node_id not in self.nodes:
            return False
        if node_id in [0, 1]:  # 不能删除source和sink
            return False

        # 删除相关边
        node = self.nodes[node_id]
        for pred_id in node.predecessors:
            if pred_id in self.nodes:
                self.nodes[pred_id].successors.remove(node_id)
        for succ_id in node.successors:
            if succ_id in self.nodes:
                self.nodes[succ_id].predecessors.remove(node_id)

        del self.nodes[node_id]
        return True

    def remove_edge(self, from_id: int, to_id: int) -> bool:
        """删除边"""
        if from_id not in self.nodes or to_id not in self.nodes:
            return False

        if to_id in self.nodes[from_id].successors:
            self.nodes[from_id].successors.remove(to_id)
        if from_id in self.nodes[to_id].predecessors:
            self.nodes[to_id].predecessors.remove(from_id)
        return True

    def _would_create_cycle(self, from_id: int, to_id: int) -> bool:
        """检查添加边是否会形成环"""
        # 如果to能到达from，则添加from->to会形成环
        visited = set()
        queue = [to_id]

        while queue:
            current = queue.pop(0)
            if current == from_id:
                return True
            if current in visited:
                continue
            visited.add(current)

            if current in self.nodes:
                queue.extend(self.nodes[current].successors)

        return False

    def _is_reachable(self, from_id: int, to_id: int) -> bool:
        """BFS检查可达性"""
        visited = set()
        queue = [from_id]

        while queue:
            current = queue.pop(0)
            if current == to_id:
                return True
            if current in visited:
                continue
            visited.add(current)

            if current in self.nodes:
                queue.extend(self.nodes[current].successors)

        return False

    def is_legal(self) -> Tuple[bool, List[str]]:
        """
        检查图合法性

        Returns:
            (是否合法, 错误信息列表)
        """
        errors = []

        # 1. 检查从source到sink是否可达
        if not self._is_reachable(0, 1):
            errors.append("Graph is not reachable from source to sink")

        # 2. 检查类型匹配
        for node_id, node in self.nodes.items():
            if node.node_type == NodeType.OPERATOR:
                op_type = node.operator.op_type

                # 探索/利用节点后必须接接受节点或sink
                if op_type in [OperatorType.EXPLORE, OperatorType.EXPLOIT]:
                    has_accept = False
                    for succ_id in node.successors:
                        succ = self.nodes[succ_id]
                        if succ.node_type == NodeType.SINK:
                            has_accept = True
                            break
                        if succ.operator.op_type in [OperatorType.ACCEPT, OperatorType.CONTROL]:
                            has_accept = True
                            break
                    if not has_accept:
                        errors.append(f"Node {node_id} ({node.operator.name}) must be followed by accept/control node")

        # 3. 检查至少有一个移动节点
        has_move = False
        for node in self.nodes.values():
            if node.node_type == NodeType.OPERATOR:
                if node.operator.op_type in [OperatorType.EXPLORE, OperatorType.EXPLOIT]:
                    has_move = True
                    break
        if not has_move:
            errors.append("Graph must have at least one explore/exploit node")

        # 4. 检查重组节点后有接受节点
        for node_id, node in self.nodes.items():
            if node.node_type == NodeType.OPERATOR:
                if node.operator.op_type == OperatorType.RECOMBINE:
                    has_accept = False
                    for succ_id in node.successors:
                        succ = self.nodes[succ_id]
                        if succ.node_type == NodeType.SINK:
                            has_accept = True
                            break
                        if succ.operator.op_type in [OperatorType.ACCEPT, OperatorType.EXPLOIT]:
                            has_accept = True
                            break
                    if not has_accept:
                        errors.append(f"Node {node_id} ({node.operator.name}) must be followed by accept/exploit node")

        return len(errors) == 0, errors

    def get_execution_order(self) -> List[int]:
        """
        获取拓扑排序的执行顺序

        Returns:
            节点ID列表（按执行顺序）
        """
        in_degree = {nid: len(node.predecessors) for nid, node in self.nodes.items()}
        queue = [nid for nid, deg in in_degree.items() if deg == 0]
        order = []

        while queue:
            current = queue.pop(0)
            if self.nodes[current].node_type == NodeType.OPERATOR:
                order.append(current)

            for succ_id in self.nodes[current].successors:
                in_degree[succ_id] -= 1
                if in_degree[succ_id] == 0:
                    queue.append(succ_id)

        return order

    def execute(self, ctx: OperatorContext) -> OperatorResult:
        """
        执行算子图

        Args:
            ctx: 初始上下文

        Returns:
            最终执行结果
        """
        order = self.get_execution_order()
        result = None
        current_ctx = ctx

        for node_id in order:
            node = self.nodes[node_id]
            result = node.operator.execute(current_ctx)

            # 更新上下文
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

    def to_dict(self) -> Dict[str, Any]:
        """序列化为字典"""
        return {
            'nodes': {nid: node.to_dict() for nid, node in self.nodes.items()},
            'execution_order': self.get_execution_order()
        }

    @classmethod
    def from_motif(cls, motif: Motif) -> 'OperatorGraph':
        """
        从Motif构建图

        Args:
            motif: Motif实例

        Returns:
            OperatorGraph实例
        """
        graph = cls()

        prev_id = 0  # source
        for primitive in motif.primitives:
            node_id = graph.add_node(primitive)
            graph.add_edge(prev_id, node_id)
            prev_id = node_id

        graph.add_edge(prev_id, 1)  # to sink
        return graph

    def get_stats(self) -> Dict[str, Any]:
        """获取图统计信息"""
        operator_nodes = [n for n in self.nodes.values() if n.node_type == NodeType.OPERATOR]

        type_counts = {}
        for node in operator_nodes:
            op_type = node.operator.op_type.name
            type_counts[op_type] = type_counts.get(op_type, 0) + 1

        return {
            'total_nodes': len(operator_nodes),
            'total_edges': sum(len(n.successors) for n in self.nodes.values()),
            'type_counts': type_counts,
            'is_legal': self.is_legal()[0]
        }


class GraphBuilder:
    """
    图构建器 - 提供流畅的API构建算子图
    """

    def __init__(self):
        self.graph = OperatorGraph()
        self._last_node_id = 0  # source

    def add(self, operator: Primitive) -> 'GraphBuilder':
        """添加算子节点并自动连接"""
        node_id = self.graph.add_node(operator)
        self.graph.add_edge(self._last_node_id, node_id)
        self._last_node_id = node_id
        return self

    def branch(self, operators: List[Primitive]) -> 'GraphBuilder':
        """分支：添加多个并行算子"""
        branch_ids = []
        for op in operators:
            node_id = self.graph.add_node(op)
            self.graph.add_edge(self._last_node_id, node_id)
            branch_ids.append(node_id)

        # 创建汇合节点（使用第一个分支的后续）
        self._last_node_id = branch_ids[0]
        return self

    def merge(self) -> 'GraphBuilder':
        """汇合：连接到sink"""
        self.graph.add_edge(self._last_node_id, 1)
        return self

    def build(self) -> OperatorGraph:
        """构建并返回图"""
        # 确保连接到sink
        if not self.graph._is_reachable(self._last_node_id, 1):
            self.graph.add_edge(self._last_node_id, 1)
        return self.graph


def repair_graph(graph: OperatorGraph) -> OperatorGraph:
    """
    修复非法图

    Args:
        graph: 原始图

    Returns:
        修复后的图
    """
    is_legal, errors = graph.is_legal()

    if is_legal:
        return graph

    # 尝试修复
    repaired = OperatorGraph()

    # 复制所有算子节点
    node_mapping = {0: 0, 1: 1}  # source和sink映射

    for nid, node in graph.nodes.items():
        if node.node_type == NodeType.OPERATOR:
            new_id = repaired.add_node(node.operator)
            node_mapping[nid] = new_id

    # 复制边
    for nid, node in graph.nodes.items():
        if node.node_type == NodeType.OPERATOR:
            for succ_id in node.successors:
                if succ_id in node_mapping:
                    repaired.add_edge(node_mapping[nid], node_mapping[succ_id])

    # 确保合法
    is_legal, errors = repaired.is_legal()
    if not is_legal:
        # 如果仍然非法，添加默认的接受节点
        from .accept import GreedyAccept
        accept_id = repaired.add_node(GreedyAccept())
        # 找到没有接受节点的探索/利用节点
        for nid, node in repaired.nodes.items():
            if node.node_type == NodeType.OPERATOR:
                if node.operator.op_type in [OperatorType.EXPLORE, OperatorType.EXPLOIT]:
                    if not any(repaired.nodes[s].operator.op_type == OperatorType.ACCEPT
                              for s in node.successors if repaired.nodes[s].node_type == NodeType.OPERATOR):
                        repaired.add_edge(nid, accept_id)
        repaired.add_edge(accept_id, 1)

    return repaired


# 导出
__all__ = [
    'NodeType',
    'OperatorNode',
    'OperatorGraph',
    'GraphBuilder',
    'repair_graph'
]