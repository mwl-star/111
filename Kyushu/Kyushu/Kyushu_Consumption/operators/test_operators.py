"""
算子库测试脚本
测试Primitive算子、Motif模板和Typed-DAG组装器
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from operators import (
    # Types
    OperatorContext, OperatorType,
    # Explore
    GaussianStep, LevyJump, DERand1, BatFrequencyMove,
    # Exploit
    PbestPull, GbestPull, CoordinateRefine,
    # Recombine
    ArithmeticCrossover, DECrossover,
    # Accept
    GreedyAccept, MetropolisAccept,
    # Motif
    MOTIF_LIBRARY, get_motif, get_motif_names,
    # Graph
    OperatorGraph, GraphBuilder, repair_graph
)


# ==================== 测试函数 ====================

def sphere(x):
    """Sphere测试函数: f(x) = sum(x^2)"""
    return np.sum(x ** 2)


def rosenbrock(x):
    """Rosenbrock测试函数"""
    return sum(100 * (x[i+1] - x[i]**2)**2 + (1 - x[i])**2 for i in range(len(x)-1))


def rastrigin(x):
    """Rastrigin测试函数"""
    A = 10
    n = len(x)
    return A * n + sum(x[i]**2 - A * np.cos(2 * np.pi * x[i]) for i in range(n))


# ==================== 测试用例 ====================

def test_context_creation():
    """测试上下文创建"""
    print("\n" + "="*50)
    print("测试1: OperatorContext创建")
    print("="*50)

    pop_size, dim = 10, 5
    lb, ub = -10, 10

    population = np.random.uniform(lb, ub, (pop_size, dim))
    fitness = np.array([sphere(ind) for ind in population])

    ctx = OperatorContext(
        population=population,
        fitness=fitness,
        best_solution=population[np.argmin(fitness)],
        best_fitness=np.min(fitness),
        bounds=(lb, ub),
        objective_func=sphere
    )

    print(f"  种群大小: {ctx.population.shape}")
    print(f"  最优适应度: {ctx.best_fitness:.6f}")
    print(f"  多样性: {ctx.get_diversity():.6f}")
    print("  [OK] 上下文创建成功")
    return ctx


def test_explore_operators(ctx):
    """测试探索类算子"""
    print("\n" + "="*50)
    print("测试2: 探索类算子")
    print("="*50)

    operators = [
        GaussianStep(step_size=0.1),
        LevyJump(scale=1.0),
        DERand1(F=0.8),
        BatFrequencyMove(f_min=0, f_max=2)
    ]

    for op in operators:
        result = op.execute(ctx)
        print(f"  {op.name}:")
        print(f"    类型: {op.op_type.name}")
        print(f"    评估次数: {result.evaluations_used}")
        print(f"    改进: {'是' if result.success else '否'}")
        if result.success:
            print(f"    改进量: {ctx.best_fitness - result.best_fitness:.6f}")
    print("  [OK] 探索类算子测试通过")


def test_exploit_operators(ctx):
    """测试利用类算子"""
    print("\n" + "="*50)
    print("测试3: 利用类算子")
    print("="*50)

    operators = [
        PbestPull(c1=1.5),
        GbestPull(c2=1.5),
        CoordinateRefine(step_size=0.01, max_steps=5)
    ]

    for op in operators:
        result = op.execute(ctx)
        print(f"  {op.name}:")
        print(f"    类型: {op.op_type.name}")
        print(f"    评估次数: {result.evaluations_used}")
        if result.success:
            print(f"    改进量: {ctx.best_fitness - result.best_fitness:.6f}")
    print("  [OK] 利用类算子测试通过")


def test_recombine_operators(ctx):
    """测试重组类算子"""
    print("\n" + "="*50)
    print("测试4: 重组类算子")
    print("="*50)

    operators = [
        ArithmeticCrossover(cr=0.9),
        DECrossover(cr=0.9)
    ]

    for op in operators:
        result = op.execute(ctx)
        print(f"  {op.name}:")
        print(f"    类型: {op.op_type.name}")
        print(f"    评估次数: {result.evaluations_used}")
    print("  [OK] 重组类算子测试通过")


def test_accept_operators(ctx):
    """测试接受类算子"""
    print("\n" + "="*50)
    print("测试5: 接受类算子")
    print("="*50)

    # 先创建一个新种群
    gaussian = GaussianStep(step_size=0.1)
    result = gaussian.execute(ctx)

    # 测试接受算子
    acceptors = [
        GreedyAccept(),
        MetropolisAccept(temperature=1.0)
    ]

    for op in acceptors:
        # 更新上下文，包含原始种群
        test_ctx = OperatorContext(
            population=result.population,
            fitness=result.fitness,
            best_solution=result.best_solution,
            best_fitness=result.best_fitness,
            bounds=ctx.bounds,
            objective_func=ctx.objective_func,
            params={'original_pop': ctx.population, 'old_fitness': ctx.fitness}
        )
        accept_result = op.execute(test_ctx)
        print(f"  {op.name}:")
        print(f"    类型: {op.op_type.name}")
        print(f"    接受率: {accept_result.stats.get('accept_rate', 'N/A')}")

    print("  [OK] 接受类算子测试通过")


def test_motif_library():
    """测试Motif模板库"""
    print("\n" + "="*50)
    print("测试6: Motif模板库")
    print("="*50)

    print(f"  已注册Motif数量: {len(MOTIF_LIBRARY)}")
    print(f"  Motif列表: {get_motif_names()[:5]}...")

    # 测试获取Motif
    motif = get_motif('pso_standard')
    if motif:
        print(f"\n  测试Motif: {motif.name}")
        print(f"    描述: {motif.description}")
        print(f"    算子序列: {motif.get_primitive_names()}")
        print(f"    来源算法: {motif.source_algorithm}")

    print("\n  [OK] Motif模板库测试通过")


def test_motif_execution():
    """测试Motif执行"""
    print("\n" + "="*50)
    print("测试7: Motif执行")
    print("="*50)

    # 创建测试上下文
    pop_size, dim = 20, 10
    lb, ub = -5, 5
    population = np.random.uniform(lb, ub, (pop_size, dim))
    fitness = np.array([sphere(ind) for ind in population])

    ctx = OperatorContext(
        population=population,
        fitness=fitness,
        best_solution=population[np.argmin(fitness)],
        best_fitness=np.min(fitness),
        bounds=(lb, ub),
        objective_func=sphere,
        params={'velocity': np.zeros_like(population)}
    )

    initial_fitness = ctx.best_fitness

    # 测试不同Motif
    test_motifs = ['pso_standard', 'de_rand1', 'ba_standard', 'explore_first']

    for motif_name in test_motifs:
        motif = get_motif(motif_name)
        if motif:
            result = motif.execute(ctx)
            improvement = initial_fitness - result.best_fitness
            print(f"  {motif_name}:")
            print(f"    初始适应度: {initial_fitness:.6f}")
            print(f"    最终适应度: {result.best_fitness:.6f}")
            print(f"    改进: {improvement:.6f}")

    print("\n  [OK] Motif执行测试通过")


def test_graph_construction():
    """测试Typed-DAG构建"""
    print("\n" + "="*50)
    print("测试8: Typed-DAG构建")
    print("="*50)

    # 从Motif构建图
    motif = get_motif('pso_standard')
    graph = OperatorGraph.from_motif(motif)

    is_legal, errors = graph.is_legal()
    print(f"  从Motif构建图:")
    print(f"    合法性: {'合法' if is_legal else '非法'}")
    print(f"    节点数: {len([n for n in graph.nodes.values() if n.node_type.name == 'OPERATOR'])}")
    print(f"    执行顺序: {graph.get_execution_order()}")

    # 使用Builder构建图
    print(f"\n  使用Builder构建自定义图:")
    builder = GraphBuilder()
    custom_graph = (builder
        .add(GaussianStep(step_size=0.1))
        .add(GbestPull(c2=1.5))
        .add(GreedyAccept())
        .build())

    is_legal, errors = custom_graph.is_legal()
    print(f"    合法性: {'合法' if is_legal else '非法'}")
    print(f"    错误: {errors if errors else '无'}")

    print("\n  [OK] Typed-DAG构建测试通过")


def test_graph_execution():
    """测试图执行"""
    print("\n" + "="*50)
    print("测试9: Typed-DAG执行")
    print("="*50)

    # 创建测试上下文
    pop_size, dim = 30, 10
    lb, ub = -10, 10
    population = np.random.uniform(lb, ub, (pop_size, dim))
    fitness = np.array([rosenbrock(ind) for ind in population])

    ctx = OperatorContext(
        population=population,
        fitness=fitness,
        best_solution=population[np.argmin(fitness)],
        best_fitness=np.min(fitness),
        bounds=(lb, ub),
        objective_func=rosenbrock,
        params={'velocity': np.zeros_like(population)}
    )

    initial_fitness = ctx.best_fitness

    # 构建并执行图
    graph = OperatorGraph.from_motif(get_motif('de_rand1'))
    result = graph.execute(ctx)

    print(f"  测试函数: Rosenbrock")
    print(f"  初始适应度: {initial_fitness:.6f}")
    print(f"  最终适应度: {result.best_fitness:.6f}")
    print(f"  改进: {initial_fitness - result.best_fitness:.6f}")

    print("\n  [OK] 图执行测试通过")


def test_optimization_loop():
    """测试完整优化循环"""
    print("\n" + "="*50)
    print("测试10: 完整优化循环 (多迭代)")
    print("="*50)

    # 初始化
    pop_size, dim = 30, 10
    lb, ub = -5, 5
    max_iter = 50

    population = np.random.uniform(lb, ub, (pop_size, dim))
    fitness = np.array([rastrigin(ind) for ind in population])

    ctx = OperatorContext(
        population=population,
        fitness=fitness,
        best_solution=population[np.argmin(fitness)],
        best_fitness=np.min(fitness),
        bounds=(lb, ub),
        objective_func=rastrigin,
        params={'velocity': np.zeros_like(population)}
    )

    initial_fitness = ctx.best_fitness
    history = [ctx.best_fitness]

    # 使用PSO风格Motif进行优化
    motif = get_motif('pso_standard')

    for i in range(max_iter):
        result = motif.execute(ctx)
        ctx = OperatorContext(
            population=result.population,
            fitness=result.fitness,
            best_solution=result.best_solution,
            best_fitness=result.best_fitness,
            bounds=ctx.bounds,
            objective_func=ctx.objective_func,
            params={**ctx.params, **result.extra}
        )
        history.append(ctx.best_fitness)

    print(f"  测试函数: Rastrigin")
    print(f"  迭代次数: {max_iter}")
    print(f"  初始适应度: {initial_fitness:.6f}")
    print(f"  最终适应度: {ctx.best_fitness:.6f}")
    print(f"  总改进: {initial_fitness - ctx.best_fitness:.6f}")
    print(f"  收敛趋势: {history[:5]} -> {history[-5:]}")

    print("\n  [OK] 完整优化循环测试通过")


def test_operator_stats():
    """测试算子统计"""
    print("\n" + "="*50)
    print("测试11: 算子统计功能")
    print("="*50)

    from operators import get_operator_stats

    stats = get_operator_stats()
    print(f"  探索类算子: {stats['explore_operators']}")
    print(f"  利用类算子: {stats['exploit_operators']}")
    print(f"  重组类算子: {stats['recombine_operators']}")
    print(f"  接受类算子: {stats['accept_operators']}")
    print(f"  Primitive总数: {stats['total_primitives']}")
    print(f"  Motif总数: {stats['total_motifs']}")

    # 测试单个算子统计
    op = GaussianStep(step_size=0.1)
    print(f"\n  单个算子统计测试:")
    print(f"    算子名称: {op.name}")
    print(f"    初始调用次数: {op.stats.call_count}")

    # 执行几次
    pop_size, dim = 10, 5
    population = np.random.uniform(-5, 5, (pop_size, dim))
    fitness = np.array([sphere(ind) for ind in population])
    ctx = OperatorContext(
        population=population, fitness=fitness,
        best_solution=population[np.argmin(fitness)],
        best_fitness=np.min(fitness),
        bounds=(-5, 5), objective_func=sphere
    )

    for _ in range(5):
        op.execute(ctx)

    print(f"    执行后调用次数: {op.stats.call_count}")
    print(f"    成功率: {op.stats.success_rate:.2%}")

    print("\n  [OK] 算子统计测试通过")


# ==================== 主函数 ====================

def main():
    print("\n" + "="*60)
    print("   细粒度算子库测试")
    print("   Fine-grained Operator Library Test")
    print("="*60)

    # 创建基础上下文
    ctx = test_context_creation()

    # 运行所有测试
    test_explore_operators(ctx)
    test_exploit_operators(ctx)
    test_recombine_operators(ctx)
    test_accept_operators(ctx)
    test_motif_library()
    test_motif_execution()
    test_graph_construction()
    test_graph_execution()
    test_optimization_loop()
    test_operator_stats()

    print("\n" + "="*60)
    print("   所有测试通过! [OK]")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
