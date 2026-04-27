"""
实验记录集成示例
展示如何在现有算法脚本中最小化修改以添加实验记录
"""

import numpy as np
import sys
import os

# 添加实验记录器
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from experiment_logger import create_experiment_logger

# ============================================================================
# 假设这是您现有的算法函数
# ============================================================================

def existing_pso_algorithm(iterations=100, logger=None):
    """
    现有的PSO算法（简化版）

    Args:
        iterations: 迭代次数
        logger: 实验记录器实例（可选）
    """
    # 模拟PSO算法
    best_cost = 100.0
    best_solution = np.zeros(3)
    cost_history = []

    for i in range(iterations):
        # 模拟算法迭代
        improvement = np.random.uniform(0.5, 2.0)
        best_cost -= improvement
        best_solution = np.random.randn(3) * 0.1
        cost_history.append(best_cost)

        # ================================================
        # 添加的代码：记录迭代信息（如果提供了logger）
        # ================================================
        if logger:
            logger.start_iteration()
            logger.record_iteration(
                iteration=i,
                cost_value=float(best_cost),
                best_solution=best_solution.tolist(),
                improvement=float(improvement)
            )
        # ================================================

    return best_cost, best_solution, cost_history


def existing_ba_algorithm(iterations=100, logger=None):
    """
    现有的BA算法（简化版）

    Args:
        iterations: 迭代次数
        logger: 实验记录器实例（可选）
    """
    # 模拟BA算法
    best_cost = 80.0
    best_solution = np.ones(3) * 0.5
    cost_history = []

    for i in range(iterations):
        # 模拟算法迭代
        if i < iterations // 2:
            improvement = np.random.uniform(0.8, 3.0)
        else:
            improvement = np.random.uniform(0.1, 0.5)

        best_cost -= improvement
        best_solution += np.random.randn(3) * 0.05
        cost_history.append(best_cost)

        # ================================================
        # 添加的代码：记录迭代信息
        # ================================================
        if logger:
            logger.start_iteration()
            logger.record_iteration(
                iteration=i,
                cost_value=float(best_cost),
                best_solution=best_solution.tolist(),
                phase="exploration" if i < iterations // 2 else "exploitation"
            )
        # ================================================

    return best_cost, best_solution, cost_history


# ============================================================================
# 主函数：展示如何集成实验记录
# ============================================================================

def run_experiment_with_logging():
    """运行实验并记录过程"""

    print("=" * 60)
    print("实验记录集成示例")
    print("=" * 60)

    # 1. 创建实验记录器
    logger = create_experiment_logger(
        experiment_name="PSO_BA_Comparison",
        log_dir="example_logs"
    )

    # 2. 设置算法ID（可以运行多个算法）
    logger.set_algorithm_id("PSO")

    # 3. 记录实验参数
    logger.set_parameters({
        "iterations": 50,
        "algorithm": "PSO",
        "population_size": 30,
        "dimensions": 3,
        "test_run": True
    })

    # 4. 运行算法（传递logger参数）
    print("运行PSO算法...")
    final_cost, best_solution, cost_history = existing_pso_algorithm(
        iterations=50,
        logger=logger
    )

    # 5. 记录最终结果
    logger.record_final_results(
        final_cost=float(final_cost),
        best_solution=best_solution.tolist(),
        convergence_iterations=len(cost_history),
        note="PSO算法测试运行"
    )

    # 6. 保存实验记录
    log_file = logger.save()
    logger.update_central_index()

    print(f"PSO实验记录已保存到: {log_file}")

    # 7. 运行第二个实验（BA算法）
    print("\n运行BA算法...")

    # 创建新的记录器
    logger2 = create_experiment_logger(
        experiment_name="BA_Test",
        log_dir="example_logs"
    )
    logger2.set_algorithm_id("BA")
    logger2.set_parameters({
        "iterations": 40,
        "algorithm": "BA",
        "test_run": True
    })

    # 运行BA算法
    final_cost2, best_solution2, cost_history2 = existing_ba_algorithm(
        iterations=40,
        logger=logger2
    )

    # 记录结果
    logger2.record_final_results(
        final_cost=float(final_cost2),
        best_solution=best_solution2.tolist(),
        note="BA算法测试运行"
    )

    log_file2 = logger2.save()
    logger2.update_central_index()

    print(f"BA实验记录已保存到: {log_file2}")

    # 8. 结果比较
    print("\n" + "=" * 60)
    print("实验结果比较:")
    print(f"PSO - 最终代价: {final_cost:.4f}")
    print(f"BA  - 最终代价: {final_cost2:.4f}")

    # 9. 读取并显示索引
    index_file = "example_logs/experiment_index.json"
    if os.path.exists(index_file):
        import json
        with open(index_file, 'r', encoding='utf-8') as f:
            index = json.load(f)

        print("\n中央索引内容:")
        for exp in index:
            print(f"  - {exp['experiment_name']} ({exp['algorithm_id']}): "
                  f"代价={exp['final_cost']:.4f}")


def minimal_integration_example():
    """
    最小化集成示例
    展示如何在现有脚本中添加最少的代码
    """

    print("\n" + "=" * 60)
    print("最小化集成示例")
    print("=" * 60)

    # 假设这是您现有的主函数代码
    # 只需添加以下几行：

    # 1. 导入（添加到文件开头）
    # import sys
    # import os
    # sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    # from experiment_logger import create_experiment_logger

    # 2. 在适当位置创建记录器
    logger = create_experiment_logger("Minimal_Example")
    logger.set_algorithm_id("YourAlgorithm")

    # 3. 记录参数
    logger.set_parameters({
        "param1": "value1",
        "param2": 123
    })

    # 4. 在算法循环中添加记录（可选）
    # 如果算法有迭代过程，可以添加：
    # if logger:
    #     logger.record_iteration(i, current_cost)

    # 5. 记录最终结果
    # 假设这是您的算法结果
    final_result = 42.0
    best_params = [1.0, 2.0, 3.0]

    logger.record_final_results(
        final_cost=float(final_result),
        best_solution=best_params
    )

    # 6. 保存
    logger.save()
    logger.update_central_index()

    print("最小化集成完成！")
    print("只需添加6-10行代码即可获得完整的实验记录。")


if __name__ == "__main__":
    # 运行完整示例
    run_experiment_with_logging()

    # 运行最小化示例
    minimal_integration_example()

    print("\n" + "=" * 60)
    print("实验记录文件结构:")
    print("=" * 60)

    # 显示生成的文件
    import glob
    if os.path.exists("example_logs"):
        json_files = glob.glob("example_logs/*.json")
        print(f"生成的JSON文件: {len(json_files)}个")
        for f in json_files[:3]:  # 显示前3个
            print(f"  - {os.path.basename(f)}")

        if len(json_files) > 3:
            print(f"  ... 和 {len(json_files)-3} 个其他文件")

    print("\n下一步:")
    print("1. 查看 example_logs/ 目录中的记录文件")
    print("2. 阅读 '实验记录使用指南.md' 获取详细说明")
    print("3. 将 experiment_logger.py 复制到您的项目目录")
    print("4. 按照指南修改您的算法脚本")