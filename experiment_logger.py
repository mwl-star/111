"""
实验记录模块
用于记录算法运行的过程和结果
"""

import json
import os
import time
from datetime import datetime
from typing import Any, Dict, List, Optional
import uuid


class ExperimentLogger:
    """实验记录器类"""

    def __init__(self, experiment_name: str = None, log_dir: str = "experiment_logs"):
        """
        初始化实验记录器

        Args:
            experiment_name: 实验名称，如果为None则自动生成
            log_dir: 日志目录
        """
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)

        # 生成实验ID和时间戳
        self.experiment_id = str(uuid.uuid4())[:8]
        self.start_time = datetime.now()
        self.experiment_name = experiment_name or f"experiment_{self.experiment_id}"

        # 实验数据存储
        self.data = {
            "experiment_id": self.experiment_id,
            "experiment_name": self.experiment_name,
            "start_time": self.start_time.isoformat(),
            "algorithm_id": None,
            "parameters": {},
            "results": {},
            "iteration_history": [],
            "metadata": {
                "platform": os.name,
                "working_directory": os.getcwd(),
            }
        }

        # 性能统计
        self.iteration_start_time = None

    def set_algorithm_id(self, algorithm_id: str):
        """设置算法ID"""
        self.data["algorithm_id"] = algorithm_id

    def set_parameters(self, parameters: Dict[str, Any]):
        """设置算法参数"""
        self.data["parameters"].update(parameters)

    def add_result(self, key: str, value: Any):
        """添加结果"""
        self.data["results"][key] = value

    def start_iteration(self):
        """开始迭代记录"""
        self.iteration_start_time = time.time()

    def record_iteration(self, iteration: int, cost_value: float, **kwargs):
        """
        记录一次迭代

        Args:
            iteration: 迭代次数
            cost_value: 代价函数值
            **kwargs: 其他需要记录的迭代数据
        """
        iteration_data = {
            "iteration": iteration,
            "cost_value": float(cost_value),
            "timestamp": datetime.now().isoformat()
        }

        if self.iteration_start_time:
            iteration_data["iteration_duration"] = time.time() - self.iteration_start_time
            self.iteration_start_time = time.time()

        iteration_data.update(kwargs)
        self.data["iteration_history"].append(iteration_data)

    def record_final_results(self, final_cost: float, best_solution: List[float], **kwargs):
        """记录最终结果"""
        self.add_result("final_cost", float(final_cost))
        self.add_result("best_solution", [float(x) for x in best_solution])
        self.add_result("end_time", datetime.now().isoformat())
        self.add_result("total_duration", time.time() - self.start_time.timestamp())

        for key, value in kwargs.items():
            self.add_result(key, value)

    def save(self, filename: str = None):
        """
        保存实验记录到JSON文件

        Args:
            filename: 文件名，如果为None则自动生成
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{self.experiment_name}_{self.experiment_id}_{timestamp}.json"

        filepath = os.path.join(self.log_dir, filename)

        # 确保目录存在
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.data, f, ensure_ascii=False, indent=2)

        print(f"实验记录已保存到: {filepath}")
        return filepath

    def update_central_index(self, index_file: str = "experiment_index.json"):
        """
        更新中央索引文件

        Args:
            index_file: 索引文件名
        """
        index_path = os.path.join(self.log_dir, index_file)

        if os.path.exists(index_path):
            with open(index_path, 'r', encoding='utf-8') as f:
                index_data = json.load(f)
        else:
            index_data = []

        # 添加当前实验的摘要信息
        summary = {
            "experiment_id": self.experiment_id,
            "experiment_name": self.experiment_name,
            "algorithm_id": self.data["algorithm_id"],
            "start_time": self.data["start_time"],
            "end_time": self.data.get("results", {}).get("end_time"),
            "final_cost": self.data.get("results", {}).get("final_cost"),
            "log_file": f"{self.experiment_name}_{self.experiment_id}_*.json"
        }

        index_data.append(summary)

        with open(index_path, 'w', encoding='utf-8') as f:
            json.dump(index_data, f, ensure_ascii=False, indent=2)

        print(f"中央索引已更新: {index_path}")


def create_experiment_logger(experiment_name=None, log_dir="experiment_logs"):
    """创建实验记录器的便捷函数"""
    return ExperimentLogger(experiment_name, log_dir)


# 使用示例
if __name__ == "__main__":
    # 示例用法
    logger = create_experiment_logger("PSO_Test")
    logger.set_algorithm_id("PSO")
    logger.set_parameters({
        "w": 0.8,
        "c1": 1.5,
        "c2": 1.5,
        "population_size": 30,
        "max_iterations": 100
    })

    # 模拟迭代过程
    for i in range(10):
        logger.start_iteration()
        # 模拟计算
        cost = 100 - i * 2 + (i % 3)
        logger.record_iteration(i, cost, best_solution=[i*0.1, i*0.2])

    # 记录最终结果
    logger.record_final_results(
        final_cost=5.2,
        best_solution=[0.1, 0.2, 0.3],
        additional_info="测试完成"
    )

    # 保存记录
    log_file = logger.save()
    logger.update_central_index()