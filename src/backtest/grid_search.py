"""
参数网格搜索优化
"""

import backtrader as bt
import pandas as pd
from typing import Dict, Any, List, Optional
from itertools import product
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing

from .backtrader_engine import BacktraderEngine
from .backtrader_config import BacktraderConfig

logger = logging.getLogger(__name__)


class GridSearchOptimizer:
    """
    参数网格搜索优化器
    支持多线程并发优化
    """
    
    def __init__(
        self,
        engine: BacktraderEngine,
        max_workers: Optional[int] = None
    ):
        """
        初始化网格搜索优化器
        
        Args:
            engine: 回测引擎
            max_workers: 最大并发数，默认使用CPU核心数
        """
        self.engine = engine
        self.max_workers = max_workers or multiprocessing.cpu_count()
    
    def generate_param_combinations(
        self,
        param_grid: Dict[str, List[Any]]
    ) -> List[Dict[str, Any]]:
        """
        生成参数组合
        
        Args:
            param_grid: 参数网格，例如 {"fast_period": [10, 20], "slow_period": [30, 40]}
        
        Returns:
            List[Dict[str, Any]]: 参数组合列表
        """
        # 获取所有参数的键和值列表
        keys = list(param_grid.keys())
        values = list(param_grid.values())
        
        # 生成所有组合
        combinations = []
        for combo in product(*values):
            combinations.append(dict(zip(keys, combo)))
        
        return combinations
    
    def optimize(
        self,
        data: pd.DataFrame,
        strategy_class: type,
        param_grid: Dict[str, List[Any]],
        metric: str = "total_return"
    ) -> List[Dict[str, Any]]:
        """
        执行网格搜索优化
        
        Args:
            data: K线数据
            strategy_class: 策略类
            param_grid: 参数网格
            metric: 优化指标（如"total_return", "sharpe_ratio"）
        
        Returns:
            List[Dict[str, Any]]: 优化结果列表，按指标值排序
        """
        # 生成参数组合
        param_combinations = self.generate_param_combinations(param_grid)
        logger.info(f"开始网格搜索，共 {len(param_combinations)} 组参数")
        
        results = []
        
        # 使用线程池并发执行
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # 提交所有任务
            future_to_params = {
                executor.submit(
                    self._run_single_backtest,
                    data,
                    strategy_class,
                    params
                ): params
                for params in param_combinations
            }
            
            # 收集结果
            for future in as_completed(future_to_params):
                params = future_to_params[future]
                try:
                    result = future.result()
                    if result and result.get("success"):
                        performance = result.get("performance", {})
                        metric_value = performance.get(metric, 0)
                        
                        results.append({
                            "params": params,
                            "result": result,
                            "metric_value": metric_value,
                            "performance": performance
                        })
                        logger.info(f"参数 {params} 完成，{metric}={metric_value:.4f}")
                    else:
                        logger.warning(f"参数 {params} 回测失败")
                except Exception as e:
                    logger.error(f"参数 {params} 执行出错: {e}")
        
        # 按指标值排序
        results.sort(key=lambda x: x["metric_value"], reverse=True)
        
        logger.info(f"网格搜索完成，共 {len(results)} 组有效结果")
        return results
    
    def _run_single_backtest(
        self,
        data: pd.DataFrame,
        strategy_class: type,
        params: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        执行单次回测
        
        Args:
            data: K线数据
            strategy_class: 策略类
            params: 策略参数
        
        Returns:
            dict: 回测结果
        """
        try:
            return self.engine.run_backtest(
                data=data,
                strategy_class=strategy_class,
                strategy_params=params,
                use_cache=False  # 网格搜索不使用缓存
            )
        except Exception as e:
            logger.error(f"回测执行失败: {e}")
            return None
    
    def get_best_params(
        self,
        results: List[Dict[str, Any]],
        metric: str = "total_return",
        top_n: int = 1
    ) -> List[Dict[str, Any]]:
        """
        获取最优参数
        
        Args:
            results: 优化结果列表
            metric: 优化指标
            top_n: 返回前N个最优结果
        
        Returns:
            List[Dict[str, Any]]: 最优参数列表
        """
        if not results:
            return []
        
        # 已按指标值排序，直接返回前N个
        return results[:top_n]

