"""
多策略对比功能
"""

import pandas as pd
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class StrategyComparator:
    """
    策略对比器
    支持多策略性能对比
    """
    
    def __init__(self):
        """初始化"""
        self.comparison_results: List[Dict[str, Any]] = []
    
    def add_result(
        self,
        strategy_name: str,
        result: Dict[str, Any],
        config: Optional[Dict[str, Any]] = None
    ):
        """
        添加策略回测结果
        
        Args:
            strategy_name: 策略名称
            result: 回测结果
            config: 策略配置
        """
        if not result.get("success"):
            logger.warning(f"策略 {strategy_name} 回测失败，跳过对比")
            return
        
        performance = result.get("performance", {})
        
        self.comparison_results.append({
            "strategy_name": strategy_name,
            "performance": performance,
            "config": config or {},
            "result": result
        })
    
    def get_comparison_table(self) -> pd.DataFrame:
        """
        获取对比表格
        
        Returns:
            pd.DataFrame: 对比表格
        """
        if not self.comparison_results:
            return pd.DataFrame()
        
        rows = []
        for item in self.comparison_results:
            perf = item["performance"]
            row = {
                "策略名称": item["strategy_name"],
                "总收益率": f"{perf.get('total_return', 0):.2%}",
                "年化收益率": f"{perf.get('annual_return', 0):.2%}",
                "最大回撤": f"{perf.get('max_drawdown', 0):.2%}",
                "夏普比率": f"{perf.get('sharpe_ratio', 0):.2f}",
                "胜率": f"{perf.get('win_rate', 0):.2%}",
                "盈亏比": f"{perf.get('profit_loss_ratio', 0):.2f}",
                "总交易次数": perf.get("total_trades", 0),
                "SQN": f"{perf.get('sqn', 0):.2f}",
            }
            rows.append(row)
        
        return pd.DataFrame(rows)
    
    def get_best_strategy(self, metric: str = "total_return") -> Optional[Dict[str, Any]]:
        """
        获取最优策略
        
        Args:
            metric: 评价指标
        
        Returns:
            dict: 最优策略信息
        """
        if not self.comparison_results:
            return None
        
        best = max(
            self.comparison_results,
            key=lambda x: x["performance"].get(metric, 0)
        )
        
        return best
    
    def clear(self):
        """清空对比结果"""
        self.comparison_results = []
    
    def get_metrics_comparison(self) -> Dict[str, List[float]]:
        """
        获取指标对比数据（用于图表）
        
        Returns:
            dict: 指标对比数据
        """
        if not self.comparison_results:
            return {}
        
        metrics = {
            "strategy_names": [],
            "total_return": [],
            "annual_return": [],
            "max_drawdown": [],
            "sharpe_ratio": [],
            "win_rate": [],
        }
        
        for item in self.comparison_results:
            perf = item["performance"]
            metrics["strategy_names"].append(item["strategy_name"])
            metrics["total_return"].append(perf.get("total_return", 0))
            metrics["annual_return"].append(perf.get("annual_return", 0))
            metrics["max_drawdown"].append(perf.get("max_drawdown", 0))
            metrics["sharpe_ratio"].append(perf.get("sharpe_ratio", 0))
            metrics["win_rate"].append(perf.get("win_rate", 0))
        
        return metrics

