"""
风险管理模块：止盈止损、仓位管理
"""

import backtrader as bt
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


class StopLossOrder(bt.Order):
    """止损订单"""
    pass


class TakeProfitOrder(bt.Order):
    """止盈订单"""
    pass


class RiskManager:
    """
    风险管理器
    支持止盈止损、仓位管理
    """
    
    def __init__(
        self,
        stop_loss_pct: Optional[float] = None,  # 止损百分比
        stop_loss_price: Optional[float] = None,  # 止损价格
        take_profit_pct: Optional[float] = None,  # 止盈百分比
        take_profit_price: Optional[float] = None,  # 止盈价格
        trailing_stop: Optional[float] = None,  # 移动止盈（ATR倍数）
        position_size: Optional[float] = None,  # 固定仓位大小
        position_pct: Optional[float] = None,  # 仓位百分比
        max_positions: Optional[int] = None,  # 最大持仓数
    ):
        """
        初始化风险管理器
        
        Args:
            stop_loss_pct: 止损百分比（如0.03表示3%）
            stop_loss_price: 止损价格（固定价格）
            take_profit_pct: 止盈百分比
            take_profit_price: 止盈价格（固定价格）
            trailing_stop: 移动止盈（ATR倍数）
            position_size: 固定仓位大小
            position_pct: 仓位百分比（0-1）
            max_positions: 最大持仓数
        """
        self.stop_loss_pct = stop_loss_pct
        self.stop_loss_price = stop_loss_price
        self.take_profit_pct = take_profit_pct
        self.take_profit_price = take_profit_price
        self.trailing_stop = trailing_stop
        self.position_size = position_size
        self.position_pct = position_pct
        self.max_positions = max_positions
        
        # 持仓跟踪
        self.entry_prices = {}  # 持仓的入场价格
        self.entry_times = {}  # 持仓的入场时间
    
    def calculate_position_size(
        self,
        strategy: bt.Strategy,
        price: float
    ) -> float:
        """
        计算仓位大小
        
        Args:
            strategy: 策略实例
            price: 当前价格
        
        Returns:
            float: 仓位大小
        """
        broker = strategy.broker
        cash = broker.getcash()
        value = broker.getvalue()
        
        # 固定仓位大小
        if self.position_size is not None:
            return self.position_size
        
        # 仓位百分比
        if self.position_pct is not None:
            return (value * self.position_pct) / price
        
        # 默认：使用所有可用资金
        return cash / price
    
    def check_max_positions(self, strategy: bt.Strategy) -> bool:
        """
        检查是否超过最大持仓数
        
        Args:
            strategy: 策略实例
        
        Returns:
            bool: 是否可以开新仓
        """
        if self.max_positions is None:
            return True
        
        # 计算当前持仓数
        positions = sum(1 for data in strategy.datas if strategy.getposition(data).size != 0)
        return positions < self.max_positions
    
    def set_stop_loss_take_profit(
        self,
        strategy: bt.Strategy,
        data: Any,  # bt.feeds data object
        size: float,
        entry_price: float
    ):
        """
        设置止盈止损订单
        
        Args:
            strategy: 策略实例
            data: 数据源
            size: 持仓大小
            entry_price: 入场价格
        """
        # 止损
        if self.stop_loss_pct is not None:
            stop_loss_price = entry_price * (1 - self.stop_loss_pct)
            strategy.sell(
                data=data,
                size=abs(size),
                exectype=bt.Order.Stop,
                price=stop_loss_price,
                name="止损"
            )
        elif self.stop_loss_price is not None:
            strategy.sell(
                data=data,
                size=abs(size),
                exectype=bt.Order.Stop,
                price=self.stop_loss_price,
                name="止损"
            )
        
        # 止盈
        if self.take_profit_pct is not None:
            take_profit_price = entry_price * (1 + self.take_profit_pct)
            strategy.sell(
                data=data,
                size=abs(size),
                exectype=bt.Order.Limit,
                price=take_profit_price,
                name="止盈"
            )
        elif self.take_profit_price is not None:
            strategy.sell(
                data=data,
                size=abs(size),
                exectype=bt.Order.Limit,
                price=self.take_profit_price,
                name="止盈"
            )


def create_risk_manager(params: Dict[str, Any]) -> RiskManager:
    """
    从参数字典创建风险管理器
    
    Args:
        params: 参数字典
    
    Returns:
        RiskManager: 风险管理器实例
    """
    return RiskManager(
        stop_loss_pct=params.get("stop_loss_pct"),
        stop_loss_price=params.get("stop_loss_price"),
        take_profit_pct=params.get("take_profit_pct"),
        take_profit_price=params.get("take_profit_price"),
        trailing_stop=params.get("trailing_stop"),
        position_size=params.get("position_size"),
        position_pct=params.get("position_pct"),
        max_positions=params.get("max_positions")
    )

