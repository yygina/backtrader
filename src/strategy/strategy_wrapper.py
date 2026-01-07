"""
策略包装器：支持风险管理、多时间框架
"""

import backtrader as bt
from typing import Optional, Dict, Any, List
import logging
from .risk_management import RiskManager, create_risk_manager

logger = logging.getLogger(__name__)


class StrategyWrapper(bt.Strategy):
    """
    策略包装器
    将基础策略与风险管理功能结合
    """
    
    params = (
        ('base_strategy', None),  # 基础策略类
        ('risk_params', {}),  # 风险管理参数
        ('strategy_params', {}),  # 策略参数
    )
    
    def __init__(self):
        """初始化策略包装器"""
        # 创建基础策略实例
        if self.params.base_strategy:
            self.base_strategy = self.params.base_strategy(**self.params.strategy_params)
            # 复制基础策略的指标
            for attr_name in dir(self.base_strategy):
                if not attr_name.startswith('_'):
                    attr = getattr(self.base_strategy, attr_name)
                    if not callable(attr) or isinstance(attr, bt.Indicator):
                        setattr(self, attr_name, attr)
        
        # 创建风险管理器
        self.risk_manager = create_risk_manager(self.params.risk_params) if self.params.risk_params else None
        
        # 持仓跟踪
        self.entry_prices = {}
        self.entry_times = {}
    
    def next(self):
        """策略逻辑"""
        # 先执行基础策略逻辑
        if hasattr(self.base_strategy, 'next'):
            self.base_strategy.next()
        
        # 检查并更新止盈止损
        if self.risk_manager:
            self._update_stop_loss_take_profit()
    
    def _update_stop_loss_take_profit(self):
        """更新止盈止损订单"""
        if not self.risk_manager:
            return
        
        # 遍历所有数据源
        for i, data in enumerate(self.datas):
            position = self.getposition(data)
            if position.size == 0:
                continue
            
            # 获取入场价格
            entry_price = self.entry_prices.get(i, data.close[0])
            current_price = data.close[0]
            
            # 移动止盈（如果启用）
            if self.risk_manager.trailing_stop:
                # 计算ATR（如果可用）
                if hasattr(self, 'atr'):
                    atr_value = self.atr[0]
                    trailing_stop_price = current_price - (atr_value * self.risk_manager.trailing_stop)
                    # 更新止损价格
                    if trailing_stop_price > entry_price * (1 - self.risk_manager.stop_loss_pct):
                        # 取消旧订单，创建新订单
                        pass  # 这里需要更复杂的订单管理逻辑
    
    def buy(
        self,
        data: Optional[Any] = None,  # bt.feeds data object
        size: Optional[float] = None,
        price: Optional[float] = None,
        plimit: Optional[float] = None,
        exectype: Optional[Any] = None,  # bt.Order.ExecType
        valid: Optional[Any] = None,  # bt.Order.ValidType
        tradeid: int = 0,
        oco: Optional[bt.Order] = None,
        trailamount: Optional[float] = None,
        trailpercent: Optional[float] = None,
        parent: Optional[bt.Order] = None,
        transmit: bool = True,
        **kwargs
    ) -> bt.Order:
        """买入（带风险管理）"""
        # 检查最大持仓数
        if self.risk_manager and not self.risk_manager.check_max_positions(self):
            logger.warning("已达到最大持仓数限制")
            return None
        
        # 计算仓位大小
        if size is None and self.risk_manager:
            if data is None:
                data = self.datas[0]
            size = self.risk_manager.calculate_position_size(self, data.close[0])
        
        # 执行买入
        order = super().buy(
            data=data,
            size=size,
            price=price,
            plimit=plimit,
            exectype=exectype,
            valid=valid,
            tradeid=tradeid,
            oco=oco,
            trailamount=trailamount,
            trailpercent=trailpercent,
            parent=parent,
            transmit=transmit,
            **kwargs
        )
        
        # 设置止盈止损
        if order and self.risk_manager and data:
            data_idx = self.datas.index(data) if data in self.datas else 0
            entry_price = price if price else data.close[0]
            self.entry_prices[data_idx] = entry_price
            self.entry_times[data_idx] = self.datas[0].datetime.datetime(0)
            
            # 延迟设置止盈止损（在订单执行后）
            # 这里简化处理，实际应该在notify_order中处理
        
        return order
    
    def notify_order(self, order):
        """订单通知"""
        if hasattr(self.base_strategy, 'notify_order'):
            self.base_strategy.notify_order(order)
        
        if order.status in [order.Completed]:
            # 订单完成，设置止盈止损
            if self.risk_manager and order.isbuy():
                data = order.data
                data_idx = self.datas.index(data) if data in self.datas else 0
                entry_price = order.executed.price
                size = order.executed.size
                
                self.risk_manager.set_stop_loss_take_profit(
                    self, data, size, entry_price
                )

