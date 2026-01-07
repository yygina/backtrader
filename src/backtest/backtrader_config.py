"""
Backtrader配置层：Cerebro初始化、Broker参数配置
完全对齐Backtrader原生能力
"""

import backtrader as bt
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


class BacktraderConfig:
    """
    Backtrader配置类
    负责初始化Cerebro引擎和Broker参数配置
    """
    
    def __init__(
        self,
        initial_cash: float = 10000.0,
        commission: float = 0.001,  # 0.1%
        commission_type: str = "percentage",  # "percentage" or "fixed"
        maker_commission: Optional[float] = None,
        taker_commission: Optional[float] = None,
        slippage: Optional[float] = None,
        slippage_type: str = "percentage",  # "percentage" or "fixed"
        leverage: Optional[int] = None,  # 仅合约市场
        margin: Optional[float] = None,  # 保证金比例（合约）
        min_trade_size: Optional[float] = None,
        commission_currency: str = "USDT"
    ):
        """
        初始化回测配置
        
        Args:
            initial_cash: 初始资金（默认10000 USDT）
            commission: 手续费率（百分比，如0.001表示0.1%）
            commission_type: 手续费类型，"percentage"或"fixed"
            maker_commission: Maker手续费率（可选，分开配置）
            taker_commission: Taker手续费率（可选，分开配置）
            slippage: 滑点（百分比或固定值）
            slippage_type: 滑点类型，"percentage"或"fixed"
            leverage: 杠杆倍数（1-125，仅合约）
            margin: 保证金比例（合约专用）
            min_trade_size: 最小交易单位
            commission_currency: 佣金货币（默认USDT）
        """
        self.initial_cash = initial_cash
        self.commission = commission
        self.commission_type = commission_type
        self.maker_commission = maker_commission
        self.taker_commission = taker_commission
        self.slippage = slippage
        self.slippage_type = slippage_type
        self.leverage = leverage
        self.margin = margin
        self.min_trade_size = min_trade_size
        self.commission_currency = commission_currency
        
        # 参数校验
        self._validate_params()
    
    def _validate_params(self):
        """参数校验（快速失败）"""
        if self.initial_cash <= 0:
            raise ValueError("初始资金必须大于0")
        
        if self.commission_type == "percentage":
            if self.commission < 0 or self.commission > 0.1:
                raise ValueError("手续费率必须在0-0.1之间（0-10%）")
        else:
            if self.commission < 0:
                raise ValueError("固定手续费必须大于等于0")
        
        if self.leverage is not None:
            if self.leverage < 1 or self.leverage > 125:
                raise ValueError("杠杆倍数必须在1-125之间")
        
        if self.slippage is not None and self.slippage < 0:
            raise ValueError("滑点必须大于等于0")
    
    def create_cerebro(self) -> bt.Cerebro:
        """
        创建并配置Cerebro引擎
        
        Returns:
            bt.Cerebro: 配置好的Cerebro实例
        """
        cerebro = bt.Cerebro()
        
        # 设置初始资金
        cerebro.broker.setcash(self.initial_cash)
        
        # 配置手续费
        if self.commission_type == "percentage":
            if self.maker_commission is not None and self.taker_commission is not None:
                # 分别配置maker/taker（Backtrader使用单一费率，取平均值）
                avg_commission = (self.maker_commission + self.taker_commission) / 2
                cerebro.broker.setcommission(commission=avg_commission)
            else:
                cerebro.broker.setcommission(commission=self.commission)
        else:
            # 固定手续费
            cerebro.broker.setcommission(
                commission=self.commission,
                commtype=bt.CommInfoBase.COMM_FIXED
            )
        
        # 配置滑点（通过Slippage类实现）
        if self.slippage is not None:
            if self.slippage_type == "percentage":
                # 百分比滑点
                cerebro.broker.set_slippage_perc(
                    perc=self.slippage,
                    slip_open=True,
                    slip_limit=True,
                    slip_match=True,
                    slip_out=False
                )
            else:
                # 固定滑点（点数）
                cerebro.broker.set_slippage_fixed(
                    fixed=self.slippage,
                    slip_open=True,
                    slip_limit=True,
                    slip_match=True,
                    slip_out=False
                )
        
        # 配置杠杆（仅合约市场）
        if self.leverage is not None:
            cerebro.broker.setcommission(leverage=self.leverage)
        
        # 配置最小交易单位
        if self.min_trade_size is not None:
            # Backtrader通过CommissionInfo设置最小交易单位
            # 这里需要在数据层面处理
            pass
        
        # 设置分析器（绩效指标）
        cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
        cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
        cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
        cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
        cerebro.addanalyzer(bt.analyzers.SQN, _name='sqn')
        
        # 添加观察者记录资金曲线
        cerebro.addobserver(bt.observers.Value, _name='value')
        
        logger.info(f"Cerebro配置完成: 初始资金={self.initial_cash}, "
                   f"手续费={self.commission}, 杠杆={self.leverage}")
        
        return cerebro
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典（用于缓存键生成）"""
        return {
            "initial_cash": self.initial_cash,
            "commission": self.commission,
            "commission_type": self.commission_type,
            "maker_commission": self.maker_commission,
            "taker_commission": self.taker_commission,
            "slippage": self.slippage,
            "slippage_type": self.slippage_type,
            "leverage": self.leverage,
            "margin": self.margin,
            "min_trade_size": self.min_trade_size,
            "commission_currency": self.commission_currency
        }

