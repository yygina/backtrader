"""
内置策略模板：MA/EMA/RSI/MACD/BOLL等经典策略
"""

import backtrader as bt
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


class MAStrategy(bt.Strategy):
    """
    MA移动平均线交叉策略
    短期MA上穿长期MA时买入，下穿时卖出
    """
    
    params = (
        ('fast_period', 10),  # 短期MA周期
        ('slow_period', 30),  # 长期MA周期
        ('printlog', False),
    )
    
    def __init__(self):
        """初始化指标"""
        self.fast_ma = bt.indicators.SMA(
            self.data.close,
            period=self.params.fast_period
        )
        self.slow_ma = bt.indicators.SMA(
            self.data.close,
            period=self.params.slow_period
        )
        
        # 交叉信号
        self.crossover = bt.indicators.CrossOver(self.fast_ma, self.slow_ma)
        
        # 记录订单执行信息（用于后续提取买卖点）
        self.order_history = []
    
    def prenext(self):
        """数据不足时的处理"""
        # Backtrader会自动等待直到有足够数据
        pass
    
    def notify_order(self, order):
        """订单通知：记录订单执行信息"""
        if order.status == order.Completed:
            # 记录订单执行信息
            order_info = {
                "date": self.data.datetime.date(0),
                "price": order.executed.price,
                "size": order.executed.size,
                "type": "buy" if order.isbuy() else "sell"
            }
            self.order_history.append(order_info)
    
    def next(self):
        """策略逻辑"""
        # CrossOver指标会在交叉时返回非0值
        # > 0 表示快线上穿慢线（金叉）
        # < 0 表示快线下穿慢线（死叉）
        # == 0 表示没有交叉
        
        # 金叉：买入
        if self.crossover[0] > 0:
            if not self.position:
                self.buy()
                logger.info(f'买入信号: 日期={self.data.datetime.date(0)}, 价格={self.data.close[0]:.2f}, 快MA={self.fast_ma[0]:.2f}, 慢MA={self.slow_ma[0]:.2f}')
                if self.params.printlog:
                    self.log(f'买入: {self.data.close[0]:.2f}')
        
        # 死叉：卖出
        elif self.crossover[0] < 0:
            if self.position:
                self.sell()
                logger.info(f'卖出信号: 日期={self.data.datetime.date(0)}, 价格={self.data.close[0]:.2f}, 快MA={self.fast_ma[0]:.2f}, 慢MA={self.slow_ma[0]:.2f}')
                if self.params.printlog:
                    self.log(f'卖出: {self.data.close[0]:.2f}')
    
    def log(self, txt, dt=None):
        """日志记录"""
        dt = dt or self.datas[0].datetime.date(0)
        logger.info(f'{dt.isoformat()}: {txt}')


class EMAStrategy(bt.Strategy):
    """
    EMA指数移动平均线交叉策略
    """
    
    params = (
        ('fast_period', 12),
        ('slow_period', 26),
        ('printlog', False),
    )
    
    def __init__(self):
        self.fast_ema = bt.indicators.EMA(
            self.data.close,
            period=self.params.fast_period
        )
        self.slow_ema = bt.indicators.EMA(
            self.data.close,
            period=self.params.slow_period
        )
        self.crossover = bt.indicators.CrossOver(self.fast_ema, self.slow_ema)
        self.order_history = []  # 初始化订单历史记录
    
    def notify_order(self, order):
        """订单通知：记录订单执行信息"""
        # 记录所有订单状态（用于调试）
        status_map = {
            order.Submitted: '已提交',
            order.Accepted: '已接受',
            order.Rejected: '已拒绝',
            order.Margin: '保证金不足',
            order.Canceled: '已取消',
            order.Expired: '已过期',
            order.Partial: '部分成交',
            order.Completed: '已完成'
        }
        status_str = status_map.get(order.status, f'未知({order.status})')
        
        # 记录所有订单状态变化（用于调试）
        if order.status == order.Completed:
            # 获取订单执行日期 - 简化逻辑，直接使用当前数据源的日期
            try:
                order_date = self.data.datetime.date(0)
                order_info = {
                    "date": order_date,
                    "price": float(order.executed.price),
                    "size": float(order.executed.size),
                    "type": "buy" if order.isbuy() else "sell"
                }
                self.order_history.append(order_info)
                logger.info(f' EMA订单完成: {order_info["type"]}, 日期={order_info["date"]}, 价格={order_info["price"]:.2f}')
            except Exception as e:
                logger.error(f' EMA无法记录订单: {e}, 价格={order.executed.price if hasattr(order, "executed") else "N/A"}')
        elif order.status in [order.Rejected, order.Margin, order.Canceled]:
            # 记录失败的订单
            logger.warning(f' EMA订单失败: 类型={"买入" if order.isbuy() else "卖出"}, 状态={status_str}')
        else:
            logger.debug(f'EMA订单状态: 类型={"买入" if order.isbuy() else "卖出"}, 状态={status_str}')
    
    def next(self):
        # 金叉：买入
        if self.crossover[0] > 0:
            if not self.position:
                self.buy()
                logger.info(f'EMA买入信号: 日期={self.data.datetime.date(0)}, 快EMA={self.fast_ema[0]:.2f}, 慢EMA={self.slow_ema[0]:.2f}')
        # 死叉：卖出
        elif self.crossover[0] < 0:
            if self.position:
                self.sell()
                logger.info(f'EMA卖出信号: 日期={self.data.datetime.date(0)}, 快EMA={self.fast_ema[0]:.2f}, 慢EMA={self.slow_ema[0]:.2f}')


class RSIStrategy(bt.Strategy):
    """
    RSI超买超卖策略
    RSI < 30时买入，RSI > 70时卖出
    """
    
    params = (
        ('rsi_period', 14),
        ('rsi_oversold', 30),
        ('rsi_overbought', 70),
        ('printlog', False),
    )
    
    def __init__(self):
        self.rsi = bt.indicators.RSI(
            self.data.close,
            period=self.params.rsi_period
        )
        self.order_history = []  # 初始化订单历史记录
    
    def notify_order(self, order):
        """订单通知：记录订单执行信息"""
        # 记录所有订单状态（用于调试）
        status_map = {
            order.Submitted: '已提交',
            order.Accepted: '已接受',
            order.Rejected: '已拒绝',
            order.Margin: '保证金不足',
            order.Canceled: '已取消',
            order.Expired: '已过期',
            order.Partial: '部分成交',
            order.Completed: '已完成'
        }
        status_str = status_map.get(order.status, f'未知({order.status})')
        
        # 记录所有订单状态变化（用于调试）
        if order.status == order.Completed:
            # 获取订单执行日期 - 简化逻辑，直接使用当前数据源的日期
            try:
                order_date = self.data.datetime.date(0)
                order_info = {
                    "date": order_date,
                    "price": float(order.executed.price),
                    "size": float(order.executed.size),
                    "type": "buy" if order.isbuy() else "sell"
                }
                self.order_history.append(order_info)
                logger.info(f' EMA订单完成: {order_info["type"]}, 日期={order_info["date"]}, 价格={order_info["price"]:.2f}')
            except Exception as e:
                logger.error(f' EMA无法记录订单: {e}, 价格={order.executed.price if hasattr(order, "executed") else "N/A"}')
        elif order.status in [order.Rejected, order.Margin, order.Canceled]:
            # 记录失败的订单
            logger.warning(f' EMA订单失败: 类型={"买入" if order.isbuy() else "卖出"}, 状态={status_str}')
        else:
            logger.debug(f'EMA订单状态: 类型={"买入" if order.isbuy() else "卖出"}, 状态={status_str}')
    
    def next(self):
        # 超卖：买入
        if self.rsi < self.params.rsi_oversold:
            if not self.position:
                self.buy()
                if self.params.printlog:
                    self.log(f'RSI超卖买入: RSI={self.rsi[0]:.2f}')
        
        # 超买：卖出
        elif self.rsi > self.params.rsi_overbought:
            if self.position:
                self.sell()
                if self.params.printlog:
                    self.log(f'RSI超买卖出: RSI={self.rsi[0]:.2f}')


class MACDStrategy(bt.Strategy):
    """
    MACD金叉死叉策略
    MACD线上穿信号线时买入，下穿时卖出
    """
    
    params = (
        ('macd_fast', 12),
        ('macd_slow', 26),
        ('macd_signal', 9),
        ('printlog', False),
    )
    
    def __init__(self):
        self.macd = bt.indicators.MACD(
            self.data.close,
            period_me1=self.params.macd_fast,
            period_me2=self.params.macd_slow,
            period_signal=self.params.macd_signal
        )
        self.crossover = bt.indicators.CrossOver(
            self.macd.macd,
            self.macd.signal
        )
        self.order_history = []
    
    def notify_order(self, order):
        """订单通知：记录订单执行信息"""
        # 记录所有订单状态（用于调试）
        status_map = {
            order.Submitted: '已提交',
            order.Accepted: '已接受',
            order.Rejected: '已拒绝',
            order.Margin: '保证金不足',
            order.Canceled: '已取消',
            order.Expired: '已过期',
            order.Partial: '部分成交',
            order.Completed: '已完成'
        }
        status_str = status_map.get(order.status, f'未知({order.status})')
        
        # 记录所有订单状态变化（用于调试）
        if order.status == order.Completed:
            # 获取订单执行日期 - 简化逻辑，直接使用当前数据源的日期
            try:
                order_date = self.data.datetime.date(0)
                order_info = {
                    "date": order_date,
                    "price": float(order.executed.price),
                    "size": float(order.executed.size),
                    "type": "buy" if order.isbuy() else "sell"
                }
                self.order_history.append(order_info)
                logger.info(f' EMA订单完成: {order_info["type"]}, 日期={order_info["date"]}, 价格={order_info["price"]:.2f}')
            except Exception as e:
                logger.error(f' EMA无法记录订单: {e}, 价格={order.executed.price if hasattr(order, "executed") else "N/A"}')
        elif order.status in [order.Rejected, order.Margin, order.Canceled]:
            # 记录失败的订单
            logger.warning(f' EMA订单失败: 类型={"买入" if order.isbuy() else "卖出"}, 状态={status_str}')
        else:
            logger.debug(f'EMA订单状态: 类型={"买入" if order.isbuy() else "卖出"}, 状态={status_str}')
    
    def next(self):
        # 金叉：买入
        if self.crossover[0] > 0:
            if not self.position:
                self.buy()
                logger.info(f'MACD买入信号: 日期={self.data.datetime.date(0)}')
        # 死叉：卖出
        elif self.crossover[0] < 0:
            if self.position:
                self.sell()
                logger.info(f'MACD卖出信号: 日期={self.data.datetime.date(0)}')


class BOLLStrategy(bt.Strategy):
    """
    BOLL布林带突破策略
    价格突破上轨时买入，跌破下轨时卖出
    """
    
    params = (
        ('boll_period', 20),
        ('boll_dev', 2.0),
        ('printlog', False),
    )
    
    def __init__(self):
        self.boll = bt.indicators.BollingerBands(
            self.data.close,
            period=self.params.boll_period,
            devfactor=self.params.boll_dev
        )
    
    def next(self):
        # 突破上轨：买入
        if self.data.close > self.boll.lines.top:
            if not self.position:
                self.buy()
        
        # 跌破下轨：卖出
        elif self.data.close < self.boll.lines.bot:
            if self.position:
                self.sell()


def get_strategy_class(strategy_name: str) -> Optional[type]:
    """
    根据策略名称获取策略类
    
    Args:
        strategy_name: 策略名称（"MA", "EMA", "RSI", "MACD", "BOLL"）
    
    Returns:
        type: 策略类，不存在返回None
    """
    strategy_map = {
        "MA": MAStrategy,
        "EMA": EMAStrategy,
        "RSI": RSIStrategy,
        "MACD": MACDStrategy,
        "BOLL": BOLLStrategy
    }
    return strategy_map.get(strategy_name.upper())

