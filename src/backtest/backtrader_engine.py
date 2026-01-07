"""
回测引擎层：策略执行、订单管理、绩效指标计算
遵循设计原则：资源极简复用（复用Cerebro对象）
"""

# 设置matplotlib后端（必须在导入backtrader之前，因为backtrader可能使用matplotlib）
import os
os.environ['MPLBACKEND'] = 'Agg'  # 通过环境变量设置，确保Backtrader内部也使用Agg

import matplotlib
matplotlib.use('Agg', force=True)  # 强制使用Agg后端，避免macOS线程问题

import backtrader as bt
import pandas as pd
from typing import Optional, Dict, Any, List, Type
import logging
from datetime import datetime
import hashlib
import json

from .backtrader_config import BacktraderConfig
from .cache_manager import CacheManager
from ..strategy.strategy_templates import get_strategy_class
from ..system.exception_handler import ExceptionHandler

logger = logging.getLogger(__name__)


class BacktraderEngine:
    """
    回测引擎
    负责执行回测、计算绩效指标、管理结果缓存
    """
    
    def __init__(
        self,
        config: Optional[BacktraderConfig] = None,
        cache_manager: Optional[CacheManager] = None
    ):
        """
        初始化回测引擎
        
        Args:
            config: 回测配置，默认使用默认配置
            cache_manager: 缓存管理器
        """
        self.config = config or BacktraderConfig()
        self.exception_handler = ExceptionHandler()
        self.cache_manager = cache_manager or CacheManager()
        self._cerebro_cache: Optional[bt.Cerebro] = None
    
    def _get_cerebro(self) -> bt.Cerebro:
        """
        获取Cerebro实例（资源复用）
        
        Returns:
            bt.Cerebro: Cerebro实例
        """
        # 每次创建新的Cerebro（因为需要注入不同的数据和策略）
        # 但可以复用配置逻辑
        return self.config.create_cerebro()
    
    def _prepare_datafeed(
        self,
        df: pd.DataFrame,
        name: str = "data"
    ) -> bt.feeds.PandasData:
        """
        将DataFrame转换为Backtrader DataFeed
        
        Args:
            df: K线数据DataFrame
            name: 数据名称
        
        Returns:
            bt.feeds.PandasData: Backtrader数据源
        """
        # 确保列名符合Backtrader要求
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in df.columns for col in required_columns):
            raise ValueError(f"DataFrame必须包含列: {required_columns}")
        
        # 确保数据按时间排序
        df = df.sort_index()
        
        # 检查数据有效性
        logger.info(f"数据准备: 共 {len(df)} 条, 时间范围: {df.index[0]} 到 {df.index[-1]}")
        if len(df) < 50:
            logger.warning(f"数据量较少 ({len(df)} 条)，可能影响策略执行")
        
        # 创建PandasData Feed
        datafeed = bt.feeds.PandasData(
            dataname=df,
            datetime=None,  # 使用索引作为时间
            open=0,
            high=1,
            low=2,
            close=3,
            volume=4,
            openinterest=-1
        )
        
        return datafeed
    
    def run_backtest(
        self,
        data: pd.DataFrame,
        strategy_class: Type[bt.Strategy],
        strategy_params: Optional[Dict[str, Any]] = None,
        use_cache: bool = True,
        symbol: Optional[str] = None,
        interval: Optional[str] = None,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        执行回测
        
        Args:
            data: K线数据DataFrame
            strategy_class: 策略类
            strategy_params: 策略参数
            use_cache: 是否使用缓存
        
        Returns:
            dict: 回测结果，包含绩效指标和交易记录
        """
        try:
            # 参数校验（快速失败）
            if data.empty:
                raise ValueError("数据为空，无法执行回测")
            
            if strategy_params is None:
                strategy_params = {}
            
            # 初始化变量
            use_cached_performance = False
            cached_result = None
            
            # 检查缓存（但需要重新运行以获取cerebro对象用于绘图）
            if use_cache and symbol and interval and start_time and end_time:
                cached_result = self.cache_manager.get(
                    symbol=symbol,
                    interval=interval,
                    start_time=str(start_time),
                    end_time=str(end_time),
                    strategy_name=strategy_class.__name__,
                    strategy_params=strategy_params or {},
                    backtest_params=self.config.to_dict()
                )
                if cached_result:
                    logger.info("找到缓存结果，但需要重新运行以生成图表和完整绩效指标")
                    # 不再使用缓存的performance，总是重新计算以确保数据完整
                    use_cached_performance = False
                    # 不直接返回，继续执行以获取cerebro对象和完整绩效指标
            
            # 创建Cerebro
            cerebro = self._get_cerebro()
            
            # 准备数据
            datafeed = self._prepare_datafeed(data)
            
            # 检查数据量是否足够（根据策略参数计算最小需求）
            min_period = 0
            if strategy_params:
                # 获取策略所需的最大周期
                if 'slow_period' in strategy_params:
                    min_period = max(min_period, strategy_params['slow_period'])
                if 'fast_period' in strategy_params:
                    min_period = max(min_period, strategy_params['fast_period'])
                if 'period' in strategy_params:
                    min_period = max(min_period, strategy_params['period'])
            
            # 如果数据量不足，返回错误
            if len(data) < min_period:
                error_msg = f"数据量不足: 需要至少 {min_period} 条数据，但只有 {len(data)} 条。请扩大时间范围或选择更短的时间周期。"
                logger.error(error_msg)
                return {
                    "success": False,
                    "error": error_msg,
                    "performance": {},
                    "trades": [],
                    "final_value": self.config.initial_cash,
                    "initial_value": self.config.initial_cash
                }
            
            cerebro.adddata(datafeed)
            
            # 设置仓位大小：使用百分比仓位（使用95%的资金，留5%作为缓冲）
            # 这样可以避免"保证金不足"的问题
            cerebro.addsizer(bt.sizers.PercentSizer, percents=95)
            
            # 创建策略包装器来记录资金曲线和指标数据
            class EquityRecorderStrategy(strategy_class):
                def __init__(self):
                    super().__init__()
                    self.equity_curve_data = []
                    self.indicator_data = {}  # 存储指标数据
                
                def next(self):
                    super().next()
                    # 记录每日资金
                    try:
                        current_date = self.data.datetime.date(0)
                        current_value = self.broker.getvalue()
                        self.equity_curve_data.append({
                            'date': current_date,
                            'value': current_value
                        })
                        
                        # 记录指标数据（如果策略有指标）
                        # MA策略
                        if hasattr(self, 'fast_ma') and hasattr(self, 'slow_ma'):
                            if current_date not in self.indicator_data:
                                self.indicator_data[current_date] = {}
                            self.indicator_data[current_date]['fast_ma'] = float(self.fast_ma[0])
                            self.indicator_data[current_date]['slow_ma'] = float(self.slow_ma[0])
                        
                        # EMA策略
                        if hasattr(self, 'fast_ema') and hasattr(self, 'slow_ema'):
                            if current_date not in self.indicator_data:
                                self.indicator_data[current_date] = {}
                            self.indicator_data[current_date]['fast_ema'] = float(self.fast_ema[0])
                            self.indicator_data[current_date]['slow_ema'] = float(self.slow_ema[0])
                        
                        # RSI策略
                        if hasattr(self, 'rsi'):
                            if current_date not in self.indicator_data:
                                self.indicator_data[current_date] = {}
                            self.indicator_data[current_date]['rsi'] = float(self.rsi[0])
                        
                        # MACD策略
                        if hasattr(self, 'macd') and hasattr(self, 'signal'):
                            if current_date not in self.indicator_data:
                                self.indicator_data[current_date] = {}
                            self.indicator_data[current_date]['macd'] = float(self.macd[0])
                            self.indicator_data[current_date]['signal'] = float(self.signal[0])
                        
                        # BOLL策略
                        if hasattr(self, 'boll') and hasattr(self.boll, 'top') and hasattr(self.boll, 'bot'):
                            if current_date not in self.indicator_data:
                                self.indicator_data[current_date] = {}
                            self.indicator_data[current_date]['boll_top'] = float(self.boll.top[0])
                            self.indicator_data[current_date]['boll_bot'] = float(self.boll.bot[0])
                            self.indicator_data[current_date]['boll_mid'] = float(self.boll.mid[0])
                    except Exception as e:
                        logger.debug(f"记录资金曲线或指标数据时出错: {e}")
            
            # 添加策略（使用包装后的策略类）
            cerebro.addstrategy(EquityRecorderStrategy, **strategy_params)
            
            # 执行回测
            logger.info("开始执行回测...")
            logger.info(f"数据量: {len(data)} 条, 策略: {strategy_class.__name__}, 参数: {strategy_params}")
            
            results = cerebro.run()
            result = results[0]
            
            # 检查策略是否执行
            final_value = cerebro.broker.getvalue()
            logger.info(f"策略执行完成，最终资金: {final_value:.2f}")
            
            # 检查持仓状态
            position = cerebro.broker.getposition(datafeed)
            if position.size != 0:
                logger.info(f" 回测结束时仍有持仓: {position.size}")
                # 如果有持仓，尝试在最后一天强制平仓（用于计算最终收益）
                # 注意：这不会影响TradeAnalyzer的统计，因为它只统计已完成的交易对
                final_price = datafeed.close[0]
                logger.info(f" 持仓价值: {position.size * final_price:.2f} USDT (价格: {final_price:.2f})")
            
            # 提取绩效指标
            performance = self._extract_performance(result, cerebro)
            
            # 提取交易记录
            trades = self._extract_trades(result)
            
            # 提取资金曲线数据（用于绘制图表）
            equity_curve = self._extract_equity_curve(result, cerebro, datafeed)
            
            # 调试：检查资金曲线数据
            if isinstance(equity_curve, pd.DataFrame) and not equity_curve.empty:
                logger.info(f"资金曲线数据: {len(equity_curve)} 条, 范围: {equity_curve.iloc[0, 0]:.2f} - {equity_curve.iloc[-1, 0]:.2f}")
            elif isinstance(equity_curve, list):
                logger.info(f"资金曲线数据: {len(equity_curve)} 条记录")
            else:
                logger.warning(f"资金曲线数据为空或格式不正确: {type(equity_curve)}")
            
            # 提取指标数据
            indicator_data = {}
            if hasattr(result, 'indicator_data'):
                indicator_data = result.indicator_data
            
            # 统计买卖信号数量（用于调试）
            buy_count = len([t for t in trades if t.get("type") == "buy_signal"])
            sell_count = len([t for t in trades if t.get("type") == "sell_signal"])
            if buy_count > 0 or sell_count > 0:
                logger.info(f" 提取到交易信号: 买入={buy_count}, 卖出={sell_count}")
                # 打印前几个信号用于调试
                if buy_count > 0:
                    first_buy = [t for t in trades if t.get("type") == "buy_signal"][0]
                    logger.debug(f"第一个买入信号: {first_buy}")
                if sell_count > 0:
                    first_sell = [t for t in trades if t.get("type") == "sell_signal"][0]
                    logger.debug(f"第一个卖出信号: {first_sell}")
            else:
                # 只有在没有提取到任何信号时才显示警告
                # 检查策略是否有order_history
                if hasattr(result, 'order_history'):
                    order_history_len = len(result.order_history) if result.order_history else 0
                    if order_history_len > 0:
                        logger.warning(f" 策略order_history有{order_history_len}条记录，但未提取到信号。可能原因：日期格式问题或数据范围不匹配")
                        # 打印第一条记录用于调试
                        if result.order_history:
                            logger.debug(f"第一条order_history记录: {result.order_history[0]}")
                    else:
                        logger.warning(f" 策略order_history存在但为空: 0条记录。可能原因：订单未完成或notify_order未调用")
                else:
                    logger.warning(" 策略没有order_history属性，可能订单未执行")
            
            total_trades = performance.get('total_trades', 0)
            logger.info(f"回测完成: 总收益率={performance.get('total_return', 0):.2%}, 交易次数={total_trades}")
            
            if total_trades == 0:
                logger.warning(" 交易次数为0，但策略可能产生了信号。可能原因：")
                logger.warning("  1. 交易未完整完成（只有买入没有卖出，或反之）")
                logger.warning("  2. TradeAnalyzer只统计完整交易对（开仓+平仓）")
            
            # 不再使用缓存的performance数据，总是使用新提取的完整数据
            # 这样可以确保所有绩效指标都是最新的、完整的
            # 缓存只用于快速判断是否需要重新运行回测，但不用于覆盖新数据
            # if use_cached_performance and cached_result:
            #     performance = cached_result.get("performance", performance)
            #     trades = cached_result.get("trades", trades)
            
            # 计算手续费（从broker获取）
            total_commission = 0
            initial_value = self.config.initial_cash
            final_value = cerebro.broker.getvalue()
            
            try:
                # 尝试从broker获取总手续费
                # Backtrader的broker不直接提供总手续费，需要通过订单计算
                if hasattr(result, 'broker') and hasattr(result.broker, 'orders'):
                    for order in result.broker.orders:
                        if hasattr(order, 'executed') and hasattr(order.executed, 'comm'):
                            total_commission += abs(order.executed.comm)
            except Exception as e:
                logger.debug(f"计算手续费时出错: {e}")
            
            # 如果没有从订单中获取到手续费，使用估算方法
            if total_commission == 0:
                # 简化估算：手续费 = 交易次数 * 平均交易金额 * 手续费率
                # 这里使用一个简化的估算方法
                total_trades = performance.get('total_trades', 0)
                if total_trades > 0:
                    # 估算：每次交易的平均金额约为初始资金的某个比例
                    avg_trade_size = initial_value * 0.5  # 假设每次交易使用50%资金
                    total_commission = total_trades * 2 * avg_trade_size * self.config.commission  # 买入+卖出各一次
                else:
                    total_commission = 0
            
            # 将equity_curve转换为字典格式以便序列化
            equity_curve_dict = None
            if isinstance(equity_curve, pd.DataFrame) and not equity_curve.empty:
                # 转换为字典列表格式
                equity_df = equity_curve.reset_index()
                # 确保列名正确
                if 'date' not in equity_df.columns:
                    if len(equity_df.columns) == 2:
                        equity_df.columns = ['date', 'equity']
                    else:
                        equity_df.columns = ['date', 'value'] if 'value' in str(equity_df.columns[1]).lower() else ['date', 'equity']
                equity_curve_dict = equity_df.to_dict('records')
                logger.info(f"资金曲线转换为字典: {len(equity_curve_dict)} 条记录")
            elif isinstance(equity_curve, list):
                equity_curve_dict = equity_curve
                logger.info(f"资金曲线已是列表格式: {len(equity_curve_dict)} 条记录")
            
            result = {
                "success": True,
                "performance": performance,
                "trades": trades,
                "final_value": cerebro.broker.getvalue(),
                "initial_value": self.config.initial_cash,
                "total_commission": total_commission,
                "commission_rate": self.config.commission,
                "equity_curve": equity_curve_dict,
                "indicator_data": indicator_data
                # 注意：不再保存cerebro对象，因为Backtrader的plot()在macOS上无法工作
                # 所有图表功能已迁移到Plotly
            }
            
            # 保存缓存（不包含cerebro对象，因为无法序列化）
            if use_cache and symbol and interval and start_time and end_time and not use_cached_performance:
                cache_result = {
                    "success": True,
                    "performance": performance,
                    "trades": trades,
                    "final_value": cerebro.broker.getvalue(),
                    "initial_value": self.config.initial_cash
                }
                self.cache_manager.set(
                    symbol=symbol,
                    interval=interval,
                    start_time=str(start_time),
                    end_time=str(end_time),
                    strategy_name=strategy_class.__name__,
                    strategy_params=strategy_params or {},
                    backtest_params=self.config.to_dict(),
                    result=cache_result
                )
            
            return result
            
        except Exception as e:
            error_info = self.exception_handler.handle_strategy_error(
                strategy_name=strategy_class.__name__,
                error=e,
                sanitize=True
            )
            logger.error(f"回测失败: {e}")
            return {
                "success": False,
                "error": error_info["error_message"],
                "performance": {},
                "trades": []
            }
    
    def _extract_performance(
        self,
        result: bt.Strategy,
        cerebro: bt.Cerebro
    ) -> Dict[str, Any]:
        """
        提取绩效指标
        
        Args:
            result: 策略执行结果
            cerebro: Cerebro实例
        
        Returns:
            dict: 绩效指标字典
        """
        performance = {}
        
        try:
            # 基础指标
            final_value = cerebro.broker.getvalue()
            initial_value = self.config.initial_cash
            total_return = (final_value - initial_value) / initial_value
            performance["total_return"] = total_return
            
            # 调试信息
            logger.debug(f"初始资金: {initial_value}, 最终资金: {final_value}, 收益率: {total_return:.4%}")
            
            # 从分析器提取指标
            if hasattr(result, 'analyzers'):
                # 调试：打印所有分析器
                logger.debug(f"可用分析器: {list(result.analyzers.keys()) if hasattr(result.analyzers, 'keys') else 'N/A'}")
                
                # Returns分析器
                if hasattr(result.analyzers, 'returns'):
                    ret_analyzer = result.analyzers.returns.get_analysis()
                    # rnorm100是年化收益率（百分比），需要除以100
                    rnorm100 = ret_analyzer.get('rnorm100', 0)
                    if rnorm100 is not None:
                        annual_return = rnorm100 / 100
                    else:
                        # 尝试使用rnorm（不是百分比）
                        annual_return = ret_analyzer.get('rnorm', 0) or 0
                    performance["annual_return"] = annual_return if annual_return is not None else 0
                
                # Sharpe比率
                if hasattr(result.analyzers, 'sharpe'):
                    sharpe_analyzer = result.analyzers.sharpe.get_analysis()
                    sharpe_value = sharpe_analyzer.get('sharperatio', 0)
                    performance["sharpe_ratio"] = sharpe_value if sharpe_value is not None else 0
                
                # 最大回撤
                if hasattr(result.analyzers, 'drawdown'):
                    dd_analyzer = result.analyzers.drawdown.get_analysis()
                    drawdown_value = dd_analyzer.get('max', {}).get('drawdown', 0)
                    if drawdown_value is not None:
                        performance["max_drawdown"] = abs(drawdown_value) / 100
                    else:
                        performance["max_drawdown"] = 0
                    max_dd_period = dd_analyzer.get('max', {}).get('len', 0)
                    performance["max_drawdown_period"] = max_dd_period if max_dd_period is not None else 0
                
                # 交易分析
                # Backtrader的分析器使用属性访问，不是字典
                if hasattr(result.analyzers, 'trades'):
                    try:
                        trades_analyzer = result.analyzers.trades.get_analysis()
                        
                        # 提取交易统计（TradeAnalyzer的数据结构）
                        # TradeAnalyzer的结构: {'total': {'total': X, 'open': Y, 'closed': Z}, ...}
                        total_dict = trades_analyzer.get('total', {})
                        # 使用closed数量（已完成的交易对），这是真正完成的交易
                        closed_trades = total_dict.get('closed', 0)
                        total_trades = total_dict.get('total', 0)  # 保留total用于调试
                        won_trades = trades_analyzer.get('won', {}).get('total', 0)
                        lost_trades = trades_analyzer.get('lost', {}).get('total', 0)
                        
                        # 调试：打印完整分析结果
                        open_trades = total_dict.get('open', 0)
                        logger.debug(f"TradeAnalyzer: total={total_trades}, closed={closed_trades}, open={open_trades}")
                        logger.info(f"交易统计: 已关闭交易={closed_trades}, 盈利={won_trades}, 亏损={lost_trades}")
                        if open_trades > 0:
                            logger.info(f" 提示: 有 {open_trades} 个未完成的交易（只有买入没有卖出，或反之）。TradeAnalyzer只统计已完成的交易对。")
                        
                        # 使用closed数量作为总交易次数（只统计完整交易对）
                        performance["total_trades"] = closed_trades
                        performance["win_rate"] = won_trades / closed_trades if closed_trades > 0 else 0
                        
                        # 盈亏比
                        won_pnl = trades_analyzer.get('won', {}).get('pnl', {})
                        lost_pnl = trades_analyzer.get('lost', {}).get('pnl', {})
                        avg_win = won_pnl.get('average', 0) if won_pnl else 0
                        avg_loss = abs(lost_pnl.get('average', 0)) if lost_pnl else 0
                        performance["profit_loss_ratio"] = avg_win / avg_loss if avg_loss > 0 else 0
                    except Exception as e:
                        logger.error(f"提取TradeAnalyzer数据失败: {e}")
                        import traceback
                        logger.debug(traceback.format_exc())
                        performance["total_trades"] = 0
                        performance["win_rate"] = 0
                        performance["profit_loss_ratio"] = 0
                else:
                    logger.warning("TradeAnalyzer不存在，无法提取交易统计")
                    performance["total_trades"] = 0
                    performance["win_rate"] = 0
                    performance["profit_loss_ratio"] = 0
                
                # SQN（系统质量指标）
                if hasattr(result.analyzers, 'sqn'):
                    sqn_analyzer = result.analyzers.sqn.get_analysis()
                    performance["sqn"] = sqn_analyzer.get('sqn', 0)
        
        except Exception as e:
            logger.warning(f"提取绩效指标时出错: {e}")
        
        return performance
    
    def _extract_trades(self, result: bt.Strategy) -> List[Dict[str, Any]]:
        """
        提取交易记录和买卖信号
        
        Args:
            result: 策略执行结果
        
        Returns:
            List[dict]: 交易记录列表，包含买卖信号
        """
        trades = []
        buy_signals = []
        sell_signals = []
        
        try:
            # 方法1: 从策略的order_history中提取（如果策略记录了）
            if hasattr(result, 'order_history') and result.order_history:
                for order_info in result.order_history:
                    signal = {
                        "date": order_info.get("date"),
                        "price": float(order_info.get("price", 0)),
                        "size": float(order_info.get("size", 0)),
                        "type": order_info.get("type", "buy")
                    }
                    if signal["type"] == "buy":
                        buy_signals.append(signal)
                    else:
                        sell_signals.append(signal)
                if len(buy_signals) > 0 or len(sell_signals) > 0:
                    logger.info(f"从策略order_history提取到 {len(buy_signals)} 个买入, {len(sell_signals)} 个卖出")
            
            # 方法2: 从broker的订单中提取（备选方案）
            if not buy_signals and not sell_signals:
                broker = result.broker if hasattr(result, 'broker') else None
                if broker and hasattr(broker, 'orders'):
                    # 获取所有已完成的订单
                    for order in broker.orders:
                        try:
                            if hasattr(order, 'status') and order.status == order.Completed:
                                # 获取订单日期和价格
                                order_date = None
                                order_price = None
                                
                                # 获取执行价格
                                if hasattr(order, 'executed') and hasattr(order.executed, 'price'):
                                    order_price = order.executed.price
                                
                                # 尝试从订单的data中获取日期
                                # 注意：订单完成后，data.datetime可能无法直接访问
                                # 所以优先使用策略中记录的order_history
                                if hasattr(order, 'data') and order.data is not None:
                                    try:
                                        # 尝试获取订单执行时的日期
                                        # 但这个方法在订单完成后可能不可靠
                                        if hasattr(order.data, 'datetime'):
                                            # 使用datetime的date方法
                                            dt = order.data.datetime
                                            if hasattr(dt, 'date'):
                                                try:
                                                    order_date = dt.date(0)
                                                except:
                                                    pass
                                    except Exception as date_error:
                                        logger.debug(f"从订单data提取日期失败: {date_error}")
                                
                                # 如果仍然没有日期，尝试从策略的数据源中获取（最后备选）
                                if not order_date and hasattr(result, 'datas') and result.datas:
                                    try:
                                        # 使用策略的主数据源
                                        data = result.datas[0]
                                        if hasattr(data, 'datetime') and hasattr(data.datetime, 'date'):
                                            # 获取当前数据点的日期（订单执行时的日期）
                                            order_date = data.datetime.date(0)
                                    except:
                                        pass
                                
                                if order_date and order_price:
                                    signal = {
                                        "date": order_date,
                                        "price": float(order_price),
                                        "size": float(order.executed.size) if hasattr(order, 'executed') and hasattr(order.executed, 'size') else 0,
                                        "type": "buy" if order.isbuy() else "sell"
                                    }
                                    if signal["type"] == "buy":
                                        buy_signals.append(signal)
                                    else:
                                        sell_signals.append(signal)
                                else:
                                    logger.debug(f"订单信息不完整: date={order_date}, price={order_price}")
                        except Exception as order_error:
                            logger.debug(f"提取订单信息时出错: {order_error}")
                            continue
            
            # 添加买卖信号到trades列表
            if buy_signals or sell_signals:
                # 确保类型字段统一为 buy_signal 和 sell_signal
                for sig in buy_signals:
                    sig_copy = sig.copy()
                    sig_copy["type"] = "buy_signal"
                    trades.append(sig_copy)
                for sig in sell_signals:
                    sig_copy = sig.copy()
                    sig_copy["type"] = "sell_signal"
                    trades.append(sig_copy)
                logger.info(f"提取到 {len(buy_signals)} 个买入信号, {len(sell_signals)} 个卖出信号")
            
            # 如果无法从订单中提取，尝试从TradeAnalyzer获取汇总信息
            if not trades:
                if hasattr(result, 'analyzers') and 'trades' in result.analyzers:
                    trades_analyzer = result.analyzers.trades.get_analysis()
                    total = trades_analyzer.get('total', {})
                    trades.append({
                        "type": "summary",
                        "total_trades": total.get('total', 0),
                        "won": total.get('won', 0),
                        "lost": total.get('lost', 0),
                        "pnl_net": total.get('pnl', {}).get('net', 0)
                    })
        
        except Exception as e:
            logger.warning(f"提取交易记录时出错: {e}")
            import traceback
            logger.debug(traceback.format_exc())
        
        return trades
    
    def _extract_equity_curve(self, result: bt.Strategy, cerebro: bt.Cerebro, datafeed) -> pd.DataFrame:
        """
        提取资金曲线数据
        
        Args:
            result: 策略执行结果
            cerebro: Cerebro实例
            datafeed: 数据源
            
        Returns:
            pd.DataFrame: 资金曲线数据（日期、资金）
        """
        equity_data = []
        
        try:
            # 方法1: 从观察者获取资金曲线数据
            dates = []
            values = []
            initial_value = self.config.initial_cash
            
            # 方法1: 从Backtrader的Value观察者获取资金曲线数据（最准确）
            # Value观察者记录每个bar的portfolio value
            try:
                # 获取观察者 - observers是一个字典，key是观察者的名称
                if hasattr(result, 'observers'):
                    observers_dict = result.observers
                    # 尝试获取value观察者
                    if hasattr(observers_dict, 'value'):
                        value_observers = observers_dict.value
                        # value_observers可能是一个列表
                        if isinstance(value_observers, list) and len(value_observers) > 0:
                            value_observer = value_observers[0]
                        else:
                            value_observer = value_observers
                        
                        # 获取value line和datetime line
                        if hasattr(value_observer, 'lines'):
                            if hasattr(value_observer.lines, 'value'):
                                value_line = value_observer.lines.value
                                # 获取数据长度
                                value_len = len(value_line)
                                
                                # 遍历所有数据点
                                for i in range(value_len):
                                    try:
                                        # 获取日期 - 优先使用datafeed的日期
                                        date = datafeed.datetime.date(i)
                                        
                                        # 获取资金值
                                        value = float(value_line[i])
                                        dates.append(date)
                                        values.append(value)
                                    except Exception as e:
                                        logger.debug(f"从Value观察者提取数据点{i}时出错: {e}")
                                        continue
                                
                                if dates and values:
                                    logger.info(f"从Backtrader Value观察者提取到 {len(dates)} 个资金曲线数据点")
            except Exception as e:
                logger.debug(f"尝试从Value观察者获取数据时出错: {e}")
                import traceback
                logger.debug(traceback.format_exc())
            
            # 方法2: 优先使用策略记录的equity_curve_data（更可靠）
            # 检查策略是否记录了资金曲线
            if hasattr(result, 'equity_curve_data') and result.equity_curve_data:
                logger.info(f"策略equity_curve_data有 {len(result.equity_curve_data)} 条记录")
                # 优先使用策略数据，因为它记录了每个bar的实际资金值
                dates = []
                values = []
                for item in result.equity_curve_data:
                    try:
                        date = item.get('date')
                        value = item.get('value')
                        if date and value is not None:
                            # 转换为pandas日期
                            if isinstance(date, str):
                                date = pd.to_datetime(date).date()
                            elif hasattr(date, 'date'):
                                date = date.date()
                            dates.append(date)
                            values.append(float(value))
                    except Exception as e:
                        logger.debug(f"处理资金曲线数据项时出错: {e}")
                        continue
                if dates and values:
                    logger.info(f"从策略equity_curve_data提取到 {len(dates)} 个资金曲线数据点, 范围: {min(values):.2f} - {max(values):.2f}")
            
            # 方法3: 如果以上方法都不可用，使用简化方法（基于交易记录估算）
            if not dates or not values:
                # 使用交易记录和价格数据估算资金曲线
                dates = []
                values = []
                current_value = initial_value
                
                # 获取交易记录
                trades = []
                if hasattr(result, 'order_history') and result.order_history:
                    trades = result.order_history
                
                # 按日期排序交易
                sorted_trades = sorted(trades, key=lambda x: x.get('date', pd.Timestamp.min))
                
                # 遍历数据源，计算每日资金
                trade_idx = 0
                for i in range(len(datafeed)):
                    try:
                        date = datafeed.datetime.date(i)
                        dates.append(date)
                        
                        # 检查是否有交易发生在这一天或之前
                        while trade_idx < len(sorted_trades):
                            trade = sorted_trades[trade_idx]
                            trade_date = trade.get('date')
                            if isinstance(trade_date, str):
                                trade_date = pd.to_datetime(trade_date).date()
                            elif hasattr(trade_date, 'date'):
                                trade_date = trade_date.date()
                            
                            if trade_date <= date:
                                # 更新资金（简化：假设交易立即影响资金）
                                trade_type = trade.get('type')
                                trade_price = trade.get('price', 0)
                                trade_size = trade.get('size', 0)
                                
                                if trade_type == 'buy':
                                    # 买入：减少现金，增加持仓
                                    cost = trade_price * trade_size
                                    commission = cost * self.config.commission
                                    current_value = current_value - cost - commission
                                elif trade_type == 'sell':
                                    # 卖出：增加现金，减少持仓
                                    revenue = trade_price * trade_size
                                    commission = revenue * self.config.commission
                                    current_value = current_value + revenue - commission
                                
                                trade_idx += 1
                            else:
                                break
                        
                        # 如果有持仓，计算持仓价值
                        if hasattr(result, 'position') and result.position.size != 0:
                            current_price = datafeed.close[i]
                            position_value = result.position.size * current_price
                            # 资金 = 现金 + 持仓价值
                            total_value = current_value + position_value
                        else:
                            total_value = current_value
                        
                        values.append(total_value)
                    except Exception as e:
                        logger.debug(f"估算资金曲线数据点{i}时出错: {e}")
                        continue
                
                # 如果还是没有数据，使用线性插值作为最后手段
                if not dates or not values:
                    final_value = cerebro.broker.getvalue()
                    for i in range(len(datafeed)):
                        try:
                            date = datafeed.datetime.date(i)
                            dates.append(date)
                            if i == 0:
                                values.append(initial_value)
                            elif i == len(datafeed) - 1:
                                values.append(final_value)
                            else:
                                # 线性插值
                                progress = i / (len(datafeed) - 1)
                                value = initial_value + (final_value - initial_value) * progress
                                values.append(value)
                        except Exception as e:
                            logger.debug(f"线性插值资金曲线数据点{i}时出错: {e}")
                            continue
            
            if dates and values and len(dates) == len(values):
                equity_data = pd.DataFrame({
                    'date': dates,
                    'equity': values
                })
                equity_data.set_index('date', inplace=True)
                logger.info(f" 提取资金曲线: {len(dates)} 个数据点")
            else:
                logger.warning(f" 资金曲线数据不完整: dates={len(dates)}, values={len(values)}")
                equity_data = pd.DataFrame(columns=['equity'])
        except Exception as e:
            logger.warning(f"提取资金曲线时出错: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            equity_data = pd.DataFrame(columns=['equity'])
        
        return equity_data
    
    def _generate_cache_key(
        self,
        data: pd.DataFrame,
        strategy_class: Type[bt.Strategy],
        strategy_params: Dict[str, Any]
    ) -> str:
        """
        生成缓存键
        
        Args:
            data: K线数据
            strategy_class: 策略类
            strategy_params: 策略参数
        
        Returns:
            str: 缓存键
        """
        # 使用数据的时间范围、策略类和参数生成唯一键
        key_data = {
            "data_start": str(data.index[0]) if not data.empty else "",
            "data_end": str(data.index[-1]) if not data.empty else "",
            "data_length": len(data),
            "strategy": strategy_class.__name__,
            "strategy_params": json.dumps(strategy_params, sort_keys=True),
            "config": json.dumps(self.config.to_dict(), sort_keys=True)
        }
        
        key_string = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_string.encode()).hexdigest()

