"""
Streamlit可视化界面
遵循设计原则：分层解耦、快速失败
"""

# 必须在导入matplotlib之前设置后端（macOS兼容性）
import os
import sys
os.environ['MPLBACKEND'] = 'Agg'  # 通过环境变量强制设置

import matplotlib
matplotlib.use('Agg', force=True)  # 强制使用非交互式后端，避免macOS线程问题

import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from typing import Optional, Dict, Any, List
from datetime import datetime, date, timedelta
import logging
import matplotlib.pyplot as plt

from ..data.binance_fetcher import BinanceDataFetcher
from ..backtest.backtrader_config import BacktraderConfig
from ..backtest.backtrader_engine import BacktraderEngine
from ..strategy.strategy_templates import get_strategy_class
from ..system.proxy_manager import ProxyManager
from ..system.exception_handler import ExceptionHandler

logger = logging.getLogger(__name__)


class StreamlitUI:
    """
    Streamlit界面管理器
    负责界面布局、参数配置、图表展示
    """
    
    def __init__(self):
        """初始化UI"""
        self.proxy_manager = ProxyManager()
        self.exception_handler = ExceptionHandler()
        self.data_fetcher: Optional[BinanceDataFetcher] = None
        self.engine: Optional[BacktraderEngine] = None
        
        # 初始化session state
        if 'trading_pairs' not in st.session_state:
            st.session_state.trading_pairs = {}
        if 'klines_data' not in st.session_state:
            st.session_state.klines_data = {}
        if 'backtest_result' not in st.session_state:
            st.session_state.backtest_result = None
        if 'comparison_results' not in st.session_state:
            st.session_state.comparison_results = []
    
    def setup_page(self):
        """设置页面配置"""
        st.set_page_config(
            page_title="Crypto Trader回测平台",
            page_icon="",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # 自定义CSS
        st.markdown("""
        <style>
        .main-header {
            font-size: 2.5rem;
            font-weight: bold;
            color: #1f77b4;
            margin-bottom: 1rem;
        }
        .metric-card {
            background-color: #f0f2f6;
            padding: 1rem;
            border-radius: 0.5rem;
            margin: 0.5rem 0;
        }
        </style>
        """, unsafe_allow_html=True)
    
    def render_header(self):
        """渲染页面头部"""
        st.markdown('<div class="main-header"> Crypto Trader回测与可视化分析平台</div>', unsafe_allow_html=True)
        st.markdown("---")
    
    def render_sidebar(self) -> Dict[str, Any]:
        """
        渲染侧边栏（参数配置区）
        
        Returns:
            dict: 配置参数字典
        """
        st.sidebar.header(" 回测配置")
        
        # 市场选择
        market = st.sidebar.selectbox(
            "市场类型",
            ["spot", "futures"],
            format_func=lambda x: "现货" if x == "spot" else "永续合约",
            index=0
        )
        
        # 交易对选择
        st.sidebar.subheader("交易对选择")
        
        # 加载交易对按钮
        if st.sidebar.button(" 加载交易对", use_container_width=True):
            with st.sidebar:
                with st.spinner("加载交易对中..."):
                    try:
                        import asyncio
                        if self.data_fetcher is None:
                            self.data_fetcher = BinanceDataFetcher(self.proxy_manager)
                        
                        pairs = st.session_state.get(f'trading_pairs_{market}', [])
                        if not pairs:
                            # 异步调用处理
                            try:
                                loop = asyncio.get_event_loop()
                            except RuntimeError:
                                loop = asyncio.new_event_loop()
                                asyncio.set_event_loop(loop)
                            
                            pairs = loop.run_until_complete(
                                self.data_fetcher.get_trading_pairs(market)
                            )
                            st.session_state[f'trading_pairs_{market}'] = pairs
                        
                        st.success(f"加载完成，共 {len(pairs)} 个交易对")
                    except Exception as e:
                        error_msg = self.exception_handler.format_user_friendly_error({
                            "error_type": "API_ERROR",
                            "error_message": str(e)
                        })
                        st.error(error_msg)
        
        # 显示交易对列表
        pairs = st.session_state.get(f'trading_pairs_{market}', [])
        if pairs:
            selected_pair = st.sidebar.selectbox(
                "选择交易对",
                pairs,
                index=0 if pairs else None
            )
        else:
            selected_pair = st.sidebar.text_input(
                "输入交易对",
                value="BTC/USDT",
                placeholder="例如: BTC/USDT"
            )
        
        # 时间范围配置
        st.sidebar.subheader("时间范围")
        col1, col2 = st.sidebar.columns(2)
        with col1:
            start_date = st.date_input(
                "起始日期",
                value=date.today() - timedelta(days=365),
                max_value=date.today()
            )
        with col2:
            end_date = st.date_input(
                "结束日期",
                value=date.today(),
                max_value=date.today()
            )
        
        # 时间周期
        interval = st.sidebar.selectbox(
            "时间周期",
            ["1m", "5m", "15m", "30m", "1h", "2h", "4h", "1d", "1w", "1M"],
            index=7  # 默认1d
        )
        
        # 回测参数配置
        st.sidebar.subheader("回测参数")
        initial_cash = st.sidebar.number_input(
            "初始资金 (USDT)",
            min_value=100.0,
            value=10000.0,
            step=1000.0
        )
        
        commission = st.sidebar.slider(
            "手续费率 (%)",
            min_value=0.0,
            max_value=1.0,
            value=0.1,
            step=0.01
        ) / 100  # 转换为小数
        
        slippage = st.sidebar.slider(
            "滑点 (%)",
            min_value=0.0,
            max_value=1.0,
            value=0.0,
            step=0.01
        ) / 100
        
        # 杠杆（仅合约市场）
        leverage = None
        if market == "futures":
            leverage = st.sidebar.slider(
                "杠杆倍数",
                min_value=1,
                max_value=125,
                value=1,
                step=1
            )
        
        # 策略选择
        st.sidebar.subheader("策略配置")
        strategy_type = st.sidebar.radio(
            "策略类型",
            ["内置策略", "自定义策略"],
            index=0
        )
        
        strategy_name = None
        strategy_params = {}
        custom_strategy_code = None
        
        if strategy_type == "内置策略":
            strategy_name = st.sidebar.selectbox(
                "选择策略",
                ["MA", "EMA", "RSI", "MACD", "BOLL"],
                index=0
            )
            # 策略参数（根据策略类型动态显示）
            strategy_params = self._get_strategy_params_ui(strategy_name)
        else:
            # 自定义策略
            if "custom_strategy_template" in st.session_state:
                custom_strategy_code = st.session_state.custom_strategy_template
            else:
                from ..strategy.custom_strategy import StrategyExecutor
                executor = StrategyExecutor()
                custom_strategy_code = executor.get_strategy_template()
            
            custom_strategy_code = st.sidebar.text_area(
                "策略代码",
                value=custom_strategy_code,
                height=200,
                help="输入Backtrader策略代码，必须继承自bt.Strategy"
            )
            
            if st.sidebar.button("📝 加载模板", use_container_width=True):
                from ..strategy.custom_strategy import StrategyExecutor
                executor = StrategyExecutor()
                st.session_state.custom_strategy_template = executor.get_strategy_template()
                st.rerun()
        
        # 风险管理配置
        st.sidebar.subheader("风险管理")
        risk_params = self._get_risk_params_ui()
        
        return {
            "market": market,
            "symbol": selected_pair,
            "start_date": start_date,
            "end_date": end_date,
            "interval": interval,
            "initial_cash": initial_cash,
            "commission": commission,
            "slippage": slippage,
            "leverage": leverage,
            "strategy_name": strategy_name,
            "strategy_params": strategy_params,
            "strategy_type": strategy_type,
            "custom_strategy_code": custom_strategy_code,
            "risk_params": risk_params
        }
    
    def _get_strategy_params_ui(self, strategy_name: str) -> Dict[str, Any]:
        """根据策略类型显示参数配置UI"""
        params = {}
        
        if strategy_name in ["MA", "EMA"]:
            col1, col2 = st.sidebar.columns(2)
            with col1:
                params["fast_period"] = st.number_input(
                    "短期周期",
                    min_value=1,
                    value=10,
                    key=f"fast_{strategy_name}"
                )
            with col2:
                params["slow_period"] = st.number_input(
                    "长期周期",
                    min_value=1,
                    value=30,
                    key=f"slow_{strategy_name}"
                )
        
        elif strategy_name == "RSI":
            params["rsi_period"] = st.sidebar.number_input(
                "RSI周期",
                min_value=1,
                value=14,
                key="rsi_period"
            )
            col1, col2 = st.sidebar.columns(2)
            with col1:
                params["rsi_oversold"] = st.number_input(
                    "超卖阈值",
                    min_value=0,
                    max_value=50,
                    value=30,
                    key="rsi_oversold"
                )
            with col2:
                params["rsi_overbought"] = st.number_input(
                    "超买阈值",
                    min_value=50,
                    max_value=100,
                    value=70,
                    key="rsi_overbought"
                )
        
        elif strategy_name == "MACD":
            col1, col2, col3 = st.sidebar.columns(3)
            with col1:
                params["macd_fast"] = st.number_input("快线", min_value=1, value=12, key="macd_fast")
            with col2:
                params["macd_slow"] = st.number_input("慢线", min_value=1, value=26, key="macd_slow")
            with col3:
                params["macd_signal"] = st.number_input("信号线", min_value=1, value=9, key="macd_signal")
        
        elif strategy_name == "BOLL":
            col1, col2 = st.sidebar.columns(2)
            with col1:
                params["boll_period"] = st.number_input("周期", min_value=1, value=20, key="boll_period")
            with col2:
                params["boll_dev"] = st.number_input("标准差倍数", min_value=0.1, value=2.0, step=0.1, key="boll_dev")
        
        return params
    
    def _get_risk_params_ui(self) -> Dict[str, Any]:
        """获取风险管理参数配置UI"""
        risk_params = {}
        
        # 止盈止损
        use_stop_loss = st.sidebar.checkbox("启用止损", value=False)
        if use_stop_loss:
            stop_loss_type = st.sidebar.radio(
                "止损类型",
                ["百分比", "固定价格"],
                horizontal=True
            )
            if stop_loss_type == "百分比":
                risk_params["stop_loss_pct"] = st.sidebar.slider(
                    "止损百分比 (%)",
                    min_value=0.1,
                    max_value=20.0,
                    value=3.0,
                    step=0.1
                ) / 100
            else:
                risk_params["stop_loss_price"] = st.sidebar.number_input(
                    "止损价格",
                    min_value=0.0,
                    value=0.0,
                    step=0.01
                )
        
        use_take_profit = st.sidebar.checkbox("启用止盈", value=False)
        if use_take_profit:
            take_profit_type = st.sidebar.radio(
                "止盈类型",
                ["百分比", "固定价格"],
                horizontal=True
            )
            if take_profit_type == "百分比":
                risk_params["take_profit_pct"] = st.sidebar.slider(
                    "止盈百分比 (%)",
                    min_value=0.1,
                    max_value=50.0,
                    value=5.0,
                    step=0.1
                ) / 100
            else:
                risk_params["take_profit_price"] = st.sidebar.number_input(
                    "止盈价格",
                    min_value=0.0,
                    value=0.0,
                    step=0.01
                )
        
        # 仓位管理
        position_type = st.sidebar.selectbox(
            "仓位管理",
            ["全部资金", "固定手数", "资金比例", "最大持仓数"],
            index=0
        )
        
        if position_type == "固定手数":
            risk_params["position_size"] = st.sidebar.number_input(
                "固定手数",
                min_value=0.01,
                value=1.0,
                step=0.01
            )
        elif position_type == "资金比例":
            risk_params["position_pct"] = st.sidebar.slider(
                "资金比例 (%)",
                min_value=1,
                max_value=100,
                value=100,
                step=1
            ) / 100
        elif position_type == "最大持仓数":
            risk_params["max_positions"] = st.sidebar.number_input(
                "最大持仓数",
                min_value=1,
                value=1,
                step=1
            )
        
        return risk_params
    
    def render_main_content(self, config: Dict[str, Any]):
        """
        渲染主内容区（结果展示）
        
        Args:
            config: 配置参数字典
        """
        # 功能标签页
        tab1, tab2, tab3, tab4 = st.tabs([" 单策略回测", " 参数优化", " 策略对比", " 高级功能"])
        
        with tab1:
            self._render_single_backtest(config)
        
        with tab2:
            self._render_grid_search(config)
        
        with tab3:
            self._render_strategy_comparison(config)
        
        with tab4:
            self._render_advanced_features(config)
    
    def _render_single_backtest(self, config: Dict[str, Any]):
        """渲染单策略回测界面"""
        # 操作按钮
        col1, col2, col3 = st.columns([1, 1, 4])
        with col1:
            if st.button(" 开始回测", use_container_width=True, type="primary"):
                self._run_backtest(config)
        with col2:
            if st.button(" 重置", use_container_width=True):
                st.session_state.backtest_result = None
                st.rerun()
        
        st.markdown("---")
        
        # 显示回测结果
        if st.session_state.backtest_result:
            self._render_backtest_results(st.session_state.backtest_result, config)
        else:
            st.info(" 请配置参数并点击「开始回测」")
    
    def _run_backtest(self, config: Dict[str, Any]):
        """执行回测"""
        try:
            # 初始化组件
            if self.data_fetcher is None:
                self.data_fetcher = BinanceDataFetcher(self.proxy_manager)
            
            # 显示进度
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # 1. 获取K线数据
            status_text.text("📥 正在获取K线数据...")
            progress_bar.progress(20)
            
            # 异步调用处理
            import asyncio
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            
            df = loop.run_until_complete(
                self.data_fetcher.get_klines_batch(
                    symbol=config["symbol"],
                    interval=config["interval"],
                    start_time=config["start_date"],
                    end_time=config["end_date"],
                    market=config["market"]
                )
            )
            
            # 关闭HTTP会话
            try:
                loop.run_until_complete(self.data_fetcher.close())
            except:
                pass
            
            if df.empty:
                st.error(" 未能获取到K线数据，请检查网络连接和代理设置")
                return
            
            # 2. 创建回测配置
            status_text.text(" 正在配置回测参数...")
            progress_bar.progress(40)
            
            backtest_config = BacktraderConfig(
                initial_cash=config["initial_cash"],
                commission=config["commission"],
                slippage=config["slippage"],
                leverage=config.get("leverage")
            )
            
            # 3. 获取策略类
            status_text.text(" 正在加载策略...")
            progress_bar.progress(60)
            
            strategy_class = None
            if config.get("strategy_type") == "自定义策略":
                # 自定义策略
                custom_code = config.get("custom_strategy_code")
                if not custom_code:
                    st.error(" 请输入自定义策略代码")
                    return
                
                from ..strategy.custom_strategy import create_strategy_from_code
                success, strategy_class, error = create_strategy_from_code(custom_code)
                if not success:
                    st.error(f" 策略代码错误: {error}")
                    return
            else:
                # 内置策略
                strategy_class = get_strategy_class(config["strategy_name"])
                if strategy_class is None:
                    st.error(f" 未知策略: {config['strategy_name']}")
                    return
            
            # 4. 执行回测
            status_text.text(" 正在执行回测...")
            progress_bar.progress(80)
            
            if self.engine is None:
                self.engine = BacktraderEngine(backtest_config)
            
            result = self.engine.run_backtest(
                data=df,
                strategy_class=strategy_class,
                strategy_params=config["strategy_params"],
                symbol=config["symbol"],
                interval=config["interval"],
                start_time=str(config["start_date"]),
                end_time=str(config["end_date"])
            )
            
            # 5. 保存结果
            progress_bar.progress(100)
            status_text.text(" 回测完成！")
            
            st.session_state.backtest_result = {
                "result": result,
                "data": df,
                "config": config
            }
            
            st.rerun()
            
        except Exception as e:
            error_msg = self.exception_handler.format_user_friendly_error({
                "error_type": "STRATEGY_ERROR",
                "error_message": str(e)
            })
            st.error(f"回测失败: {error_msg}")
            logger.exception("回测执行失败")
            
            # 检查是否是数据量不足的错误
            if "数据量不足" in str(e) or "数据量不足" in error_msg:
                st.warning("提示: 请扩大时间范围或选择更短的时间周期以获取更多数据")
    
    def _render_backtest_results(self, backtest_data: Dict[str, Any], config: Dict[str, Any]):
        """渲染回测结果"""
        result = backtest_data["result"]
        df = backtest_data["data"]
        
        if not result.get("success"):
            st.error(f" 回测失败: {result.get('error', '未知错误')}")
            return
        
        performance = result.get("performance", {})
        
        # 绩效指标卡片
        st.subheader(" 绩效指标")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_return = performance.get('total_return', 0) or 0
            st.metric(
                "总收益率",
                f"{total_return:.2%}"
            )
        with col2:
            annual_return = performance.get('annual_return', 0) or 0
            st.metric(
                "年化收益率",
                f"{annual_return:.2%}"
            )
        with col3:
            max_drawdown = performance.get('max_drawdown', 0) or 0
            st.metric(
                "最大回撤",
                f"{max_drawdown:.2%}"
            )
        with col4:
            sharpe_ratio = performance.get('sharpe_ratio', 0) or 0
            st.metric(
                "夏普比率",
                f"{sharpe_ratio:.2f}"
            )
        
        col5, col6, col7, col8 = st.columns(4)
        with col5:
            st.metric("总交易次数", performance.get("total_trades", 0))
        with col6:
            win_rate = performance.get('win_rate', 0) or 0
            st.metric("胜率", f"{win_rate:.2%}")
        with col7:
            profit_loss_ratio = performance.get('profit_loss_ratio', 0) or 0
            st.metric("盈亏比", f"{profit_loss_ratio:.2f}")
        with col8:
            st.metric("最终资金", f"{result.get('final_value', 0):.2f} USDT")
        
        # 手续费信息
        col9 = st.columns(1)[0]
        with col9:
            total_commission = result.get('total_commission', 0)
            st.metric("总手续费", f"{total_commission:.2f} USDT")
        
        st.markdown("---")
        
        # 资金曲线图
        st.subheader(" 策略资金曲线")
        self._render_equity_curve(backtest_data, config)
        
        st.markdown("---")
        
        # 图表展示（使用Plotly，包含买卖点标记）
        st.subheader(" 回测图表（含买卖点）")
        self._render_klines_chart(df, config, backtest_data)
        
        # 禁用Backtrader原生图表（macOS线程限制导致无法工作）
        # 所有功能已集成到Plotly图表中
        
        # 交易记录
        st.markdown("---")
        st.subheader(" 交易记录")
        trades = result.get("trades", [])
        if trades:
            trades_df = pd.DataFrame(trades)
            st.dataframe(trades_df, use_container_width=True)
        else:
            st.info("暂无交易记录")
    
    def _render_equity_curve(self, backtest_data: Dict[str, Any], config: Dict[str, Any]):
        """渲染资金曲线图"""
        result = backtest_data.get("result", {})
        equity_curve = result.get("equity_curve")
        
        # 处理equity_curve数据
        if equity_curve is None:
            st.info("资金曲线数据不可用")
            return
        
        # 如果equity_curve是字典列表，转换为DataFrame
        if isinstance(equity_curve, list):
            if not equity_curve:
                st.info("资金曲线数据不可用")
                return
            equity_df = pd.DataFrame(equity_curve)
            logger.info(f"资金曲线数据: {len(equity_df)} 条, 列: {equity_df.columns.tolist()}, 示例: {equity_df.head(1).to_dict('records') if not equity_df.empty else 'None'}")
            
            # 处理日期列
            if 'date' in equity_df.columns:
                equity_df['date'] = pd.to_datetime(equity_df['date'])
                equity_df = equity_df.set_index('date')
            elif 'equity' in equity_df.columns:
                # 如果没有date列，使用索引
                equity_df.index = pd.to_datetime(equity_df.index)
            else:
                # 如果既没有date也没有equity列，尝试使用第一列作为日期，第二列作为值
                if len(equity_df.columns) >= 2:
                    equity_df.columns = ['date', 'equity']
                    equity_df['date'] = pd.to_datetime(equity_df['date'])
                    equity_df = equity_df.set_index('date')
            
            equity_curve = equity_df
            if not equity_curve.empty:
                value_col = 'equity' if 'equity' in equity_curve.columns else equity_curve.columns[0]
                logger.info(f"转换后资金曲线: {len(equity_curve)} 条, 值范围: {equity_curve[value_col].min():.2f} - {equity_curve[value_col].max():.2f}")
        elif isinstance(equity_curve, pd.DataFrame):
            if equity_curve.empty:
                st.info("资金曲线数据不可用")
                return
            # 确保索引是日期类型
            if not isinstance(equity_curve.index, pd.DatetimeIndex):
                equity_curve.index = pd.to_datetime(equity_curve.index)
        
        # 创建Plotly图表
        fig = go.Figure()
        
        # 确定equity列名
        equity_col = 'equity' if 'equity' in equity_curve.columns else equity_curve.columns[0]
        
        # 计算资金变化范围，用于调整Y轴显示
        equity_values = equity_curve[equity_col]
        min_value = equity_values.min()
        max_value = equity_values.max()
        value_range = max_value - min_value
        initial_value = result.get('initial_value', equity_values.iloc[0] if len(equity_values) > 0 else 0)
        
        # 如果变化范围很小，调整Y轴范围使其更明显
        if value_range > 0 and value_range < initial_value * 0.1:
            # 变化范围小于初始资金的10%，扩大显示范围
            y_min = min_value - value_range * 0.5
            y_max = max_value + value_range * 0.5
        else:
            y_min = None
            y_max = None
        
        # 添加资金曲线 - 使用最平滑的连续曲线
        # 对所有数据进行插值处理，确保曲线最平滑连续
        try:
            # 创建时间序列
            equity_series = pd.Series(equity_values.values, index=equity_curve.index)
            
            # 对所有数据都进行插值，增加数据点密度以获得最平滑的曲线
            if len(equity_curve.index) > 1:
                # 计算目标数据点数量（至少500个点，或原始数据的5倍，取较大值）
                target_points = max(500, len(equity_curve.index) * 5)
                
                # 创建更密集的时间索引
                new_index = pd.date_range(
                    start=equity_curve.index[0],
                    end=equity_curve.index[-1],
                    periods=target_points
                )
                
                # 使用三次样条插值（最平滑的方法）
                try:
                    # 尝试使用三次样条插值
                    equity_series_dense = equity_series.reindex(new_index).interpolate(method='cubic')
                except:
                    # 如果三次样条失败（可能数据点太少），使用二次插值
                    try:
                        equity_series_dense = equity_series.reindex(new_index).interpolate(method='quadratic')
                    except:
                        # 如果二次插值也失败，使用线性插值
                        equity_series_dense = equity_series.reindex(new_index).interpolate(method='linear')
                
                # 计算插值后的收益率
                customdata_smooth = [(v - initial_value) / initial_value for v in equity_series_dense.values]
                
                fig.add_trace(go.Scatter(
                    x=equity_series_dense.index,
                    y=equity_series_dense.values,
                    mode='lines',
                    name='资金曲线',
                    line=dict(
                        color='#1f77b4',
                        width=2.5,
                        shape='spline'  # 使用样条曲线，最平滑
                    ),
                    fill='tozeroy',
                    fillcolor='rgba(31, 119, 180, 0.1)',
                    hovertemplate='日期: %{x|%Y-%m-%d %H:%M}<br>资金: %{y:,.2f} USDT<br>收益率: %{customdata:.2%}<extra></extra>',
                    customdata=customdata_smooth
                ))
            else:
                # 数据点太少，直接绘制
                fig.add_trace(go.Scatter(
                    x=equity_curve.index,
                    y=equity_values,
                    mode='lines',
                    name='资金曲线',
                    line=dict(
                        color='#1f77b4',
                        width=2.5,
                        shape='spline'  # 使用样条曲线
                    ),
                    fill='tozeroy',
                    fillcolor='rgba(31, 119, 180, 0.1)',
                    hovertemplate='日期: %{x|%Y-%m-%d %H:%M}<br>资金: %{y:,.2f} USDT<br>收益率: %{customdata:.2%}<extra></extra>',
                    customdata=[(v - initial_value) / initial_value for v in equity_values]
                ))
        except Exception as e:
            logger.warning(f"平滑曲线处理失败，使用原始数据: {e}")
            # 如果插值失败，使用原始数据但启用样条曲线
            fig.add_trace(go.Scatter(
                x=equity_curve.index,
                y=equity_values,
                mode='lines',
                name='资金曲线',
                line=dict(
                    color='#1f77b4',
                    width=2.5,
                    shape='spline'  # 使用样条曲线
                ),
                fill='tozeroy',
                fillcolor='rgba(31, 119, 180, 0.1)',
                hovertemplate='日期: %{x|%Y-%m-%d %H:%M}<br>资金: %{y:,.2f} USDT<br>收益率: %{customdata:.2%}<extra></extra>',
                customdata=[(v - initial_value) / initial_value for v in equity_values]
            ))
        
        # 添加初始资金线（参考线）
        if initial_value > 0:
            fig.add_hline(
                y=initial_value,
                line_dash="dash",
                line_color="gray",
                line_width=1.5,
                annotation_text=f"初始资金: {initial_value:,.2f} USDT",
                annotation_position="bottom right",
                annotation_font_size=10
            )
        
        # 添加最终资金标注
        if len(equity_values) > 0:
            final_value = equity_values.iloc[-1]
            final_date = equity_curve.index[-1]
            fig.add_annotation(
                x=final_date,
                y=final_value,
                text=f"最终: {final_value:,.2f} USDT",
                showarrow=True,
                arrowhead=2,
                arrowcolor="#ff7f0e",
                bgcolor="rgba(255, 255, 255, 0.8)",
                bordercolor="#ff7f0e",
                borderwidth=1,
                font=dict(size=10, color="#ff7f0e")
            )
        
        # 更新布局
        fig.update_layout(
            title=dict(
                text=f"{config.get('symbol', '')} 策略资金曲线",
                font=dict(size=16, color='#2c3e50')
            ),
            xaxis_title="日期",
            yaxis_title="资金 (USDT)",
            hovermode='x unified',
            height=500,
            showlegend=True,
            template='plotly_white',
            yaxis=dict(
                range=[y_min, y_max] if y_min is not None and y_max is not None else None,
                tickformat=',.0f',  # 格式化Y轴数字
                showgrid=True,
                gridcolor='rgba(128, 128, 128, 0.2)'
            ),
            xaxis=dict(
                showgrid=True,
                gridcolor='rgba(128, 128, 128, 0.2)'
            ),
            margin=dict(l=60, r=60, t=60, b=60)
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def _render_klines_chart(self, df: pd.DataFrame, config: Dict[str, Any], backtest_data: Optional[Dict[str, Any]] = None):
        """渲染K线图表，包含买卖点标记"""
        fig = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.1,
            row_heights=[0.7, 0.3],
            subplot_titles=(f"{config['symbol']} K线图（含买卖点）", "成交量")
        )
        
        # K线图
        fig.add_trace(
            go.Candlestick(
                x=df.index,
                open=df['open'],
                high=df['high'],
                low=df['low'],
                close=df['close'],
                name="K线"
            ),
            row=1, col=1
        )
        
        # 提取并绘制指标线
        if backtest_data:
            result = backtest_data.get("result", {})
            indicator_data = result.get("indicator_data", {})
            strategy_name = config.get("strategy_name", "")
            
            # 根据策略类型绘制相应的指标
            if indicator_data:
                # 准备指标数据
                indicator_dates = []
                fast_values = []
                slow_values = []
                rsi_values = []
                macd_values = []
                signal_values = []
                boll_top_values = []
                boll_mid_values = []
                boll_bot_values = []
                
                for date_str, indicators in indicator_data.items():
                    try:
                        # 转换日期
                        if isinstance(date_str, str):
                            pd_date = pd.to_datetime(date_str)
                        elif hasattr(date_str, 'date'):
                            pd_date = pd.to_datetime(date_str)
                        else:
                            pd_date = pd.to_datetime(str(date_str))
                        
                        # 确保日期在数据范围内
                        if pd_date.date() >= df.index[0].date() and pd_date.date() <= df.index[-1].date():
                            indicator_dates.append(pd_date)
                            
                            # MA/EMA指标
                            if 'fast_ma' in indicators:
                                fast_values.append(indicators['fast_ma'])
                            elif 'fast_ema' in indicators:
                                fast_values.append(indicators['fast_ema'])
                            else:
                                fast_values.append(None)
                            
                            if 'slow_ma' in indicators:
                                slow_values.append(indicators['slow_ma'])
                            elif 'slow_ema' in indicators:
                                slow_values.append(indicators['slow_ema'])
                            else:
                                slow_values.append(None)
                            
                            # RSI指标
                            if 'rsi' in indicators:
                                rsi_values.append(indicators['rsi'])
                            else:
                                rsi_values.append(None)
                            
                            # MACD指标
                            if 'macd' in indicators:
                                macd_values.append(indicators['macd'])
                            else:
                                macd_values.append(None)
                            
                            if 'signal' in indicators:
                                signal_values.append(indicators['signal'])
                            else:
                                signal_values.append(None)
                            
                            # BOLL指标
                            if 'boll_top' in indicators:
                                boll_top_values.append(indicators['boll_top'])
                            else:
                                boll_top_values.append(None)
                            
                            if 'boll_mid' in indicators:
                                boll_mid_values.append(indicators['boll_mid'])
                            else:
                                boll_mid_values.append(None)
                            
                            if 'boll_bot' in indicators:
                                boll_bot_values.append(indicators['boll_bot'])
                            else:
                                boll_bot_values.append(None)
                    except Exception as e:
                        logger.debug(f"处理指标数据时出错: {e}")
                        continue
                
                # 绘制MA/EMA指标
                if fast_values and any(v is not None for v in fast_values):
                    fig.add_trace(
                        go.Scatter(
                            x=indicator_dates,
                            y=fast_values,
                            mode='lines',
                            name='快线' if 'MA' in strategy_name or 'EMA' in strategy_name else '快EMA',
                            line=dict(color='orange', width=1.5, dash='dash'),
                            hovertemplate='快线: %{y:.2f}<extra></extra>'
                        ),
                        row=1, col=1
                    )
                
                if slow_values and any(v is not None for v in slow_values):
                    fig.add_trace(
                        go.Scatter(
                            x=indicator_dates,
                            y=slow_values,
                            mode='lines',
                            name='慢线' if 'MA' in strategy_name or 'EMA' in strategy_name else '慢EMA',
                            line=dict(color='blue', width=1.5, dash='dash'),
                            hovertemplate='慢线: %{y:.2f}<extra></extra>'
                        ),
                        row=1, col=1
                    )
                
                # 绘制BOLL指标
                if boll_top_values and any(v is not None for v in boll_top_values):
                    fig.add_trace(
                        go.Scatter(
                            x=indicator_dates,
                            y=boll_top_values,
                            mode='lines',
                            name='BOLL上轨',
                            line=dict(color='purple', width=1),
                            hovertemplate='BOLL上轨: %{y:.2f}<extra></extra>'
                        ),
                        row=1, col=1
                    )
                    
                    fig.add_trace(
                        go.Scatter(
                            x=indicator_dates,
                            y=boll_mid_values,
                            mode='lines',
                            name='BOLL中轨',
                            line=dict(color='purple', width=1, dash='dot'),
                            hovertemplate='BOLL中轨: %{y:.2f}<extra></extra>'
                        ),
                        row=1, col=1
                    )
                    
                    fig.add_trace(
                        go.Scatter(
                            x=indicator_dates,
                            y=boll_bot_values,
                            mode='lines',
                            name='BOLL下轨',
                            line=dict(color='purple', width=1),
                            hovertemplate='BOLL下轨: %{y:.2f}<extra></extra>'
                        ),
                        row=1, col=1
                    )
        
        # 提取买卖信号并标注
        if backtest_data:
            trades = backtest_data.get("result", {}).get("trades", [])
            logger.info(f" 从backtest_data提取到 {len(trades)} 个交易记录")
            # 支持两种类型：buy_signal/sell_signal 和 buy/sell
            buy_signals = [t for t in trades if t.get("type") in ["buy_signal", "buy"]]
            sell_signals = [t for t in trades if t.get("type") in ["sell_signal", "sell"]]
            logger.info(f" 买入信号: {len(buy_signals)}, 卖出信号: {len(sell_signals)}")
            if buy_signals:
                logger.debug(f"第一个买入信号示例: {buy_signals[0]}")
            if sell_signals:
                logger.debug(f"第一个卖出信号示例: {sell_signals[0]}")
            
            # 添加买入点（绿色向上箭头）
            if buy_signals:
                buy_dates = []
                buy_prices = []
                for signal in buy_signals:
                    signal_date = signal.get("date")
                    price = signal.get("price")
                    if signal_date and price:
                        try:
                            # 转换为pandas时间戳
                            if isinstance(signal_date, str):
                                pd_date = pd.to_datetime(signal_date)
                            elif hasattr(signal_date, 'isoformat'):
                                pd_date = pd.to_datetime(signal_date.isoformat())
                            elif hasattr(signal_date, 'date'):
                                pd_date = pd.to_datetime(signal_date)
                            else:
                                pd_date = pd.to_datetime(str(signal_date))
                            
                            # 确保日期在数据范围内（使用日期部分比较）
                            if pd_date.date() >= df.index[0].date() and pd_date.date() <= df.index[-1].date():
                                buy_dates.append(pd_date)
                                buy_prices.append(float(price))
                            else:
                                logger.debug(f"买入信号日期超出范围: {pd_date.date()}, 数据范围: {df.index[0].date()} 到 {df.index[-1].date()}")
                        except Exception as e:
                            logger.warning(f"处理买入信号日期失败: {signal_date}, 类型: {type(signal_date)}, 错误: {e}")
                            continue
                
                if buy_dates:
                    fig.add_trace(
                        go.Scatter(
                            x=buy_dates,
                            y=buy_prices,
                            mode='markers',
                            name='买入点',
                            marker=dict(
                                symbol='triangle-up',
                                size=15,
                                color='green',
                                line=dict(width=2, color='darkgreen')
                            ),
                            hovertemplate='买入<br>日期: %{x}<br>价格: %{y:.4f}<extra></extra>'
                        ),
                        row=1, col=1
                    )
                    logger.info(f" 在图表上标注了 {len(buy_dates)} 个买入点")
                else:
                    logger.warning(f" 有 {len(buy_signals)} 个买入信号，但无法在图表上标注")
            
            # 添加卖出点（红色向下箭头）
            if sell_signals:
                sell_dates = []
                sell_prices = []
                for signal in sell_signals:
                    signal_date = signal.get("date")
                    price = signal.get("price")
                    if signal_date and price:
                        try:
                            # 转换为pandas时间戳
                            if isinstance(signal_date, str):
                                pd_date = pd.to_datetime(signal_date)
                            elif hasattr(signal_date, 'isoformat'):
                                pd_date = pd.to_datetime(signal_date.isoformat())
                            elif hasattr(signal_date, 'date'):
                                pd_date = pd.to_datetime(signal_date)
                            else:
                                pd_date = pd.to_datetime(str(signal_date))
                            
                            # 确保日期在数据范围内（使用日期部分比较）
                            if pd_date.date() >= df.index[0].date() and pd_date.date() <= df.index[-1].date():
                                sell_dates.append(pd_date)
                                sell_prices.append(float(price))
                            else:
                                logger.debug(f"卖出信号日期超出范围: {pd_date.date()}, 数据范围: {df.index[0].date()} 到 {df.index[-1].date()}")
                        except Exception as e:
                            logger.warning(f"处理卖出信号日期失败: {signal_date}, 类型: {type(signal_date)}, 错误: {e}")
                            continue
                
                if sell_dates:
                    fig.add_trace(
                        go.Scatter(
                            x=sell_dates,
                            y=sell_prices,
                            mode='markers',
                            name='卖出点',
                            marker=dict(
                                symbol='triangle-down',
                                size=15,
                                color='red',
                                line=dict(width=2, color='darkred')
                            ),
                            hovertemplate='卖出<br>日期: %{x}<br>价格: %{y:.4f}<extra></extra>'
                        ),
                        row=1, col=1
                    )
                    logger.info(f" 在图表上标注了 {len(sell_dates)} 个卖出点")
                else:
                    logger.warning(f" 有 {len(sell_signals)} 个卖出信号，但无法在图表上标注")
        
        # 成交量
        colors = ['red' if df['close'].iloc[i] >= df['open'].iloc[i] else 'green'
                 for i in range(len(df))]
        fig.add_trace(
            go.Bar(
                x=df.index,
                y=df['volume'],
                name="成交量",
                marker_color=colors
            ),
            row=2, col=1
        )
        
        fig.update_layout(
            height=800,
            xaxis_rangeslider_visible=False,
            showlegend=True
        )
        
        fig.update_xaxes(title_text="时间", row=2, col=1)
        fig.update_yaxes(title_text="价格", row=1, col=1)
        fig.update_yaxes(title_text="成交量", row=2, col=1)
        
        st.plotly_chart(fig, use_container_width=True)
    
    def _render_backtrader_chart(self, backtest_data: Dict[str, Any]):
        """渲染Backtrader原生图表（转换为图片在网页显示）"""
        try:
            cerebro = backtest_data.get("result", {}).get("cerebro")
            if cerebro is None:
                st.warning(" 无法显示Backtrader图表：cerebro对象不可用（可能使用了缓存）")
                st.info(" 提示：请清除缓存后重新运行回测以查看Backtrader原生图表")
                return
            
            # 使用Backtrader的plot方法生成图表
            st.info(" 正在生成Backtrader图表...")
            
            try:
                # 强制设置matplotlib后端（必须在Backtrader内部导入之前）
                import os
                os.environ['MPLBACKEND'] = 'Agg'
                
                # 重新设置后端（防止Backtrader内部重新导入）
                # 注意：不能在这里重新import matplotlib，因为会导致UnboundLocalError
                # 使用模块级别的matplotlib
                matplotlib.use('Agg', force=True)
                
                # 如果Backtrader已经导入了matplotlib，尝试monkey patch
                try:
                    import backtrader.plot.plot as btplot
                    if hasattr(btplot, 'mpyplot'):
                        # 重新导入pyplot以确保使用Agg后端
                        import matplotlib.pyplot as mpyplot
                        btplot.mpyplot = mpyplot
                except Exception as patch_error:
                    logger.debug(f"无法patch Backtrader plot模块: {patch_error}")
                
                # 使用candle样式，绿色上涨，红色下跌，显示成交量
                # 根据Backtrader文档，plot()返回figure列表
                figs = cerebro.plot(
                    style='candle',
                    barup='green',
                    bardown='red',
                    volume=True,
                    iplot=False,  # 不在Jupyter中自动显示
                    show=False    # 不自动显示（重要：避免阻塞）
                )
                
                if figs and len(figs) > 0:
                    # 将matplotlib图表转换为图片并在Streamlit中显示
                    from io import BytesIO
                    
                    for i, fig in enumerate(figs):
                        if fig:
                            # 将图表保存到内存中的BytesIO对象
                            buf = BytesIO()
                            fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
                            buf.seek(0)
                            
                            # 在Streamlit中显示图片
                            st.image(buf, use_container_width=True, caption=f"Backtrader回测图表 {i+1}")
                            
                            # 关闭图表释放内存
                            plt.close(fig)
                            buf.close()
                else:
                    # 如果没有返回figures，尝试获取当前figure
                    fig = plt.gcf()
                    if fig and len(fig.axes) > 0:
                        from io import BytesIO
                        buf = BytesIO()
                        fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
                        buf.seek(0)
                        st.image(buf, use_container_width=True, caption="Backtrader回测图表")
                        plt.close(fig)
                        buf.close()
                    else:
                        st.warning(" 未能生成图表，请检查数据")
                        
            except Exception as e:
                logger.error(f"Backtrader绘图错误: {e}")
                import traceback
                logger.error(traceback.format_exc())
                st.error(f" 生成Backtrader图表失败: {str(e)}")
                st.warning(" 由于macOS线程限制，Backtrader原生图表可能无法显示。")
                st.info(" 建议使用上方的Plotly交互图表，功能更强大且兼容性更好。")
                st.info(" Plotly图表支持缩放、悬停查看数据、下载图片等功能。")
                
        except Exception as e:
            logger.exception("渲染Backtrader图表失败")
            st.error(f" 显示Backtrader图表时出错: {e}")
            st.info(" 建议使用Plotly交互图表，功能更强大且兼容性更好")
    
    def _render_grid_search(self, config: Dict[str, Any]):
        """渲染参数网格搜索界面"""
        st.subheader(" 参数网格搜索优化")
        st.info("通过网格搜索找到最优策略参数组合")
        
        # 参数网格配置
        st.markdown("### 参数范围配置")
        
        strategy_name = config.get("strategy_name")
        if not strategy_name:
            st.warning("请先选择内置策略")
            return
        
        param_grid = {}
        
        if strategy_name in ["MA", "EMA"]:
            col1, col2 = st.columns(2)
            with col1:
                fast_periods = st.text_input(
                    "短期周期范围",
                    value="10,20,30",
                    help="用逗号分隔，如：10,20,30"
                )
                if fast_periods:
                    param_grid["fast_period"] = [int(x.strip()) for x in fast_periods.split(",")]
            with col2:
                slow_periods = st.text_input(
                    "长期周期范围",
                    value="30,40,50",
                    help="用逗号分隔，如：30,40,50"
                )
                if slow_periods:
                    param_grid["slow_period"] = [int(x.strip()) for x in slow_periods.split(",")]
        
        # 优化指标选择
        metric = st.selectbox(
            "优化指标",
            ["total_return", "sharpe_ratio", "win_rate", "profit_loss_ratio"],
            index=0
        )
        
        if st.button(" 开始优化", type="primary", key="grid_search"):
            self._run_grid_search(config, param_grid, metric)
    
    def _run_grid_search(self, config: Dict[str, Any], param_grid: Dict[str, List[Any]], metric: str):
        """执行网格搜索优化"""
        try:
            if not param_grid:
                st.warning("请配置参数范围")
                return
            
            # 初始化组件
            if self.data_fetcher is None:
                self.data_fetcher = BinanceDataFetcher(self.proxy_manager)
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # 获取数据
            status_text.text("📥 正在获取K线数据...")
            import asyncio
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            
            df = loop.run_until_complete(
                self.data_fetcher.get_klines_batch(
                    symbol=config["symbol"],
                    interval=config["interval"],
                    start_time=config["start_date"],
                    end_time=config["end_date"],
                    market=config["market"]
                )
            )
            
            try:
                loop.run_until_complete(self.data_fetcher.close())
            except:
                pass
            
            if df.empty:
                st.error(" 未能获取到K线数据")
                return
            
            # 创建回测配置
            backtest_config = BacktraderConfig(
                initial_cash=config["initial_cash"],
                commission=config["commission"],
                slippage=config["slippage"],
                leverage=config.get("leverage")
            )
            
            # 获取策略类
            strategy_class = get_strategy_class(config["strategy_name"])
            if strategy_class is None:
                st.error(f" 未知策略: {config['strategy_name']}")
                return
            
            # 执行网格搜索
            status_text.text(" 正在执行网格搜索...")
            from ..backtest.grid_search import GridSearchOptimizer
            
            if self.engine is None:
                self.engine = BacktraderEngine(backtest_config)
            
            optimizer = GridSearchOptimizer(self.engine)
            results = optimizer.optimize(df, strategy_class, param_grid, metric)
            
            # 显示结果
            status_text.text(" 优化完成！")
            progress_bar.progress(100)
            
            if results:
                st.success(f"找到 {len(results)} 组有效参数组合")
                
                # 显示最优结果
                best = results[0]
                st.markdown("### 🏆 最优参数组合")
                st.json(best["params"])
                st.metric("最优指标值", f"{best['metric_value']:.4f}")
                
                # 显示前10名
                st.markdown("###  Top 10 参数组合")
                top_results = results[:10]
                comparison_data = []
                for i, r in enumerate(top_results, 1):
                    comparison_data.append({
                        "排名": i,
                        "参数": str(r["params"]),
                        metric: f"{r['metric_value']:.4f}",
                        "总收益率": f"{r['performance'].get('total_return', 0):.2%}",
                        "夏普比率": f"{r['performance'].get('sharpe_ratio', 0):.2f}",
                    })
                
                st.dataframe(pd.DataFrame(comparison_data), use_container_width=True)
            else:
                st.warning("未找到有效结果，请调整参数范围")
                
        except Exception as e:
            st.error(f" 网格搜索失败: {e}")
            logger.exception("网格搜索执行失败")
    
    def _render_strategy_comparison(self, config: Dict[str, Any]):
        """渲染策略对比界面"""
        st.subheader(" 多策略对比")
        st.info("可以运行多个策略并对比其性能")
        
        if "comparison_results" not in st.session_state:
            st.session_state.comparison_results = []
        
        # 策略列表
        if st.session_state.comparison_results:
            from ..backtest.strategy_comparison import StrategyComparator
            comparator = StrategyComparator()
            for result in st.session_state.comparison_results:
                comparator.add_result(
                    result["strategy_name"],
                    result["result"],
                    result.get("config")
                )
            
            # 显示对比表格
            comparison_df = comparator.get_comparison_table()
            st.dataframe(comparison_df, use_container_width=True)
            
            # 对比图表
            metrics = comparator.get_metrics_comparison()
            if metrics:
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=metrics["strategy_names"],
                    y=metrics["total_return"],
                    name="总收益率"
                ))
                fig.update_layout(
                    title="策略对比 - 总收益率",
                    xaxis_title="策略",
                    yaxis_title="收益率",
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)
            
            if st.button(" 清空对比结果"):
                st.session_state.comparison_results = []
                st.rerun()
        else:
            st.info("暂无对比结果，请先运行多个策略回测")
    
    def _render_advanced_features(self, config: Dict[str, Any]):
        """渲染高级功能界面"""
        st.subheader(" 高级功能")
        
        # 缓存管理
        st.markdown("### 缓存管理")
        col1, col2 = st.columns(2)
        with col1:
            if st.button(" 清空缓存", use_container_width=True):
                from ..backtest.cache_manager import CacheManager
                cache_manager = CacheManager()
                cache_manager.clear()
                st.success("缓存已清空")
        with col2:
            if st.button(" 清理过期缓存", use_container_width=True):
                from ..backtest.cache_manager import CacheManager
                cache_manager = CacheManager()
                cache_manager.clear_expired()
                st.success("过期缓存已清理")
        
        # 多时间框架分析
        st.markdown("### 多时间框架分析")
        st.info("多时间框架分析功能开发中...")
    
    def run(self):
        """运行Streamlit应用"""
        self.setup_page()
        self.render_header()
        
        # 侧边栏配置
        config = self.render_sidebar()
        
        # 主内容区
        self.render_main_content(config)
        
        # 清理资源
        if self.data_fetcher:
            import asyncio
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # 如果事件循环正在运行，使用回调清理
                    loop.create_task(self.data_fetcher.close())
                else:
                    loop.run_until_complete(self.data_fetcher.close())
            except:
                pass
