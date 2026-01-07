"""
多时间框架分析支持
"""

import backtrader as bt
import pandas as pd
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class MultiTimeframeData:
    """
    多时间框架数据管理器
    """
    
    def __init__(self):
        """初始化"""
        self.datafeeds: Dict[str, bt.feeds.PandasData] = {}
    
    def add_datafeed(
        self,
        name: str,
        df: pd.DataFrame,
        timeframe: bt.TimeFrame = bt.TimeFrame.Days
    ) -> bt.feeds.PandasData:
        """
        添加数据源
        
        Args:
            name: 数据源名称
            df: K线数据DataFrame
            timeframe: 时间框架
        
        Returns:
            bt.feeds.PandasData: 数据源
        """
        # 确保列名符合Backtrader要求
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in df.columns for col in required_columns):
            raise ValueError(f"DataFrame必须包含列: {required_columns}")
        
        # 创建PandasData Feed
        datafeed = bt.feeds.PandasData(
            dataname=df,
            datetime=None,
            open=0,
            high=1,
            low=2,
            close=3,
            volume=4,
            openinterest=-1,
            timeframe=timeframe
        )
        
        self.datafeeds[name] = datafeed
        return datafeed
    
    def get_timeframe(self, interval: str) -> bt.TimeFrame:
        """
        将时间周期字符串转换为Backtrader TimeFrame
        
        Args:
            interval: 时间周期（如"1m", "1h", "1d"）
        
        Returns:
            bt.TimeFrame: Backtrader时间框架
        """
        timeframe_map = {
            "1m": bt.TimeFrame.Minutes,
            "5m": bt.TimeFrame.Minutes,
            "15m": bt.TimeFrame.Minutes,
            "30m": bt.TimeFrame.Minutes,
            "1h": bt.TimeFrame.Minutes,
            "2h": bt.TimeFrame.Minutes,
            "4h": bt.TimeFrame.Minutes,
            "1d": bt.TimeFrame.Days,
            "1w": bt.TimeFrame.Weeks,
            "1M": bt.TimeFrame.Months
        }
        
        return timeframe_map.get(interval, bt.TimeFrame.Days)
    
    def get_compression(self, interval: str) -> int:
        """
        获取压缩倍数
        
        Args:
            interval: 时间周期
        
        Returns:
            int: 压缩倍数
        """
        compression_map = {
            "1m": 1,
            "5m": 5,
            "15m": 15,
            "30m": 30,
            "1h": 60,
            "2h": 120,
            "4h": 240,
            "1d": 1,
            "1w": 1,
            "1M": 1
        }
        
        return compression_map.get(interval, 1)


def add_multiple_timeframes(
    cerebro: bt.Cerebro,
    data_dict: Dict[str, pd.DataFrame],
    main_interval: str = "1d"
):
    """
    向Cerebro添加多时间框架数据
    
    Args:
        cerebro: Cerebro实例
        data_dict: 数据字典，key为时间周期，value为DataFrame
        main_interval: 主时间周期
    """
    mtf = MultiTimeframeData()
    
    # 添加主数据
    if main_interval in data_dict:
        main_data = mtf.add_datafeed(
            "main",
            data_dict[main_interval],
            timeframe=mtf.get_timeframe(main_interval)
        )
        main_data.compression = mtf.get_compression(main_interval)
        cerebro.adddata(main_data)
    
    # 添加其他时间框架数据
    for interval, df in data_dict.items():
        if interval == main_interval:
            continue
        
        data = mtf.add_datafeed(
            interval,
            df,
            timeframe=mtf.get_timeframe(interval)
        )
        data.compression = mtf.get_compression(interval)
        cerebro.adddata(data, name=interval)

