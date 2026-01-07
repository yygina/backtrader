"""
回测结果缓存管理器
"""

import hashlib
import json
import pickle
from typing import Optional, Dict, Any
from pathlib import Path
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class CacheManager:
    """
    缓存管理器
    缓存回测结果，避免重复计算
    """
    
    def __init__(self, cache_dir: str = ".cache", ttl_hours: int = 1):
        """
        初始化缓存管理器
        
        Args:
            cache_dir: 缓存目录
            ttl_hours: 缓存有效期（小时）
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.ttl = timedelta(hours=ttl_hours)
    
    def _generate_cache_key(
        self,
        symbol: str,
        interval: str,
        start_time: str,
        end_time: str,
        strategy_name: str,
        strategy_params: Dict[str, Any],
        backtest_params: Dict[str, Any]
    ) -> str:
        """
        生成缓存键
        
        Args:
            symbol: 交易对
            interval: 时间周期
            start_time: 起始时间
            end_time: 结束时间
            strategy_name: 策略名称
            strategy_params: 策略参数
            backtest_params: 回测参数
        
        Returns:
            str: 缓存键（MD5哈希）
        """
        key_data = {
            "symbol": symbol,
            "interval": interval,
            "start_time": start_time,
            "end_time": end_time,
            "strategy_name": strategy_name,
            "strategy_params": json.dumps(strategy_params, sort_keys=True),
            "backtest_params": json.dumps(backtest_params, sort_keys=True)
        }
        
        key_string = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def get(
        self,
        symbol: str,
        interval: str,
        start_time: str,
        end_time: str,
        strategy_name: str,
        strategy_params: Dict[str, Any],
        backtest_params: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        获取缓存结果
        
        Args:
            symbol: 交易对
            interval: 时间周期
            start_time: 起始时间
            end_time: 结束时间
            strategy_name: 策略名称
            strategy_params: 策略参数
            backtest_params: 回测参数
        
        Returns:
            dict: 缓存的结果，如果不存在或已过期返回None
        """
        cache_key = self._generate_cache_key(
            symbol, interval, start_time, end_time,
            strategy_name, strategy_params, backtest_params
        )
        
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        
        if not cache_file.exists():
            return None
        
        # 检查是否过期
        file_mtime = datetime.fromtimestamp(cache_file.stat().st_mtime)
        if datetime.now() - file_mtime > self.ttl:
            logger.info(f"缓存已过期: {cache_key}")
            cache_file.unlink()
            return None
        
        # 读取缓存
        try:
            with open(cache_file, 'rb') as f:
                result = pickle.load(f)
            logger.info(f"从缓存读取结果: {cache_key}")
            return result
        except Exception as e:
            logger.warning(f"读取缓存失败: {e}")
            return None
    
    def set(
        self,
        symbol: str,
        interval: str,
        start_time: str,
        end_time: str,
        strategy_name: str,
        strategy_params: Dict[str, Any],
        backtest_params: Dict[str, Any],
        result: Dict[str, Any]
    ):
        """
        保存缓存结果
        
        Args:
            symbol: 交易对
            interval: 时间周期
            start_time: 起始时间
            end_time: 结束时间
            strategy_name: 策略名称
            strategy_params: 策略参数
            backtest_params: 回测参数
            result: 回测结果
        """
        cache_key = self._generate_cache_key(
            symbol, interval, start_time, end_time,
            strategy_name, strategy_params, backtest_params
        )
        
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(result, f)
            logger.info(f"缓存结果已保存: {cache_key}")
        except Exception as e:
            logger.warning(f"保存缓存失败: {e}")
    
    def clear(self):
        """清空所有缓存"""
        for cache_file in self.cache_dir.glob("*.pkl"):
            cache_file.unlink()
        logger.info("缓存已清空")
    
    def clear_expired(self):
        """清除过期缓存"""
        now = datetime.now()
        cleared = 0
        
        for cache_file in self.cache_dir.glob("*.pkl"):
            file_mtime = datetime.fromtimestamp(cache_file.stat().st_mtime)
            if now - file_mtime > self.ttl:
                cache_file.unlink()
                cleared += 1
        
        if cleared > 0:
            logger.info(f"已清除 {cleared} 个过期缓存文件")

