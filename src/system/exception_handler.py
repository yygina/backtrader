"""
异常处理器：统一异常捕获、友好化提示、异常信息脱敏
"""

import logging
import traceback
from typing import Optional, Dict, Any
from datetime import datetime

logger = logging.getLogger(__name__)


class ExceptionHandler:
    """异常处理器 - 统一处理异常，提供友好化提示"""
    
    @staticmethod
    def handle_api_error(
        symbol: str,
        error_type: str,
        error_detail: str,
        retry_count: int = 0
    ) -> Dict[str, Any]:
        """
        处理API请求错误
        
        Args:
            symbol: 交易对
            error_type: 错误类型
            error_detail: 错误详情
            retry_count: 重试次数
        
        Returns:
            dict: 错误信息字典
        """
        error_msg = f"交易对 {symbol} - {error_type}: {error_detail}"
        if retry_count > 0:
            error_msg += f" (已重试 {retry_count} 次)"
        
        logger.error(error_msg)
        
        return {
            "success": False,
            "error_type": error_type,
            "error_message": error_msg,
            "symbol": symbol,
            "timestamp": datetime.now().isoformat()
        }
    
    @staticmethod
    def handle_data_parse_error(
        symbol: str,
        error_detail: str
    ) -> Dict[str, Any]:
        """
        处理数据解析错误
        
        Args:
            symbol: 交易对
            error_detail: 错误详情
        
        Returns:
            dict: 错误信息字典
        """
        error_msg = f"交易对 {symbol} - 数据解析失败: {error_detail}"
        logger.error(error_msg)
        
        return {
            "success": False,
            "error_type": "DATA_PARSE_ERROR",
            "error_message": error_msg,
            "symbol": symbol,
            "timestamp": datetime.now().isoformat()
        }
    
    @staticmethod
    def handle_strategy_error(
        strategy_name: str,
        error: Exception,
        sanitize: bool = True
    ) -> Dict[str, Any]:
        """
        处理策略执行错误
        
        Args:
            strategy_name: 策略名称
            error: 异常对象
            sanitize: 是否脱敏（隐藏系统路径）
        
        Returns:
            dict: 错误信息字典
        """
        error_detail = str(error)
        trace = traceback.format_exc()
        
        # 脱敏处理：移除系统路径
        if sanitize:
            import os
            user_home = os.path.expanduser("~")
            trace = trace.replace(user_home, "~")
            # 移除其他系统路径
            trace = "\n".join([
                line for line in trace.split("\n")
                if not line.strip().startswith("File \"")
            ])
        
        error_msg = f"策略 {strategy_name} 执行失败: {error_detail}"
        logger.error(f"{error_msg}\n{trace}")
        
        return {
            "success": False,
            "error_type": "STRATEGY_ERROR",
            "error_message": error_msg,
            "error_detail": error_detail,
            "traceback": trace if not sanitize else None,
            "strategy_name": strategy_name,
            "timestamp": datetime.now().isoformat()
        }
    
    @staticmethod
    def handle_validation_error(
        param_name: str,
        param_value: Any,
        expected: str
    ) -> Dict[str, Any]:
        """
        处理参数校验错误
        
        Args:
            param_name: 参数名称
            param_value: 参数值
            expected: 期望值描述
        
        Returns:
            dict: 错误信息字典
        """
        error_msg = f"参数 {param_name} 无效: 当前值={param_value}, 期望: {expected}"
        logger.warning(error_msg)
        
        return {
            "success": False,
            "error_type": "VALIDATION_ERROR",
            "error_message": error_msg,
            "param_name": param_name,
            "param_value": param_value,
            "expected": expected,
            "timestamp": datetime.now().isoformat()
        }
    
    @staticmethod
    def format_user_friendly_error(error_dict: Dict[str, Any]) -> str:
        """
        将错误字典转换为用户友好的提示信息
        
        Args:
            error_dict: 错误信息字典
        
        Returns:
            str: 用户友好的错误提示
        """
        error_type = error_dict.get("error_type", "UNKNOWN_ERROR")
        error_message = error_dict.get("error_message", "未知错误")
        
        # 错误类型映射到友好提示
        friendly_messages = {
            "API_ERROR": "网络请求失败，请检查网络连接和代理设置",
            "PROXY_ERROR": "代理连接失败，请检查代理服务是否运行",
            "DATA_PARSE_ERROR": "数据解析失败，请稍后重试",
            "STRATEGY_ERROR": "策略执行出错，请检查策略代码",
            "VALIDATION_ERROR": "参数配置错误，请检查输入参数",
            "TIMEOUT_ERROR": "请求超时，请稍后重试",
        }
        
        base_message = friendly_messages.get(error_type, error_message)
        
        # 添加具体信息（如果有）
        if "symbol" in error_dict:
            base_message += f"\n交易对: {error_dict['symbol']}"
        
        return base_message

