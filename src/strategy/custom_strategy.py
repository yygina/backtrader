"""
自定义策略支持：代码编辑器、语法校验、策略执行
"""

import backtrader as bt
import ast
import sys
import io
from typing import Optional, Dict, Any, Tuple
import logging

logger = logging.getLogger(__name__)


class StrategyExecutor:
    """
    策略执行器
    支持动态加载和执行自定义策略代码
    """
    
    def __init__(self):
        """初始化"""
        self.strategy_code = ""
        self.strategy_class = None
        self.last_error = None
    
    def validate_code(self, code: str) -> Tuple[bool, Optional[str]]:
        """
        验证策略代码语法
        
        Args:
            code: 策略代码字符串
        
        Returns:
            Tuple[bool, Optional[str]]: (是否有效, 错误信息)
        """
        try:
            # 语法检查
            ast.parse(code)
            return True, None
        except SyntaxError as e:
            error_msg = f"语法错误 (行 {e.lineno}): {e.msg}"
            return False, error_msg
        except Exception as e:
            return False, str(e)
    
    def compile_strategy(
        self,
        code: str,
        strategy_name: str = "CustomStrategy"
    ) -> Tuple[bool, Optional[type], Optional[str]]:
        """
        编译策略代码
        
        Args:
            code: 策略代码字符串
            strategy_name: 策略类名称
        
        Returns:
            Tuple[bool, Optional[type], Optional[str]]: (是否成功, 策略类, 错误信息)
        """
        # 验证语法
        is_valid, error = self.validate_code(code)
        if not is_valid:
            return False, None, error
        
        # 创建命名空间
        namespace = {
            '__name__': '__main__',
            'backtrader': bt,
            'bt': bt,
        }
        
        # 执行代码
        try:
            exec(code, namespace)
            
            # 查找策略类
            strategy_class = None
            for name, obj in namespace.items():
                if (isinstance(obj, type) and
                    issubclass(obj, bt.Strategy) and
                    obj != bt.Strategy):
                    strategy_class = obj
                    break
            
            if strategy_class is None:
                # 尝试使用指定名称
                if strategy_name in namespace:
                    strategy_class = namespace[strategy_name]
                else:
                    return False, None, f"未找到策略类，请确保代码中定义了继承自bt.Strategy的类"
            
            self.strategy_class = strategy_class
            self.strategy_code = code
            return True, strategy_class, None
            
        except Exception as e:
            error_msg = f"编译失败: {str(e)}"
            logger.error(f"策略编译错误: {error_msg}")
            return False, None, error_msg
    
    def get_strategy_template(self) -> str:
        """
        获取策略模板代码
        
        Returns:
            str: 策略模板
        """
        return '''import backtrader as bt

class CustomStrategy(bt.Strategy):
    """
    自定义策略模板
    """
    
    params = (
        # 在这里定义策略参数
        ('param1', 10),
        ('param2', 20),
    )
    
    def __init__(self):
        """初始化指标"""
        # 在这里初始化技术指标
        # 例如：
        # self.sma = bt.indicators.SMA(self.data.close, period=self.params.param1)
        pass
    
    def next(self):
        """策略逻辑"""
        # 在这里编写交易逻辑
        # 例如：
        # if not self.position:
        #     if self.sma[0] > self.data.close[0]:
        #         self.buy()
        # else:
        #     if self.sma[0] < self.data.close[0]:
        #         self.sell()
        pass
    
    def log(self, txt, dt=None):
        """日志记录"""
        dt = dt or self.datas[0].datetime.date(0)
        print(f'{dt.isoformat()}: {txt}')
'''


def create_strategy_from_code(
    code: str,
    strategy_name: str = "CustomStrategy"
) -> Tuple[bool, Optional[type], Optional[str]]:
    """
    从代码创建策略类
    
    Args:
        code: 策略代码字符串
        strategy_name: 策略类名称
    
    Returns:
        Tuple[bool, Optional[type], Optional[str]]: (是否成功, 策略类, 错误信息)
    """
    executor = StrategyExecutor()
    return executor.compile_strategy(code, strategy_name)

