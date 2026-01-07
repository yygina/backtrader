"""
代理管理器：强制使用代理，支持代理状态检测
"""

import aiohttp
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class ProxyManager:
    """代理管理器 - 强制所有请求走代理"""
    
    # 固定代理地址（根据需求文档）
    DEFAULT_PROXY = "http://127.0.0.1:7890"
    
    def __init__(self, proxy_url: Optional[str] = None):
        """
        初始化代理管理器
        
        Args:
            proxy_url: 代理地址，默认使用内置代理
        """
        self.proxy_url = proxy_url or self.DEFAULT_PROXY
        self._proxy_available = None
    
    @property
    def proxy(self) -> str:
        """获取代理地址"""
        return self.proxy_url
    
    async def check_proxy_available(self) -> bool:
        """
        检测代理是否可用
        
        Returns:
            bool: 代理是否可用
        """
        if self._proxy_available is not None:
            return self._proxy_available
        
        try:
            timeout = aiohttp.ClientTimeout(total=5, connect=3)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(
                    "https://www.google.com",
                    proxy=self.proxy_url
                ) as response:
                    self._proxy_available = response.status == 200
                    return self._proxy_available
        except Exception as e:
            logger.warning(f"代理检测失败: {e}")
            self._proxy_available = False
            return False
    
    def get_proxy_dict(self) -> dict:
        """
        获取代理字典（用于aiohttp）
        
        Returns:
            dict: 代理配置字典
        """
        return {"http": self.proxy_url, "https": self.proxy_url}
    
    def get_error_message(self) -> str:
        """
        获取代理错误提示信息（友好化）
        
        Returns:
            str: 错误提示信息
        """
        return f"代理连接失败，请检查 {self.proxy_url} 代理服务是否运行"

