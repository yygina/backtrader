"""
Streamlit应用入口
"""

# 必须在导入任何可能使用matplotlib的模块之前设置后端
import os
os.environ['MPLBACKEND'] = 'Agg'  # 通过环境变量设置，确保所有模块都使用Agg

import matplotlib
matplotlib.use('Agg', force=True)  # 强制使用Agg后端，避免macOS线程问题

import logging
import sys
from pathlib import Path

# 添加src目录到路径
sys.path.insert(0, str(Path(__file__).parent))

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Streamlit应用主函数
def main():
    """主函数"""
    from src.visualization.streamlit_ui import StreamlitUI
    ui = StreamlitUI()
    ui.run()

if __name__ == "__main__":
    main()

