#!/usr/bin/env python3
# 股票项目分析脚本

import os
import pandas as pd
import numpy as np
from datetime import datetime

def analyze_stock_project():
    """分析股票项目文件夹"""
    print("=" * 60)
    print("股票分析项目状态检查")
    print("=" * 60)
    
    # 1. 检查文件
    print("\n1. 文件检查:")
    files = os.listdir('.')
    
    py_files = [f for f in files if f.endswith('.py')]
    csv_files = [f for f in files if f.endswith('.csv')]
    excel_files = [f for f in files if f.endswith('.xlsx')]
    
    print(f"  Python脚本: {len(py_files)} 个")
    for f in sorted(py_files)[:10]:
        print(f"    - {f}")
    
    print(f"\n  CSV数据文件: {len(csv_files)} 个")
    for f in sorted(csv_files)[:5]:
        size = os.path.getsize(f) / 1024
        print(f"    - {f} ({size:.1f} KB)")
    
    print(f"\n  Excel文件: {len(excel_files)} 个")
    for f in excel_files:
        size = os.path.getsize(f) / 1024
        print(f"    - {f} ({size:.1f} KB)")
    
    # 2. 检查历史数据文件夹
    print("\n2. 历史数据文件夹检查:")
    history_dir = './history_data'
    if os.path.exists(history_dir):
        history_files = os.listdir(history_dir)
        print(f"  history_data 中有 {len(history_files)} 个文件")
        if history_files:
            # 统计文件类型
            csv_count = sum(1 for f in history_files if f.endswith('.csv'))
            print(f"    其中CSV文件: {csv_count} 个")
            print(f"    示例文件: {history_files[0]}")
    else:
        print("  history_data 文件夹不存在")
    
    # 3. 分析回测结果
    print("\n3. 回测结果分析:")
    if csv_files:
        try:
            # 尝试读取网格搜索结果
            grid_files = [f for f in csv_files if 'grid' in f.lower()]
            if grid_files:
                df = pd.read_csv(grid_files[0])
                print(f"  文件: {grid_files[0]}")
                print(f"  数据行数: {len(df)}")
                print(f"  参数组合数: {len(df)}")
                
                if '总收益率(%)' in df.columns:
                    best_idx = df['总收益率(%)'].idxmax()
                    worst_idx = df['总收益率(%)'].idxmin()
                    
                    print(f"  最佳收益率: {df.loc[best_idx, '总收益率(%)']:.2f}%")
                    print(f"  最差收益率: {df.loc[worst_idx, '总收益率(%)']:.2f}%")
                    print(f"  平均收益率: {df['总收益率(%)'].mean():.2f}%")
                    
                    # 显示最佳参数
                    print(f"  最佳参数组合:")
                    best_row = df.loc[best_idx]
                    if '止盈(%)' in df.columns and '止损(%)' in df.columns:
                        print(f"    止盈: {best_row['止盈(%)']}%, 止损: {best_row['止损(%)']}%")
                        print(f"    胜率: {best_row.get('胜率(%)', 'N/A')}%")
                        print(f"    交易笔数: {best_row.get('总交易笔数', 'N/A')}")
        except Exception as e:
            print(f"  分析错误: {e}")
    
    # 4. 检查依赖
    print("\n4. Python包依赖检查:")
    try:
        import pandas as pd
        print(f"  pandas: {pd.__version__} ✓")
    except:
        print("  pandas: 未安装 ✗")
    
    try:
        import numpy as np
        print(f"  numpy: {np.__version__} ✓")
    except:
        print("  numpy: 未安装 ✗")
    
    try:
        import akshare as ak
        print(f"  akshare: 已安装 ✓")
    except:
        print("  akshare: 未安装 ✗")
    
    try:
        import baostock as bs
        print(f"  baostock: 已安装 ✓")
    except:
        print("  baostock: 未安装 ✗")
    
    # 5. 建议
    print("\n5. 建议:")
    print("  a) 如果要下载数据: python3 download_history.py")
    print("  b) 如果要运行分析: python3 stock2026.py")
    print("  c) 如果要运行回测: python3 stock_backtest_pro.py")
    print("  d) 查看HTML报告: 打开 index.html")
    
    print("\n" + "=" * 60)
    print("分析完成!")
    print("=" * 60)

if __name__ == "__main__":
    analyze_stock_project()