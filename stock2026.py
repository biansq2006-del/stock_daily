#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import akshare as ak
import pandas as pd
import numpy as np
import datetime
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

# ==========================================
# 1. 核心算法函数定义 (保持不变)
# ==========================================

def sma(series, n, m):
    """
    策略1核心：通达信SMA递归算法
    公式: Y = (X*M + Y'*(N-M)) / N
    """
    sma_values = []
    # 为了加速，转换为numpy array处理
    series_array = series.values
    
    # 初始值处理
    # 模拟通达信：前N-1个无效，第N个为简单平均，之后递归
    # 但为了Python计算方便，通常从第一个有效值开始递归
    
    val = np.nan
    for i, x in enumerate(series_array):
        if i < n - 1:
            sma_values.append(np.nan)
        elif i == n - 1:
            # 初始值：取前n个的平均值作为启动值
            val = np.nanmean(series_array[:n])
            sma_values.append(val)
        else:
            if np.isnan(val):
                # 如果前值是NaN，尝试重新初始化
                val = np.nanmean(series_array[:i+1])
            else:
                # 递归公式
                val = (x * m + val * (n - m)) / n
            sma_values.append(val)
            
    return pd.Series(sma_values, index=series.index)

def calculate_xma(series, window):
    """策略2核心：EMA算法 (XMA)"""
    return series.ewm(span=window, adjust=False).mean()

# ==========================================
# 2. 单只股票处理引擎
# ==========================================

def process_stock(stock_info, start_date, end_date):
    """
    处理单只股票的所有指标和信号
    stock_info: 包含 {'code':..., 'name':..., 'industry':..., 'area':..., 'type':...}
    """
    symbol = stock_info['code']
    try:
        # 1. 获取数据
        # 策略1需要长期数据计算500日均线，往前推3年 (约1000天)
        fetch_start = (pd.to_datetime(start_date) - datetime.timedelta(days=1000)).strftime('%Y%m%d')
        fetch_end = pd.to_datetime(end_date).strftime('%Y%m%d')
        
        # 使用 akshare 获取后复权数据
        df = ak.stock_zh_a_hist(symbol=symbol, period="daily", start_date=fetch_start, end_date=fetch_end, adjust="qfq")
        
        if df.empty or len(df) < 500: # 数据不足500天无法计算核心策略
            return None
            
        df['日期'] = pd.to_datetime(df['日期'])
        df.set_index('日期', inplace=True)
        df.sort_index(inplace=True)
        
        # 转换数值类型，处理潜在的非数值字符
        for c in ['开盘', '收盘', '最高', '最低']:
            df[c] = pd.to_numeric(df[c], errors='coerce')

        # ==========================================
        # 计算通用指标 (BBI, MA60, 波动率)
        # ==========================================
        # MA60
        df['MA60'] = df['收盘'].rolling(60).mean()
        
        # BBI: (MA3 + MA6 + MA12 + MA24) / 4
        ma3 = df['收盘'].rolling(3).mean()
        ma6 = df['收盘'].rolling(6).mean()
        ma12 = df['收盘'].rolling(12).mean()
        ma24 = df['收盘'].rolling(24).mean()
        df['BBI'] = (ma3 + ma6 + ma12 + ma24) / 4
        
        # 波动率 (20日年化波动率)
        df['Log_Ret'] = np.log(df['收盘'] / df['收盘'].shift(1))
        df['波动率%'] = df['Log_Ret'].rolling(20).std() * np.sqrt(252) * 100

        # ==========================================
        # 计算策略 1: 历史大底 (Deep Bottom)
        # ==========================================
        # 这里的参数完全保留了您提供的第一段代码逻辑
        for p in [500, 250, 90]:
            df[f'HHV{p}'] = df['最高'].rolling(p).max()
            df[f'LLV{p}'] = df['最低'].rolling(p).min()
            df[f'R_HHV{p}'] = df[f'HHV{p}'].rolling(21).mean()
            df[f'R_LLV{p}'] = df[f'LLV{p}'].rolling(21).mean()
            
        df['R7'] = (df['R_LLV500']*0.96 + df['R_LLV250']*0.96 + df['R_LLV90']*0.96 + 
                    df['R_HHV500']*0.558 + df['R_HHV250']*0.558 + df['R_HHV90']*0.558) / 6
        df['R8'] = (df['R_LLV500']*1.25 + df['R_LLV250']*1.23 + df['R_LLV90']*1.2 + 
                    df['R_HHV500']*0.55 + df['R_HHV250']*0.55 + df['R_HHV90']*0.65) / 6
        df['R9'] = (df['R_LLV500']*1.3 + df['R_LLV250']*1.3 + df['R_LLV90']*1.3 + 
                    df['R_HHV500']*0.68 + df['R_HHV250']*0.68 + df['R_HHV90']*0.68) / 6
        
        # 核心基准线 RA
        df['RA'] = (df['R7']*3 + df['R8']*2 + df['R9']) / 6 * 1.738
        df['RA'] = df['RA'].rolling(21).mean()
        
        # 情绪指标 RC & RD
        df['RB'] = df['最低'].shift(1)
        df['ABS_LOW_RB'] = (df['最低'] - df['RB']).abs()
        df['MAX_LOW_RB'] = (df['最低'] - df['RB']).clip(lower=0)
        df['SMA_ABS'] = sma(df['ABS_LOW_RB'], 3, 1)
        df['SMA_MAX'] = sma(df['MAX_LOW_RB'], 3, 1)
        
        # 防止除零错误
        with np.errstate(divide='ignore', invalid='ignore'):
            df['RC'] = np.where(df['SMA_MAX'] != 0, (df['SMA_ABS'] / df['SMA_MAX']) * 100, 0)
        
        df['RD'] = np.where(df['收盘']*1.35 <= df['RA'], df['RC']*10, df['RC']/10)
        df['RD'] = df['RD'].rolling(3).mean()
        df['RE'] = df['最低'].rolling(30).min()
        df['RF'] = df['RD'].rolling(30).max()
        
        # 信号判定
        df['R10'] = df['收盘'].rolling(58).mean().notna().astype(int)
        raw_signal = np.where(df['最低'] <= df['RE'], (df['RD'] + df['RF']*2)/2, 0)
        df['S1_Raw_Val'] = raw_signal * df['R10']
        df['S1_Trigger'] = (df['S1_Raw_Val'] > 0).astype(int)
        
        # *** 信号优化：连续标记3天 (如果今天、昨天、前天触发，今天都标Y) ***
        df['S1_Final_Flag'] = df['S1_Trigger'].rolling(window=3, min_periods=1).max()
        df['策略1_大底信号'] = np.where(df['S1_Final_Flag'] > 0, 'Y', '')

        # ==========================================
        # 计算策略 2: EMA波段 (Pullback)
        # ==========================================
        df['VAR1'] = (df['收盘'] + df['最高'] + df['开盘'] + df['最低']) / 4
        # 买入线：32日 EMA 下沉4%
        df['S2_BuyLine'] = calculate_xma(df['VAR1'], 32) * (1 - 4/100)
        
        # 信号生成
        df['策略2_波段信号'] = np.where(df['收盘'] < df['S2_BuyLine'], 'Y', '')

        # ==========================================
        # 数据截取与输出构建
        # ==========================================
        # 截取用户关注的时间段
        result_df = df[start_date:end_date].copy()
        
        if result_df.empty:
            return None

        output_list = []
        for date, row in result_df.iterrows():
            output_list.append({
                '股票代码': stock_info['code'],
                '股票简称': stock_info['name'],      # 对应 B列
                '主营行业': stock_info['industry'],  # 对应 C列
                '地区': stock_info['area'],         # 对应 D列
                '类型': stock_info['type'],         # 对应 E列
                '日期': date.strftime('%Y-%m-%d'),
                '收盘价': round(row['收盘'], 2),
                '策略1_大底(连续3天)': row['策略1_大底信号'],
                '策略2_波段(跌破)': row['策略2_波段信号'],
                'BBI': round(row['BBI'], 2),
                'MA60': round(row['MA60'], 2),
                '波动率(%)': round(row['波动率%'], 2)
            })
            
        return output_list

    except Exception as e:
        print(f"处理 {symbol} 时发生错误: {str(e)}")
        return None

# ==========================================
# 3. 主程序入口
# ==========================================
if __name__ == '__main__':
    # ================= 配置区 =================
    input_file = 'stock_list.xlsx'   # 输入文件名
    output_file = '综合选股结果.xlsx' # 输出文件名
    start_date = '2026-01-28'        # 分析开始日期
    end_date = '2026-01-28'          # 分析结束日期
    # =========================================

    # 1. 读取 Excel 文件
    if not os.path.exists(input_file):
        print(f"错误：找不到文件 {input_file}。请确保文件在当前目录下。")
        exit()

    print(f"正在读取 {input_file} ...")
    try:
        # 使用 usecols 读取 A-E 列
        meta_df = pd.read_excel(input_file, usecols=[0, 1, 2, 3, 4])
        meta_df.columns = ['code', 'name', 'industry', 'area', 'type']
        
        # ==========================================
        # 【关键修复】过滤空行
        # ==========================================
        # 1. 删除代码列为空的行
        meta_df.dropna(subset=['code'], inplace=True)
        # 2. 删除代码列为空字符的行
        meta_df = meta_df[meta_df['code'].astype(str).str.strip() != '']
        
        # 数据清洗：
        # 有时候Excel读取数字会变成 600519.0 (浮点数)，需要去掉 .0
        meta_df['code'] = meta_df['code'].astype(str).str.replace(r'\.0$', '', regex=True)
        # 补齐6位
        meta_df['code'] = meta_df['code'].str.zfill(6)
        
        # 构建待处理列表
        stock_list = meta_df.to_dict('records')
        
        # 打印修正后的数量，这里应该显示 570 左右
        print(f"成功加载 {len(stock_list)} 只有效股票信息。")
        
    except Exception as e:
        print(f"读取 Excel 文件失败: {e}")
        exit()

    # 2. 多线程并发计算
    print(f"开始计算，时间范围: {start_date} 至 {end_date} ...")
    all_results = []
    
    with ThreadPoolExecutor(max_workers=10) as executor:
        # 提交任务
        futures = {executor.submit(process_stock, stock, start_date, end_date): stock['code'] for stock in stock_list}
        
        count = 0
        total = len(futures)
        
        for future in as_completed(futures):
            code = futures[future]
            try:
                data = future.result()
                if data:
                    all_results.extend(data)
                count += 1

                if count % 10 == 0 or count == total:
                    print(f"进度: {count}/{total}")
            except Exception as e:
                print(f"任务执行异常 ({code}): {e}")

    # 3. 结果保存
    if all_results:
        final_df = pd.DataFrame(all_results)
        
        # 调整列顺序，确保前几列是用户要求的信息
        cols_order = [
            '日期', '股票代码', '股票简称', '主营行业', '地区', '类型',
            '收盘价', '策略1_大底(连续3天)', '策略2_波段(跌破)',
            'BBI', 'MA60', '波动率(%)'
        ]
        # 防止列名不匹配（防御性编程）
        cols_order = [c for c in cols_order if c in final_df.columns]
        
        final_df = final_df[cols_order]
        
        # 排序：先按日期，再按股票代码
        final_df.sort_values(by=['日期', '股票代码'], inplace=True)
        
        final_df.to_excel(output_file, index=False)
        print(f"\n========================================")
        print(f"计算完成！结果已保存至: {output_file}")
        print(f"包含列: A:代码, B:简称, C:行业, D:地区, E:类型 及 计算结果")
        print(f"========================================")
    else:
        print("未生成任何有效结果，请检查日期范围或网络连接。")

