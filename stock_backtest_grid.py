import os
import pandas as pd
import numpy as np
from datetime import datetime
from multiprocessing import Pool, cpu_count
import time
import itertools
import warnings
warnings.filterwarnings('ignore')

# --- 配置区域 ---
HISTORY_DATA_DIR = "./history_data"
OUTPUT_DIR = "./"
INITIAL_CAPITAL = 1_000_000.0
NUM_CORES = max(1, cpu_count() - 1)

def parse_input_list(prompt, type_func):
    """解析用户输入的逗号分隔的参数列表"""
    while True:
        raw_input = input(prompt).strip()
        # 兼容中文逗号
        raw_input = raw_input.replace('，', ',')
        try:
            return [type_func(x.strip()) for x in raw_input.split(',')]
        except ValueError:
            print("输入格式有误，请确保用逗号分隔，且输入数字。")

def get_user_inputs():
    """获取用户输入的枚举回测参数"""
    print("\n" + "="*50)
    print("🚀 主升浪(RIGHT_SIDE_PRO) 网格参数寻优系统")
    print("="*50)
    print("提示：以下参数均支持输入多个值进行枚举测试，请用逗号分隔 (如: 10,20,30)")
    
    start_date = input("请输入回测开始日期 (如 2021-01-01): ").strip()
    end_date = input("请输入回测结束日期 (如 2025-12-31): ").strip()
    
    tp_list = parse_input_list("请输入硬止盈百分比范围 (如 15,20,25): ", float)
    sl_list = parse_input_list("请输入硬止损百分比范围 (如 5,8,10): ", float)
    days_list = parse_input_list("请输入最大持仓天数范围 (如 10,20,30): ", int)
    slope_list = parse_input_list("请输入MA20斜率触发阈值范围 (如 20,25,30): ", float)
    
    return start_date, end_date, tp_list, sl_list, days_list, slope_list

def process_single_stock_file(args):
    """单只股票数据预处理（一次性计算好，供后续快速枚举）"""
    file, start_date, end_date = args
    filepath = os.path.join(HISTORY_DATA_DIR, file)
    try:
        df = pd.read_csv(filepath)
        df['日期'] = pd.to_datetime(df['日期'])
        df.sort_values('日期', inplace=True)

        if len(df) < 60: return None

        for c in ['开盘', '收盘', '最高', '最低', '成交量']:
            df[c] = pd.to_numeric(df[c], errors='coerce')

        df['MA5'] = df['收盘'].rolling(5, min_periods=1).mean()
        df['MA10'] = df['收盘'].rolling(10, min_periods=1).mean()
        df['MA20'] = df['收盘'].rolling(20, min_periods=1).mean()
        df['MA60'] = df['收盘'].rolling(60, min_periods=1).mean()
        df['VOL_MA5'] = df['成交量'].rolling(5, min_periods=1).mean()

        # 预先计算独立于参数的指标
        df['MA20_ANGLE'] = np.degrees(np.arctan((df['MA20'] / df['MA20'].shift(1) - 1) * 100))
        cond_trend = (df['收盘'] > df['MA10']) & (df['MA5'] > df['MA20']) & (df['MA20'] > df['MA60']) & (df['MA60'] > df['MA60'].shift(1))
        cond_power = (df['收盘'] / df['收盘'].shift(1) > 1.03) & (df['收盘'] > df['开盘'])
        cond_vol = df['成交量'] > df['VOL_MA5']

        df['DIF'] = df['收盘'].ewm(span=12, adjust=False).mean() - df['收盘'].ewm(span=26, adjust=False).mean()
        df['DEA'] = df['DIF'].ewm(span=9, adjust=False).mean()
        cond_macd = (df['DIF'] > 0) & (df['DIF'] > df['DEA'])

        # 基础买入条件 (不含斜率阈值，斜率在枚举时动态判断)
        df['BASE_BUY'] = cond_trend & cond_power & cond_vol & cond_macd

        # 卖出条件是固定的
        cross_ma10 = (df['收盘'].shift(1) >= df['MA10'].shift(1)) & (df['收盘'] < df['MA10'])
        ma20_bad = (df['MA20_ANGLE'] < 0) & (df['收盘'] < df['MA20'])
        df['SELL_SIGNAL'] = cross_ma10 | ma20_bad
        # =================【新增：涨跌停判定】=================
        stock_code = file.replace('.csv', '')
        limit_threshold = 19.8 if stock_code.startswith('688') or stock_code.startswith('30') else 9.8
        df['pct_change'] = (df['收盘'] / df['收盘'].shift(1) - 1) * 100
        df['is_limit_up'] = df['pct_change'] >= limit_threshold
        df['is_limit_down'] = df['pct_change'] <= -limit_threshold
        # ====================================================
        mask = (df['日期'] >= pd.to_datetime(start_date)) & (df['日期'] <= pd.to_datetime(end_date))
        df = df[mask]
        
        if df.empty: return None

        df['股票代码'] = file.replace('.csv', '')
        
        # 英文列名以便于极速迭代器 itertuples 调用
        df = df[['日期', '股票代码', '开盘', '收盘', 'MA20_ANGLE', 'BASE_BUY', 'SELL_SIGNAL', 'is_limit_up', 'is_limit_down']]
        df.columns = ['date', 'code', 'open', 'close', 'angle', 'base_buy', 'sell_signal', 'is_limit_up', 'is_limit_down']
        return df.copy()

        
    except Exception:
        return None

def run_grid_search():
    start_date, end_date, tp_list, sl_list, days_list, slope_list = get_user_inputs()
    
    stock_files = [f for f in os.listdir(HISTORY_DATA_DIR) if f.endswith('.csv')]
    
    print(f"\n📡 正在预处理 {len(stock_files)} 只股票的历史数据...")
    start_time = time.time()
    
    with Pool(processes=NUM_CORES) as pool:
        args_list = [(file, start_date, end_date) for file in stock_files]
        results = pool.map(process_single_stock_file, args_list)
        
    all_signals = [res for res in results if res is not None and not res.empty]
    if not all_signals:
        print("❌ 在指定日期范围内未找到任何数据，程序退出。")
        return
        
    master_history = pd.concat(all_signals, ignore_index=True)
    
    # 【核心优化】先按日期正序，同日按MA20斜率倒序！同等条件下优先买入斜率最猛的龙头！
    master_history.sort_values(by=['date', 'angle'], ascending=[True, False], inplace=True)
    
    print(f"✅ 数据预处理完成，耗时 {time.time() - start_time:.2f} 秒。共 {len(master_history)} 条日切片数据。")

    # 生成所有参数组合
    combinations = list(itertools.product(tp_list, sl_list, days_list, slope_list))
    total_combos = len(combinations)
    print(f"\n⚙️ 即将开始网格搜索，共需枚举计算 {total_combos} 种参数组合...")
    
    final_results = []
    
    combo_count = 0
    search_start_time = time.time()
    
    # 开始枚举计算
    for tp_pct, sl_pct, max_days, slope_thresh in combinations:
        combo_count += 1
        print(f"正在计算 [{combo_count}/{total_combos}] -> 止盈:{tp_pct}%, 止损:{sl_pct}%, 期限:{max_days}天, 斜率:{slope_thresh}°", end='\r')
        
        cash = INITIAL_CAPITAL
        holdings = {}
        total_trades = 0
        winning_trades = 0
        last_close_prices = {} # 用于记录期末未平仓股票的最新价
        
        tp_ratio = tp_pct / 100.0
        sl_ratio = sl_pct / 100.0
        
        # 极速遍历算法
        for row in master_history.itertuples(index=False):
            code = row.code
            close_price = row.close
            open_price = row.open
            is_limit_up = row.is_limit_up
            is_limit_down = row.is_limit_down
            # 更新股票的最新价格
            last_close_prices[code] = close_price
            
            # --- 卖出判断 ---
            if code in holdings:
                info = holdings[code]
                info['days_held'] += 1
                sell_reason = False
                
                if open_price != 0:
                    profit_ratio = (open_price / info['buy_price']) - 1
                    if profit_ratio >= tp_ratio or profit_ratio <= -sl_ratio:
                        sell_reason = True
                
                if not sell_reason and (info['days_held'] >= max_days or row.sell_signal):
                    sell_reason = True
                    
                if sell_reason:
                    # 【新增跌停拦截】
                    if is_limit_down:
                        continue # 🔒跌停无法卖出，强制继续持有
                    # 判断是以开盘价还是收盘价卖出
                    if open_price != 0 and (profit_ratio >= tp_ratio or profit_ratio <= -sl_ratio):
                        sell_price = open_price
                    else:
                        sell_price = close_price
                        
                    proceeds = info['shares'] * sell_price
                    cash += proceeds
                    
                    pnl_percent = (sell_price / info['buy_price'] - 1) * 100
                    total_trades += 1
                    if pnl_percent > 0: winning_trades += 1
                        
                    del holdings[code]
            
            # --- 买入判断 ---
            # 动态判定当前斜率是否大于本轮枚举的阈值
            buy_signal = row.base_buy and (row.angle > slope_thresh)
            
            if buy_signal and code not in holdings:
                # 【新增涨停拦截】
                if is_limit_up:
                    continue # 🚫涨停无法买入，直接跳过
                code_str = str(code).zfill(6)
                min_lot = 200 if code_str.startswith('688') else 100
                
                max_shares = int(min(INITIAL_CAPITAL * 0.20, cash) // close_price)
                shares_to_buy = (max_shares // min_lot) * min_lot
                
                if shares_to_buy >= min_lot:
                    cash -= shares_to_buy * close_price
                    holdings[code] = {
                        'shares': shares_to_buy,
                        'buy_price': close_price,
                        'days_held': 0
                    }
                    
        # 计算本轮组合的最终净值
        final_value = cash
        for code, info in holdings.items():
            final_value += info['shares'] * last_close_prices.get(code, info['buy_price'])
            
        total_pnl = final_value - INITIAL_CAPITAL
        return_pct = (final_value / INITIAL_CAPITAL - 1) * 100
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0.0
        
        final_results.append({
            '回测开始日期': start_date,
            '回测结束日期': end_date,
            '止盈(%)': tp_pct,
            '止损(%)': sl_pct,
            '最大持仓(天)': max_days,
            '斜率阈值(°)': slope_thresh,
            '绝对盈亏(元)': round(total_pnl, 2),
            '总收益率(%)': round(return_pct, 2),
            '总交易笔数': total_trades,
            '胜率(%)': round(win_rate, 2)
        })

    print(f"\n\n🎉 网格搜索计算完成！总计耗时 {time.time() - search_start_time:.2f} 秒。")
    
    # 保存结果并按收益率排序
    res_df = pd.DataFrame(final_results)
    res_df.sort_values(by='总收益率(%)', ascending=False, inplace=True)
    
    csv_filename = f"grid_search_{start_date}_to_{end_date}.csv"
    res_df.to_csv(os.path.join(OUTPUT_DIR, csv_filename), index=False, encoding='utf-8-sig')
    
    print("="*50)
    print("🏆 最优参数组合 Top 3 🏆")
    print("="*50)
    print(res_df.head(3).to_string(index=False))
    print("="*50)
    print(f"📄 完整的全量枚举结果已保存至: {csv_filename}")

if __name__ == "__main__":
    if not os.path.exists(HISTORY_DATA_DIR):
        print(f"错误: 找不到 {HISTORY_DATA_DIR} 目录。请先运行下载脚本！")
        exit()
    run_grid_search()