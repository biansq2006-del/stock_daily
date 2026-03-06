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
        raw_input = raw_input.replace('，', ',')
        try:
            return [type_func(x.strip()) for x in raw_input.split(',')]
        except ValueError:
            print("输入格式有误，请确保用逗号分隔，且输入数字。")

def get_user_inputs():
    """获取用户输入的枚举回测参数"""
    print("\n" + "="*60)
    print("🚀 左侧伏击 (LEFT_SIDE) 网格参数寻优系统")
    print("="*60)
    print("提示：以下参数均支持输入多个值进行枚举测试，请用逗号分隔 (如: 8,10,12)")
    
    start_date = input("请输入回测开始日期 (如 2021-01-01): ").strip()
    end_date = input("请输入回测结束日期 (如 2025-12-31): ").strip()
    
    p1_list = parse_input_list("请输入上轨偏移率 P1 范围(卖出用, 如 4,6,8): ", float)
    p2_list = parse_input_list("请输入下轨偏移率 P2 范围(买入用, 如 8,10,12): ", float)
    bias_list = parse_input_list("请输入负乖离率 BIAS_OK 范围(输入正数, 代表 <-x%, 如 6,8,10): ", float)
    
    return start_date, end_date, p1_list, p2_list, bias_list

def process_single_stock_file(args):
    """单只股票数据预处理（计算无需动态变化的指标）"""
    file, start_date, end_date = args
    filepath = os.path.join(HISTORY_DATA_DIR, file)
    try:
        df = pd.read_csv(filepath)
        df['日期'] = pd.to_datetime(df['日期'])
        df.sort_values('日期', inplace=True)

        if len(df) < 60: return None

        for c in ['开盘', '收盘', '最高', '最低', '成交量']:
            df[c] = pd.to_numeric(df[c], errors='coerce')
        
        # 处理异常值
        df.dropna(subset=['开盘', '收盘', '最高', '最低', '成交量'], inplace=True)

        # 1. 核心轨道基础 VAR1 & MID
        df['VAR1'] = (df['收盘'] + df['最高'] + df['开盘'] + df['最低']) / 4
        df['MID'] = df['VAR1'].ewm(span=32, adjust=False).mean()
        
        # 2. 乖离率基础 MA20 & BIAS
        df['MA20'] = df['收盘'].rolling(20, min_periods=1).mean()
        df['BIAS_VAL'] = (df['收盘'] - df['MA20']) / df['MA20'] * 100
        
        # 3. 买入形态 (收阳且下影线长于上影线)
        df['B_COND2'] = (df['收盘'] > df['开盘']) & ((df['收盘'] - df['最低']) > (df['最高'] - df['收盘']))
        
        # 4. 卖出形态 (收阴或长上影) & 缩量
        body = (df['收盘'] - df['开盘']).abs()
        upper_shadow = df['最高'] - df[['收盘', '开盘']].max(axis=1)
        df['S_COND2'] = (df['收盘'] < df['开盘']) | (upper_shadow > body * 1.5)
        df['vol_shrink'] = df['成交量'] < df['成交量'].shift(1)
        
        # 5. MACD趋势滤网
        df['DIF'] = df['收盘'].ewm(span=12, adjust=False).mean() - df['收盘'].ewm(span=26, adjust=False).mean()
        df['DEA'] = df['DIF'].ewm(span=9, adjust=False).mean()
        df['up_trend'] = (df['DIF'] > 0) & (df['DEA'] > 0) & (df['DIF'] > df['DEA'])

        # 涨跌停判定
        stock_code = file.replace('.csv', '')
        limit_threshold = 19.8 if stock_code.startswith('688') or stock_code.startswith('30') else 9.8
        df['pct_change'] = (df['收盘'] / df['收盘'].shift(1) - 1) * 100
        df['is_limit_up'] = df['pct_change'] >= limit_threshold
        df['is_limit_down'] = df['pct_change'] <= -limit_threshold

        mask = (df['日期'] >= pd.to_datetime(start_date)) & (df['日期'] <= pd.to_datetime(end_date))
        df = df[mask]
        
        if df.empty: return None

        df['股票代码'] = stock_code
        
        # 提取极速迭代所需列
        df = df[['日期', '股票代码', '开盘', '收盘', '最高', '最低', 
                 'MID', 'BIAS_VAL', 'B_COND2', 'S_COND2', 'vol_shrink', 'up_trend', 
                 'is_limit_up', 'is_limit_down']]
        df.columns = ['date', 'code', 'open', 'close', 'high', 'low', 
                      'mid', 'bias_val', 'b_cond2', 's_cond2', 'vol_shrink', 'up_trend', 
                      'is_limit_up', 'is_limit_down']
        return df.copy()

    except Exception:
        return None

def run_grid_search():
    start_date, end_date, p1_list, p2_list, bias_list = get_user_inputs()
    
    stock_files = [f for f in os.listdir(HISTORY_DATA_DIR) if f.endswith('.csv')]
    
    print(f"\n📡 正在预处理 {len(stock_files)} 只股票的历史数据...")
    start_time = time.time()
    
    with Pool(processes=NUM_CORES) as pool:
        args_list = [(file, start_date, end_date) for file in stock_files]
        results = pool.map(process_single_stock_file, args_list)
        
    all_signals = [res for res in results if res is not None and not res.empty]
    if not all_signals:
        print("❌ 在指定日期范围内未找到任何有效数据，程序退出。")
        return
        
    master_history = pd.concat(all_signals, ignore_index=True)
    
    # 【排序核心】：按日期正序。同日触发时，优先买入 BIAS 最负（跌得最狠）的股票
    master_history.sort_values(by=['date', 'bias_val'], ascending=[True, True], inplace=True)
    
    print(f"✅ 数据预处理完成，耗时 {time.time() - start_time:.2f} 秒。共 {len(master_history)} 条日切片数据。")

    combinations = list(itertools.product(p1_list, p2_list, bias_list))
    total_combos = len(combinations)
    print(f"\n⚙️ 即将开始左侧网格搜索，共需枚举计算 {total_combos} 种参数组合...")
    
    final_results = []
    combo_count = 0
    search_start_time = time.time()
    
    # 开始枚举计算
    for p1, p2, bias_thresh in combinations:
        combo_count += 1
        print(f"正在计算 [{combo_count}/{total_combos}] -> P1(上轨):{p1}%, P2(下轨):{p2}%, 负乖离:{bias_thresh}%", end='\r')
        
        cash = INITIAL_CAPITAL
        holdings = {}
        total_trades = 0
        winning_trades = 0
        last_close_prices = {} 
        
        # 极速遍历算法
        for row in master_history.itertuples(index=False):
            code = row.code
            close_price = row.close
            high_price = row.high
            low_price = row.low
            
            last_close_prices[code] = close_price
            
            # --- 卖出逻辑 ---
            if code in holdings:
                info = holdings[code]
                info['days_held'] += 1
                
                # 动态计算上轨卖出线
                upper_line = row.mid * (1 + p1 / 100.0)
                
                # 卖出条件判断
                s_cond1 = high_price >= upper_line
                regular_sell = s_cond1 and row.s_cond2 and row.vol_shrink
                sell_signal = regular_sell and (not row.up_trend) # MACD滤网防止卖飞
                
                # 破位止损判断 (绑定买入动作，15日内跌破买入当日最低价)
                stop_loss = (info['days_held'] <= 15) and (close_price < info['buy_day_low'])
                
                if sell_signal or stop_loss:
                    if row.is_limit_down:
                        continue # 跌停封死无法卖出
                        
                    sell_price = close_price # 回测简化：统一按收盘价撮合
                    proceeds = info['shares'] * sell_price
                    cash += proceeds
                    
                    pnl_percent = (sell_price / info['buy_price'] - 1) * 100
                    total_trades += 1
                    if pnl_percent > 0: winning_trades += 1
                        
                    del holdings[code]
            
            # --- 买入逻辑 ---
            if code not in holdings:
                # 动态计算下轨买入线
                lower_line = row.mid * (1 - p2 / 100.0)
                
                # 买入条件判断
                bias_ok = row.bias_val < -bias_thresh
                b_cond1 = (low_price <= lower_line) and bias_ok
                buy_signal = b_cond1 and row.b_cond2
                
                if buy_signal:
                    if row.is_limit_up:
                        continue # 涨停封死无法买入
                        
                    code_str = str(code).zfill(6)
                    min_lot = 200 if code_str.startswith('688') else 100
                    
                    # 仓位控制：单只股票最多占用总资金的 20%
                    max_shares = int(min(INITIAL_CAPITAL * 0.20, cash) // close_price)
                    shares_to_buy = (max_shares // min_lot) * min_lot
                    
                    if shares_to_buy >= min_lot:
                        cash -= shares_to_buy * close_price
                        holdings[code] = {
                            'shares': shares_to_buy,
                            'buy_price': close_price,
                            'days_held': 0,
                            'buy_day_low': low_price # 极其关键：记录抄底防守线
                        }
                        
        # 计算期末净值
        final_value = cash
        for code, info in holdings.items():
            final_value += info['shares'] * last_close_prices.get(code, info['buy_price'])
            
        total_pnl = final_value - INITIAL_CAPITAL
        return_pct = (final_value / INITIAL_CAPITAL - 1) * 100
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0.0
        
        final_results.append({
            '回测开始': start_date,
            '回测结束': end_date,
            'P1_上轨偏移(%)': p1,
            'P2_下轨偏移(%)': p2,
            '负乖离要求(<-%)': bias_thresh,
            '绝对盈亏(元)': round(total_pnl, 2),
            '总收益率(%)': round(return_pct, 2),
            '总交易笔数': total_trades,
            '胜率(%)': round(win_rate, 2)
        })

    print(f"\n\n🎉 左侧网格搜索计算完成！总计耗时 {time.time() - search_start_time:.2f} 秒。")
    
    res_df = pd.DataFrame(final_results)
    # 按综合收益率降序排列
    res_df.sort_values(by='总收益率(%)', ascending=False, inplace=True)
    
    csv_filename = f"leftside_grid_{start_date}_to_{end_date}.csv"
    res_df.to_csv(os.path.join(OUTPUT_DIR, csv_filename), index=False, encoding='utf-8-sig')
    
    print("="*60)
    print("🏆 最优参数组合 Top 3 🏆")
    print("="*60)
    print(res_df.head(3).to_string(index=False))
    print("="*60)
    print(f"📄 完整的全量枚举结果已保存至: {csv_filename}")

if __name__ == "__main__":
    if not os.path.exists(HISTORY_DATA_DIR):
        print(f"错误: 找不到 {HISTORY_DATA_DIR} 目录。请确保历史数据已存在！")
        exit()
    run_grid_search()