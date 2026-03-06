import os
import pandas as pd
import numpy as np
from datetime import datetime
from multiprocessing import Pool, cpu_count
import time
import warnings
warnings.filterwarnings('ignore') # 忽略pandas的一些计算警告

# --- 配置区域 ---
# 数据文件存放的根目录 (请确保与您下载历史数据的目录一致)
HISTORY_DATA_DIR = "./history_data"
# 结果文件保存目录
OUTPUT_DIR = "./"
# 初始资金
INITIAL_CAPITAL = 1_000_000.0
# 使用的核心数量 (留1个核心给系统防卡顿)
NUM_CORES = max(1, cpu_count() - 1) 

def get_user_inputs():
    """获取用户输入的回测参数"""
    print("\n" + "="*50)
    print("🚀 左侧伏击(LEFT_SIDE_PRO) 专项回测与流水生成系统")
    print("="*50)
    start_date = input("请输入回测开始日期 (如 2021-01-01): ").strip()
    end_date = input("请输入回测结束日期 (如 2025-12-31): ").strip()
    
    p1 = float(input("请输入上轨偏移率 P1 (止盈卖出用, 如 4): "))
    p2 = float(input("请输入下轨偏移率 P2 (抄底买入用, 如 10): "))
    bias_thresh = float(input("请输入负乖离率 BIAS_OK (输入正数, 如 8 代表 <-8%): "))
    
    return start_date, end_date, p1, p2, bias_thresh

def process_single_stock_file(args):
    """单只股票处理引擎：100% 对齐通达信左侧伏击公式"""
    file, start_date, end_date, p1, p2, bias_thresh = args
    filepath = os.path.join(HISTORY_DATA_DIR, file)
    try:
        df = pd.read_csv(filepath)
        df['日期'] = pd.to_datetime(df['日期'])
        df.sort_values('日期', inplace=True)

        if len(df) < 60:
            return None

        for c in ['开盘', '收盘', '最高', '最低', '成交量']:
            df[c] = pd.to_numeric(df[c], errors='coerce')
            
        df.dropna(subset=['开盘', '收盘', '最高', '最低', '成交量'], inplace=True)

        # ========== 1. 核心轨道基础 ==========
        df['VAR1'] = (df['收盘'] + df['最高'] + df['开盘'] + df['最低']) / 4
        # 使用 ewm(adjust=False) 对应通达信的 EMA
        df['MID'] = df['VAR1'].ewm(span=32, adjust=False).mean()
        df['UPPER'] = df['MID'] * (1 + p1 / 100.0)
        df['LOWER'] = df['MID'] * (1 - p2 / 100.0)

        # ========== 2. 动能与乖离率 ==========
        df['MA20'] = df['收盘'].rolling(20, min_periods=1).mean()
        df['BIAS_VAL'] = (df['收盘'] - df['MA20']) / df['MA20'] * 100
        df['BIAS_OK'] = df['BIAS_VAL'] < -bias_thresh

        # ========== 3. MACD 趋势滤网 ==========
        df['DIF'] = df['收盘'].ewm(span=12, adjust=False).mean() - df['收盘'].ewm(span=26, adjust=False).mean()
        df['DEA'] = df['DIF'].ewm(span=9, adjust=False).mean()
        df['UP_TREND'] = (df['DIF'] > 0) & (df['DEA'] > 0) & (df['DIF'] > df['DEA']) # 主升浪判定

        # ========== 4. 买入信号 (左侧伏击) ==========
        b_cond1 = (df['最低'] <= df['LOWER']) & df['BIAS_OK']
        # 收阳且下影线长于上影线
        b_cond2 = (df['收盘'] > df['开盘']) & ((df['收盘'] - df['最低']) > (df['最高'] - df['收盘']))
        df['BUY_SIGNAL'] = b_cond1 & b_cond2

        # ========== 5. 卖出信号 (常规落袋) ==========
        s_cond1 = df['最高'] >= df['UPPER']
        body = (df['收盘'] - df['开盘']).abs()
        upper_shadow = df['最高'] - df[['收盘', '开盘']].max(axis=1)
        # 收阴或长上影线
        s_cond2 = (df['收盘'] < df['开盘']) | (upper_shadow > body * 1.5)
        vol_shrink = df['成交量'] < df['成交量'].shift(1)
        
        # 综合常规卖出，并加入MACD滤网防卖飞
        df['SELL_SIGNAL'] = s_cond1 & s_cond2 & vol_shrink & (~df['UP_TREND'])

        # ========== 涨跌停判定 ==========
        stock_code = file.replace('.csv', '')
        limit_threshold = 19.8 if stock_code.startswith('688') or stock_code.startswith('30') else 9.8
        df['pct_change'] = (df['收盘'] / df['收盘'].shift(1) - 1) * 100
        df['is_limit_up'] = df['pct_change'] >= limit_threshold
        df['is_limit_down'] = df['pct_change'] <= -limit_threshold

        # --- 截取用户指定的回测时间段 ---
        mask = (df['日期'] >= pd.to_datetime(start_date)) & (df['日期'] <= pd.to_datetime(end_date))
        df = df[mask]
        
        if df.empty:
            return None

        df['股票代码'] = stock_code
        return df[['日期', '股票代码', '开盘', '收盘', '最高', '最低', 
                   'BIAS_VAL', 'BUY_SIGNAL', 'SELL_SIGNAL', 'is_limit_up', 'is_limit_down']].copy()
        
    except Exception as e:
        return None

def run_backtest():
    start_date, end_date, p1, p2, bias_thresh = get_user_inputs()
    stock_files = [f for f in os.listdir(HISTORY_DATA_DIR) if f.endswith('.csv')]
    
    print(f"\n🚀 开始并行处理 {len(stock_files)} 个股票文件...")
    start_time = time.time()

    args_list = [(file, start_date, end_date, p1, p2, bias_thresh) for file in stock_files]

    with Pool(processes=NUM_CORES) as pool:
        results = pool.map(process_single_stock_file, args_list)
    
    print(f"✅ 数据处理完成，耗时 {time.time() - start_time:.2f} 秒。开始撮合交易...")

    all_signals = [result for result in results if result is not None and not result.empty]
    if not all_signals:
        print("❌ 在指定日期范围内没有找到任何有效数据。")
        return

    full_history = pd.concat(all_signals, ignore_index=True)
    # 【核心优化】：按日期正序。同日有多只股票触发时，优先买入 BIAS 最负（跌得最惨）的股票！
    full_history.sort_values(by=['日期', 'BIAS_VAL'], ascending=[True, True], inplace=True)

    cash = INITIAL_CAPITAL
    holdings = {} 
    trade_log = [] 

    # 模拟每日逐笔交易
    for index, row in full_history.iterrows():
        current_date = row['日期']
        stock_code = row['股票代码']
        buy_signal = row['BUY_SIGNAL']
        sell_signal = row['SELL_SIGNAL']
        close_price = row['收盘']
        low_price = row['最低']
        is_limit_up = row['is_limit_up']
        is_limit_down = row['is_limit_down']

        # --- 1. 检查卖出与止损条件 ---
        if stock_code in holdings:
            holding_info = holdings[stock_code]
            holding_info['days_held'] += 1
            sell_reason = ""
            
            # 纪律止损：买入15日内，跌破了买入当天的最低价
            if holding_info['days_held'] <= 15 and close_price < holding_info['buy_day_low']:
                sell_reason = "破位止损(破抄底价)"
            elif sell_signal:
                sell_reason = "S_落袋(触碰上轨)"
            # 兜底风控：持仓超过30天依然没触发止盈也没跌破底线，强制换仓
           
            
            if sell_reason:
                if is_limit_down:
                    continue # 🔒跌停板锁死，无法卖出
                    
                shares_to_sell = holding_info['shares']
                sell_price = close_price # 采用收盘价搓合
                proceeds = shares_to_sell * sell_price
                
                cash += proceeds
                pnl_amount = proceeds - holding_info['cost_basis']
                pnl_percent = (sell_price / holding_info['buy_price'] - 1) * 100

                trade_log.append({
                    "Date": current_date.strftime('%Y-%m-%d'),
                    "StockCode": stock_code,
                    "Action": "SELL",
                    "Shares": shares_to_sell,
                    "Price": round(sell_price, 2),
                    "Amount": round(proceeds, 2),
                    "PnL_Amount": round(pnl_amount, 2),
                    "PnL_Percent": round(pnl_percent, 2),
                    "Reason": sell_reason,
                    "Cash_Remaining": round(cash, 2)
                })
                del holdings[stock_code]

        # --- 2. 检查买入条件 ---
        if buy_signal and stock_code not in holdings:
            if is_limit_up:
                continue # 🚫涨停板封死无法抄底
                
            stock_code_str = str(stock_code).zfill(6)
            price_to_buy = close_price 
            
            min_lot_size = 200 if stock_code_str.startswith('688') else 100
            
            # 左侧风控：单只股票最多占用总资金的 20%，不要一次性满仓补底
            max_shares_for_20_percent = int((INITIAL_CAPITAL * 0.20) // price_to_buy)
            max_shares_for_cash = int(cash // price_to_buy)
            shares_before_rounding = min(max_shares_for_20_percent, max_shares_for_cash)
            
            shares_to_buy = (shares_before_rounding // min_lot_size) * min_lot_size
            
            if shares_to_buy >= min_lot_size:
                cost = shares_to_buy * price_to_buy
                cash -= cost

                holdings[stock_code] = {
                    'shares': shares_to_buy,
                    'buy_price': price_to_buy,
                    'buy_date': current_date,
                    'days_held': 0,
                    'cost_basis': cost,
                    'buy_day_low': low_price # 极其关键：记录抄底防守线
                }

                trade_log.append({
                    "Date": current_date.strftime('%Y-%m-%d'),
                    "StockCode": stock_code,
                    "Action": "BUY",
                    "Shares": shares_to_buy,
                    "Price": round(price_to_buy, 2),
                    "Amount": round(cost, 2),
                    "PnL_Amount": 0,
                    "PnL_Percent": 0,
                    "Reason": "B_伏击(乖离超跌)",
                    "Cash_Remaining": round(cash, 2)
                })

    # --- 回测结束计算 ---
    final_value = cash
    for stock, info in holdings.items():
        last_close = full_history[full_history['股票代码']==stock]['收盘'].iloc[-1]
        final_value += info['shares'] * last_close

    total_pnl = final_value - INITIAL_CAPITAL
    total_return_pct = (final_value / INITIAL_CAPITAL - 1) * 100

    print("\n" + "="*45)
    print("📉 左侧伏击策略 回测结算单")
    print("="*45)
    print(f"回测区间:   {start_date} 至 {end_date}")
    print(f"核心参数:   P1={p1}%, P2={p2}%, 负乖离率<{-bias_thresh}%")
    print(f"初始资金:   ¥{INITIAL_CAPITAL:,.2f}")
    print(f"期末总值:   ¥{final_value:,.2f} (含剩余持仓估值)")
    print(f"绝对盈亏:   ¥{total_pnl:,.2f}")
    print(f"总收益率:   {total_return_pct:.2f}%")
    
    if trade_log:
        log_df = pd.DataFrame(trade_log)
        sell_trades = log_df[log_df['Action'] == 'SELL']
        if not sell_trades.empty:
            win_rate = len(sell_trades[sell_trades['PnL_Percent'] > 0]) / len(sell_trades)
            print(f"总交易笔数: {len(sell_trades)} 笔 (已完成买卖)")
            print(f"策略胜率:   {win_rate*100:.2f}%")
        print("="*45)
        
        log_filename = f"trade_log_LeftSide_{start_date}_to_{end_date}.csv"
        output_path = os.path.join(OUTPUT_DIR, log_filename)
        log_df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"📄 详细逐笔交易流水已保存至: {output_path}")
    else:
        print("="*45)
        print("\n⚠️ 在设定的参数下，本次回测期间没有捕捉到符合要求的超跌买点。")

if __name__ == "__main__":
    if not os.path.exists(HISTORY_DATA_DIR):
        print(f"错误: 找不到历史数据目录 {HISTORY_DATA_DIR}。请先准备好数据！")
        exit()
        
    run_backtest()