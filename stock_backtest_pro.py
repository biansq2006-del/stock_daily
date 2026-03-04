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
    print("--- 🚀 主升浪(RIGHT_SIDE_PRO) 专项回测系统 ---")
    start_date = input("请输入回测开始日期 (如 2021-01-01): ")
    end_date = input("请输入回测结束日期 (如 2025-12-31): ")
    
    take_profit_pct = float(input("请输入硬止盈百分比 (如 20 表示 20%): ")) / 100.0
    stop_loss_pct = float(input("请输入硬止损百分比 (如 8 表示 -8%): ")) / 100.0
    max_holding_days = int(input("请输入最大持仓天数 (如 30): "))
    slope_threshold = float(input("请输入MA20斜率触发阈值 (推荐 25): "))
    
    return start_date, end_date, take_profit_pct, stop_loss_pct, max_holding_days, slope_threshold

def process_single_stock_file(args):
    """单只股票处理引擎：100% 对齐通达信 RIGHT_SIDE_PRO 公式"""
    file, start_date, end_date, slope_threshold = args
    filepath = os.path.join(HISTORY_DATA_DIR, file)
    try:
        df = pd.read_csv(filepath)
        df['日期'] = pd.to_datetime(df['日期'])
        df.sort_values('日期', inplace=True)

        if len(df) < 60:
            return None

        for c in ['开盘', '收盘', '最高', '最低', '成交量']:
            df[c] = pd.to_numeric(df[c], errors='coerce')

        # ========== 基础均线 ==========
        df['MA5'] = df['收盘'].rolling(5, min_periods=1).mean()
        df['MA10'] = df['收盘'].rolling(10, min_periods=1).mean()
        df['MA20'] = df['收盘'].rolling(20, min_periods=1).mean()
        df['MA60'] = df['收盘'].rolling(60, min_periods=1).mean()
        df['VOL_MA5'] = df['成交量'].rolling(5, min_periods=1).mean()

        # ========== 核心逻辑 ==========
        
        # 1. 角度过滤：计算MA20斜率转角度
        df['MA20_ANGLE'] = np.degrees(np.arctan((df['MA20'] / df['MA20'].shift(1) - 1) * 100))
        cond_angle = df['MA20_ANGLE'] > slope_threshold

        # 2. 多头排列：C>MA10 AND MA5>MA20 AND MA20>MA60 AND MA60向上
        cond_trend = (df['收盘'] > df['MA10']) & \
                     (df['MA5'] > df['MA20']) & \
                     (df['MA20'] > df['MA60']) & \
                     (df['MA60'] > df['MA60'].shift(1))

        # 3. 动能爆发：涨幅 > 3% 且 阳线
        cond_power = (df['收盘'] / df['收盘'].shift(1) > 1.03) & (df['收盘'] > df['开盘'])

        # 4. 量能确认：成交量 > 5日均量
        cond_vol = df['成交量'] > df['VOL_MA5']

        # 5. MACD 水上金叉/多头
        # 注意：通达信的 EMA 等同于 Pandas的 ewm(adjust=False)
        df['DIF'] = df['收盘'].ewm(span=12, adjust=False).mean() - df['收盘'].ewm(span=26, adjust=False).mean()
        df['DEA'] = df['DIF'].ewm(span=9, adjust=False).mean()
        cond_macd = (df['DIF'] > 0) & (df['DIF'] > df['DEA'])

        # ========== 最终买点 ==========
        df['BUY_SIGNAL'] = cond_angle & cond_trend & cond_power & cond_vol & cond_macd

       # ========== 最终卖点 ==========
        # CROSS(MA10, C) -> 昨天收盘价>=昨天MA10，今天收盘价<今天MA10 (跌破10日线)
        cross_ma10 = (df['收盘'].shift(1) >= df['MA10'].shift(1)) & (df['收盘'] < df['MA10'])
        # (MA20_ANGLE < 0 AND C < MA20) -> 均线拐头向下且股价在20日线下
        ma20_bad = (df['MA20_ANGLE'] < 0) & (df['收盘'] < df['MA20'])
        
        df['SELL_SIGNAL'] = cross_ma10 | ma20_bad

        # ========== 涨跌停判定 (新增) ==========
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
        # 修改后：把 is_limit_up 和 is_limit_down 一起传给回测引擎
        return df[['日期', '股票代码', '开盘', '收盘', '最高', '最低', '成交量', 'MA20_ANGLE', 'BUY_SIGNAL', 'SELL_SIGNAL', 'is_limit_up', 'is_limit_down']].copy()
       
        
    except Exception as e:
        return None

def run_backtest(stock_files, start_date, end_date, take_profit_pct, stop_loss_pct, max_holding_days, slope_threshold):
    print(f"\n🚀 开始并行处理 {len(stock_files)} 个股票文件...")
    start_time = time.time()

    args_list = [(file, start_date, end_date, slope_threshold) for file in stock_files]

    with Pool(processes=NUM_CORES) as pool:
        results = pool.map(process_single_stock_file, args_list)
    
    print(f"✅ 数据处理完成，耗时 {time.time() - start_time:.2f} 秒。开始撮合交易...")

    all_signals = [result for result in results if result is not None and not result.empty]
    if not all_signals:
        print("❌ 在指定日期范围内没有找到任何有效数据。")
        return

    full_history = pd.concat(all_signals, ignore_index=True)
    # 修改后：按日期正序，同日按斜率倒序，优先买入最猛的龙头！
    full_history.sort_values(by=['日期', 'MA20_ANGLE'], ascending=[True, False], inplace=True)

    cash = INITIAL_CAPITAL
    holdings = {} 
    trade_log = [] 

    for index, row in full_history.iterrows():
        current_date = row['日期']
        stock_code = row['股票代码']
        buy_signal = row['BUY_SIGNAL']
        sell_signal = row['SELL_SIGNAL']
        open_price = row['开盘']
        close_price = row['收盘']
        is_limit_up = row['is_limit_up']       # <--- 新增
        is_limit_down = row['is_limit_down']   # <--- 新增

        # --- 1. 检查卖出条件 ---
        if stock_code in holdings:
            holding_info = holdings[stock_code]
            holding_info['days_held'] += 1
            sell_reason = ""
            
            if open_price != 0:
                profit_ratio = (open_price / holding_info['buy_price']) - 1
                if profit_ratio >= take_profit_pct: sell_reason = "止盈"
                elif profit_ratio <= -stop_loss_pct: sell_reason = "止损"

            if not sell_reason and holding_info['days_held'] >= max_holding_days:
                sell_reason = f"超时强平({max_holding_days}天)"

            if not sell_reason and sell_signal:
                sell_reason = "策略S点(破均线)"
            
            if sell_reason:
                if is_limit_down:
                    continue # 🔒【新增】跌停板锁死，今天想卖也卖不掉，硬扛到明天！
                shares_to_sell = holding_info['shares']
                sell_price = open_price if sell_reason in ["止盈", "止损"] else close_price
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
                continue # 🚫【新增】涨停板封死，买不进，直接跳过寻找下一个！
            stock_code_str = str(stock_code).zfill(6)
            price_to_buy = close_price 
            
            min_lot_size = 200 if stock_code_str.startswith('688') else 100
            # 严格风控：单只股票最多占用总资金的 20%
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
                    'cost_basis': cost
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
                    "Reason": "主升浪启动",
                    "Cash_Remaining": round(cash, 2)
                })

    # --- 回测结束计算 ---
    final_value = cash
    for stock, info in holdings.items():
        last_close = full_history[full_history['股票代码']==stock]['收盘'].iloc[-1]
        final_value += info['shares'] * last_close

    total_pnl = final_value - INITIAL_CAPITAL
    total_return_pct = (final_value / INITIAL_CAPITAL - 1) * 100

    print("\n" + "="*40)
    print("📈 主升浪策略 回测结算单")
    print("="*40)
    print(f"初始资金:   ¥{INITIAL_CAPITAL:,.2f}")
    print(f"期末总值:   ¥{final_value:,.2f} (含剩余持仓估值)")
    print(f"绝对盈亏:   ¥{total_pnl:,.2f}")
    print(f"总收益率:   {total_return_pct:.2f}%")
    
    if trade_log:
        log_df = pd.DataFrame(trade_log)
        sell_trades = log_df[log_df['Action'] == 'SELL']
        if not sell_trades.empty:
            win_rate = len(sell_trades[sell_trades['PnL_Percent'] > 0]) / len(sell_trades)
            print(f"总交易笔数: {len(sell_trades)} 笔 (完整买卖)")
            print(f"策略胜率:   {win_rate*100:.2f}%")
        print("="*40)
        
        log_filename = f"backtest_MainWave_{start_date}_to_{end_date}.csv"
        output_path = os.path.join(OUTPUT_DIR, log_filename)
        log_df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"📄 详细交易记录已保存至: {output_path}")
    else:
        print("="*40)
        print("\n⚠️ 在本次回测期间内没有产生任何满足要求的交易。")

if __name__ == "__main__":
    if not os.path.exists(HISTORY_DATA_DIR):
        print(f"错误: 找不到历史数据目录 {HISTORY_DATA_DIR}。请先运行下载脚本获取数据！")
        exit()
        
    start_date, end_date, take_profit_pct, stop_loss_pct, max_holding_days, slope_threshold = get_user_inputs()
    stock_files = [f for f in os.listdir(HISTORY_DATA_DIR) if f.endswith('.csv')]
    run_backtest(stock_files, start_date, end_date, take_profit_pct, stop_loss_pct, max_holding_days, slope_threshold)