import time
import datetime
import os
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio
from concurrent.futures import ThreadPoolExecutor, as_completed
from mootdx.quotes import Quotes

# ==========================================
# 1. 核心算法函数定义
# ==========================================
def sma(series, n, m):
    sma_values = []
    series_array = series.values
    val = np.nan
    for i, x in enumerate(series_array):
        if i < n - 1:
            sma_values.append(np.nan)
        elif i == n - 1:
            val = np.nanmean(series_array[:n])
            sma_values.append(val)
        else:
            if np.isnan(val):
                val = np.nanmean(series_array[:i+1])
            else:
                val = (x * m + val * (n - m)) / n
            sma_values.append(val)
    return pd.Series(sma_values, index=series.index)

def calculate_xma(series, window):
    # 【修复】完全对齐通达信的 XMA 算法 (向历史平移)
    shift_num = int((window - 1) / 2)
    xma = series.rolling(window=window, min_periods=1).mean().shift(-shift_num)
    ma_fallback = series.rolling(window=window, min_periods=1).mean()
    return xma.fillna(ma_fallback)

# ==========================================
# 2. 通达信动态前复权 (QFQ) 算法核心
# ==========================================
def adjust_qfq_for_tdx(df_kline, df_xdxr):
    if df_xdxr is None or df_xdxr.empty:
        return df_kline
    df_xdxr = df_xdxr[df_xdxr['category'] == 1].copy()
    if df_xdxr.empty:
        return df_kline
    df_xdxr['date'] = pd.to_datetime(df_xdxr['year'].astype(str) + '-' + df_xdxr['month'].astype(str) + '-' + df_xdxr['day'].astype(str))
    df_xdxr.set_index('date', inplace=True)
    df_xdxr.sort_index(ascending=False, inplace=True)
    df_kline['adj_factor'] = 1.0
    for date, row in df_xdxr.iterrows():
        songzhuan = row.get('songzhuangu', 0) or 0
        fenhong = row.get('fenhong', 0) or 0
        peigu = row.get('peigu', 0) or 0
        peigujia = row.get('peigujia', 0) or 0
        mask = df_kline.index < date
        denominator = 1 + (songzhuan / 10.0) + (peigu / 10.0)
        df_kline.loc[mask, 'adj_factor'] = df_kline.loc[mask, 'adj_factor'] / denominator
    for col in ['开盘', '收盘', '最高', '最低']:
        df_kline[col] = df_kline[col] * df_kline['adj_factor']
    return df_kline

# ==========================================
# 3. 单只股票处理引擎
# ==========================================
def process_stock(stock_info, start_date, end_date, client):
    symbol = stock_info['code']
    try:
        # 获取最近 800 个交易日
        df = client.bars(symbol=symbol, frequency=9, offset=800)
        
        # 【修复】放宽到 60 天，兼容创业板次新股
        if df is None or df.empty or len(df) < 60:
            return None
            
        df.rename(columns={'datetime': '日期', 'open': '开盘', 'high': '最高', 'low': '最低', 'close': '收盘', 'vol': '成交量'}, inplace=True)
        df['日期'] = pd.to_datetime(df['日期']).dt.normalize()
        df.set_index('日期', inplace=True)
        df.sort_index(inplace=True)

        for c in ['开盘', '收盘', '最高', '最低', '成交量']:
            df[c] = pd.to_numeric(df[c], errors='coerce')
            
        df_xdxr = client.xdxr(symbol=symbol)
        df = adjust_qfq_for_tdx(df, df_xdxr)

        df['MA20'] = df['收盘'].rolling(20, min_periods=1).mean() 
        df['MA60'] = df['收盘'].rolling(60, min_periods=1).mean()
        
        ma3 = df['收盘'].rolling(3, min_periods=1).mean()
        ma6 = df['收盘'].rolling(6, min_periods=1).mean()
        ma12 = df['收盘'].rolling(12, min_periods=1).mean()
        ma24 = df['收盘'].rolling(24, min_periods=1).mean()
        df['BBI'] = (ma3 + ma6 + ma12 + ma24) / 4
        
        df['Log_Ret'] = np.log(df['收盘'] / df['收盘'].shift(1))
        df['波动率%'] = df['Log_Ret'].rolling(20, min_periods=1).std() * np.sqrt(252) * 100

        # --- 策略1：历史大底 ---
        for p in [500, 250, 90]:
            # 【修复】加入 min_periods=1，兼容不足 500 天的股票
            df[f'HHV{p}'] = df['最高'].rolling(p, min_periods=1).max()
            df[f'LLV{p}'] = df['最低'].rolling(p, min_periods=1).min()
            df[f'R_HHV{p}'] = df[f'HHV{p}'].rolling(21, min_periods=1).mean()
            df[f'R_LLV{p}'] = df[f'LLV{p}'].rolling(21, min_periods=1).mean()
            
        df['R7'] = (df['R_LLV500']*0.96 + df['R_LLV250']*0.96 + df['R_LLV90']*0.96 + 
                    df['R_HHV500']*0.558 + df['R_HHV250']*0.558 + df['R_HHV90']*0.558) / 6
        df['R8'] = (df['R_LLV500']*1.25 + df['R_LLV250']*1.23 + df['R_LLV90']*1.2 + 
                    df['R_HHV500']*0.55 + df['R_HHV250']*0.55 + df['R_HHV90']*0.65) / 6
        df['R9'] = (df['R_LLV500']*1.3 + df['R_LLV250']*1.3 + df['R_LLV90']*1.3 + 
                    df['R_HHV500']*0.68 + df['R_HHV250']*0.68 + df['R_HHV90']*0.68) / 6
        
        df['RA'] = (df['R7']*3 + df['R8']*2 + df['R9']) / 6 * 1.738
        df['RA'] = df['RA'].rolling(21, min_periods=1).mean()
        
        df['RB'] = df['最低'].shift(1)
        df['ABS_LOW_RB'] = (df['最低'] - df['RB']).abs()
        df['MAX_LOW_RB'] = (df['最低'] - df['RB']).clip(lower=0)
        df['SMA_ABS'] = sma(df['ABS_LOW_RB'], 3, 1)
        df['SMA_MAX'] = sma(df['MAX_LOW_RB'], 3, 1)
        
        with np.errstate(divide='ignore', invalid='ignore'):
            df['RC'] = np.where(df['SMA_MAX'] != 0, (df['SMA_ABS'] / df['SMA_MAX']) * 100, 0)
        
        df['RD'] = np.where(df['收盘']*1.35 <= df['RA'], df['RC']*10, df['RC']/10)
        df['RD'] = df['RD'].rolling(3, min_periods=1).mean()
        df['RE'] = df['最低'].rolling(30, min_periods=1).min()
        df['RF'] = df['RD'].rolling(30, min_periods=1).max()
        df['R10'] = df['收盘'].rolling(58, min_periods=1).mean().notna().astype(int)
        
        # 【修复】大底建仓最终公式对齐通达信
        # 【修改后：严格对齐通达信的红柱 COLORSTICK】
        raw_signal = np.where(df['最低'] <= df['RE'], (df['RD'] + df['RF']*2)/2, 0)
        df['S1_Raw_Val'] = pd.Series(raw_signal, index=df.index).rolling(3, min_periods=1).mean() / 618 * df['R10']
        
        # 只要公式计算结果大于 0，就对应通达信里画出红色柱子，绝不擅自延长
        df['策略1_大底信号'] = np.where(df['S1_Raw_Val'] > 0, 'Y', '')

        # --- 策略2：波段回调 (做T) ---
        # 【修改后：引入通达信的 CROSS 逻辑】
        df['VAR1'] = (df['收盘'] + df['最高'] + df['开盘'] + df['最低']) / 4
        df['S2_BuyLine'] = calculate_xma(df['VAR1'], 32) * (1 - 4/100)
        df['策略2_波段信号'] = np.where(
            (df['S2_BuyLine'].notna()) & (df['最低'] <= df['S2_BuyLine']), 
            'Y', ''
        )


        # --- 策略3：主升浪 ---
        df['MA20_Slope'] = (df['MA20'] / df['MA20'].shift(1) - 1) * 100
        df['MA20_Angle'] = np.degrees(np.arctan(df['MA20_Slope']))
        exp12 = df['收盘'].ewm(span=12, adjust=False).mean()
        exp26 = df['收盘'].ewm(span=26, adjust=False).mean()
        df['DIF'] = exp12 - exp26
        df['DEA'] = df['DIF'].ewm(span=9, adjust=False).mean()
        
        df['VOL_MA5'] = df['成交量'].rolling(5, min_periods=1).mean()
        df['MA5'] = df['收盘'].rolling(5, min_periods=1).mean()
        df['MA10'] = df['收盘'].rolling(10, min_periods=1).mean()

        cond_angle = df['MA20_Angle'] > 25
        cond_trend = (df['收盘'] > df['MA10']) & (df['MA5'] > df['MA20']) & (df['MA20'] > df['MA60']) & (df['MA60'] > df['MA60'].shift(1))
        cond_power = (df['收盘'] / df['收盘'].shift(1) > 1.03) & (df['收盘'] > df['开盘'])
        cond_vol = df['成交量'] > df['VOL_MA5']
        cond_macd = (df['DIF'] > 0) & (df['DIF'] > df['DEA'])

        df['策略3_主升浪'] = np.where(cond_angle & cond_trend & cond_power & cond_vol & cond_macd, '🔥', '')

        # --- 连续信号标记逻辑 ---
        check_list = {'策略1_大底信号': 'Y', '策略2_波段信号': 'Y', '策略3_主升浪': '🔥'}
        for col, marker in check_list.items():
            condition = df[col] == marker
            groups = (condition != condition.shift()).cumsum()
            df['temp_count'] = df.groupby(groups).cumcount() + 1
            mask = condition & (df['temp_count'] > 1)
            df.loc[mask, col] = marker + ' x' + df['temp_count'].astype(str)

        if 'temp_count' in df.columns:
            del df['temp_count']
            
        latest_date = df.index[-1]
        row = df.iloc[-1]

        output_list = [{
            '股票代码': stock_info['code'],
            '股票简称': stock_info['name'],
            '主营行业': stock_info['industry'],
            '地区': stock_info['area'],
            '类型': stock_info['type'],
            '日期': latest_date.strftime('%Y-%m-%d'),
            '收盘价': round(row['收盘'], 2),
            '策略1': row['策略1_大底信号'], 
            '策略2': row['策略2_波段信号'],
            '策略3': row['策略3_主升浪'], 
            'BBI': round(row['BBI'], 2),
            'MA60': round(row['MA60'], 2),
            '波动率': round(row['波动率%'], 2)
        }]
        # 增加极其短暂的休眠，防止连接太快被通达信封锁
        time.sleep(0.02)
        return output_list

    except Exception as e:
        # 如果还有失败的，它会明确在终端里打印出来原因！
        print(f"[{symbol}] 失败原因: {str(e)}") 
        return None

# ==========================================
# 4. HTML 生成器
# ==========================================
def generate_html_report(df, filename, date_str):
    target_columns = [
        '股票代码', '股票简称', '主营行业', '地区', '类型', '日期', '收盘价',
        '策略1', '策略2', '策略3', 'BBI', 'MA60', '波动率'
    ]
    existing_cols = [c for c in target_columns if c in df.columns]
    df = df[existing_cols]
    df = df.fillna('-')

    s1_count = df[df['策略1'].str.contains('Y', na=False)].shape[0]
    s2_count = df[df['策略2'].str.contains('Y', na=False)].shape[0]
    s3_count = df[df['策略3'].str.contains('🔥', na=False)].shape[0] 
    total_count = df.shape[0]
    
    fig_summary = px.bar(
        x=['策略1(左侧大底)', '策略2(左侧波段)', '策略3(右侧主升)'], 
        y=[s1_count, s2_count, s3_count], 
        title=f"今日信号触发数量 (总扫描: {total_count}只)",
        labels={'x':'策略类型', 'y':'触发数量'},
        color=['策略1(左侧大底)', '策略2(左侧波段)', '策略3(右侧主升)'],
        text=[s1_count, s2_count, s3_count]
    )
    fig_summary.update_layout(height=400)
    summary_chart_html = pio.to_html(fig_summary, full_html=False, include_plotlyjs='cdn')

    trigger_df_1 = df[df['策略1'].str.contains('Y', na=False)]
    if not trigger_df_1.empty:
        industry_counts_1 = trigger_df_1['主营行业'].value_counts().reset_index()
        industry_counts_1.columns = ['主营行业', '数量']
        fig_ind_1 = px.pie(industry_counts_1, values='数量', names='主营行业', title='策略1(大底) 行业分布')
        fig_ind_1.update_layout(height=400)
        ind_chart_html_1 = pio.to_html(fig_ind_1, full_html=False, include_plotlyjs=False)
    else:
        ind_chart_html_1 = "<p class='text-center mt-5'>策略1今日无信号</p>"

    trigger_df_3 = df[df['策略3'].str.contains('🔥', na=False)]
    if not trigger_df_3.empty:
        industry_counts_3 = trigger_df_3['主营行业'].value_counts().reset_index()
        industry_counts_3.columns = ['主营行业', '数量']
        fig_ind_3 = px.pie(industry_counts_3, values='数量', names='主营行业', title='🔥策略3(主升浪) 行业分布')
        fig_ind_3.update_layout(height=400)
        ind_chart_html_3 = pio.to_html(fig_ind_3, full_html=False, include_plotlyjs=False)
    else:
        ind_chart_html_3 = "<p class='text-center mt-5'>策略3今日无信号</p>"

    table_html = df.to_html(classes='display table table-striped table-bordered', index=False, table_id='stockTable', border=0)

    html_template = f"""
    <!DOCTYPE html>
    <html lang="zh-CN">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>量化选股日报 - {date_str}</title>
        <link href="https://cdn.bootcdn.net/ajax/libs/twitter-bootstrap/5.3.0/css/bootstrap.min.css" rel="stylesheet">
        <link href="https://cdn.datatables.net/1.13.6/css/dataTables.bootstrap5.min.css" rel="stylesheet">
        <style>
            body {{ font-family: "Microsoft YaHei", sans-serif; background-color: #f8f9fa; padding: 20px; }}
            .card {{ margin-bottom: 20px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); border: none; }}
            .card-header {{ background-color: #343a40; color: white; font-weight: bold; }}
            .strategy-box {{ background-color: #e9ecef; padding: 15px; border-radius: 5px; margin-bottom: 20px; border-left: 5px solid #0d6efd; }}
            .strategy-title {{ font-weight: bold; color: #0d6efd; }}
        </style>
    </head>
    <body>
    <div class="container-fluid">
        <h1 class="text-center mb-4">📈 量化交易信号日报 (极速实时版) - {date_str}</h1>
        
        <div class="card">
            <div class="card-header">策略看板</div>
            <div class="card-body">
                <div class="row">
                    <div class="col-md-4">{summary_chart_html}</div>
                    <div class="col-md-4">{ind_chart_html_1}</div>
                    <div class="col-md-4">{ind_chart_html_3}</div>
                </div>
            </div>
        </div>

        <div class="card">
            <div class="card-header">全市场扫描结果 (支持搜索与排序)</div>
            <div class="card-body">
                <div class="table-responsive">{table_html}</div>
            </div>
        </div>
    </div>

    <script src="https://cdn.bootcdn.net/ajax/libs/jquery/3.7.1/jquery.min.js"></script>
    <script src="https://cdn.datatables.net/1.13.6/js/jquery.dataTables.min.js"></script>
    <script src="https://cdn.datatables.net/1.13.6/js/dataTables.bootstrap5.min.js"></script>

    <script>
        $(document).ready(function() {{
            $('#stockTable').DataTable({{
                "order": [[ 9, "desc" ], [ 8, "desc" ], [ 7, "desc" ]],
                "pageLength": 25,
                "language": {{
                    "search": "🔍 搜索:",
                    "lengthMenu": "每页显示 _MENU_ 条",
                    "info": "第 _START_ 至 _END_ 条 / 共 _TOTAL_ 条",
                    "paginate": {{ "next": "下页", "previous": "上页" }}
                }},
                "rowCallback": function( row, data, index ) {{
                    var s1 = data[7]; 
                    var s2 = data[8];
                    var s3 = data[9];
                    
                    if (s1.includes('Y')) {{
                        $('td:eq(7)', row).html('<span class="badge bg-danger">' + s1 + '</span>');
                        $(row).addClass('table-warning');
                    }}
                    if (s2.includes('Y')) {{
                        $('td:eq(8)', row).html('<span class="badge bg-success">' + s2 + '</span>');
                        $(row).addClass('table-warning');
                    }}
                    if (s3.includes('🔥')) {{
                        var badgeClass = s3.length > 2 ? 'badge bg-danger' : 'badge bg-warning text-dark';
                        $('td:eq(9)', row).html('<span class="' + badgeClass + '">' + s3 + '</span>');
                        $(row).addClass('table-warning');
                    }}     
                }}
            }});
        }});
    </script>
    </body>
    </html>
    """
    
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(html_template)
    
    return filename

# ==========================================
# 5. 主程序入口
# ==========================================
if __name__ == '__main__':
    input_file = 'stock_list.xlsx'
    output_html = 'index.html'    
    today_str = datetime.date.today().strftime('%Y-%m-%d')
    start_date = today_str     
    end_date = today_str
    print(f"自动设定分析日期为: {today_str}")

    if not os.path.exists(input_file):
        print(f"错误：找不到文件 {input_file}")
        exit()

    print(f"正在读取 {input_file} ...")
    try:
        meta_df = pd.read_excel(input_file, usecols=[0, 1, 2, 3, 4])
        meta_df.columns = ['code', 'name', 'industry', 'area', 'type']
        meta_df.dropna(subset=['code'], inplace=True)
        meta_df = meta_df[meta_df['code'].astype(str).str.strip() != '']
        meta_df['code'] = meta_df['code'].astype(str).str.replace(r'\.0$', '', regex=True).str.zfill(6)
        
        stock_list = meta_df.to_dict('records')
        print(f"成功加载 {len(stock_list)} 只有效股票信息。")
    except Exception as e:
        print(f"读取 Excel 文件失败: {e}")
        exit()

    print(f"📡 正在连接通达信主推服务器 (极速版)...")
    client = Quotes.factory(market='std', multithread=True, heartbeat=True)

    print(f"开始计算，时间范围: {start_date} 至 {end_date} ...")
    all_results = []
    
    # 【修复】必须锁定单线程！否则通达信服务器会断开您的连接
    with ThreadPoolExecutor(max_workers=1) as executor:
        futures = {executor.submit(process_stock, stock, start_date, end_date, client): stock['code'] for stock in stock_list}
        count = 0
        total = len(futures)
        for future in as_completed(futures):
            try:
                data = future.result()
                if data:
                    all_results.extend(data)
            except Exception:
                pass
            count += 1
            if count % 10 == 0 or count == total:
                print(f"进度: {count}/{total}   ", end='\r')

    if all_results:
        final_df = pd.DataFrame(all_results)
        cols_order = ['股票代码', '股票简称', '主营行业', '地区', '类型', '日期', '收盘价', '策略1', '策略2','策略3', 'BBI', 'MA60', '波动率']
        cols_order = [c for c in cols_order if c in final_df.columns]
        final_df = final_df[cols_order]
        final_df.sort_values(by=['日期', '股票代码'], inplace=True)
        
        print(f"\n正在生成 HTML 交互式报告...")
        generate_html_report(final_df, output_html, start_date)
        
        print(f"\n========================================")
        print(f"成功！扫描出 {len(final_df)} 只股票，请在浏览器打开: {output_html}")
        print(f"========================================")
        os._exit(0)
    else:
        print("\n未生成任何有效结果。")
        os._exit(0)