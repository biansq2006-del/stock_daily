import time
import random
import akshare as ak
import pandas as pd
import numpy as np
import datetime
import os
import plotly.express as px
import plotly.io as pio
import plotly.express as px
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

# ==========================================
# 1. 核心算法函数定义 (保持不变)
# ==========================================

def sma(series, n, m):
    """
    策略1核心：通达信SMA递归算法
    """
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
    """策略2核心：EMA算法 (XMA)"""
    return series.ewm(span=window, adjust=False).mean()

# ==========================================
# 2. 单只股票处理引擎 (保持 qfq 修正)
# ==========================================
def process_stock(stock_info, start_date, end_date):
    symbol = stock_info['code']
    try:
        # 往前推3年获取数据用于计算长期均线
        fetch_start = (pd.to_datetime(start_date) - datetime.timedelta(days=1000)).strftime('%Y%m%d')
        fetch_end = pd.to_datetime(end_date).strftime('%Y%m%d')
        
        # 使用 qfq (前复权)
        df = ak.stock_zh_a_hist(symbol=symbol, period="daily", start_date=fetch_start, end_date=fetch_end, adjust="qfq")
        
        if df.empty or len(df) < 500:
            return None
            
        df['日期'] = pd.to_datetime(df['日期'])
        df.set_index('日期', inplace=True)
        df.sort_index(inplace=True)
        
        # 【注意】这里增加了 '成交量' 的数值转换，防止报错
        for c in ['开盘', '收盘', '最高', '最低', '成交量']:
            df[c] = pd.to_numeric(df[c], errors='coerce')

        # --- 基础指标 ---
        df['MA20'] = df['收盘'].rolling(20).mean() # 新增 MA20
        df['MA60'] = df['收盘'].rolling(60).mean()
        
        # BBI
        ma3 = df['收盘'].rolling(3).mean()
        ma6 = df['收盘'].rolling(6).mean()
        ma12 = df['收盘'].rolling(12).mean()
        ma24 = df['收盘'].rolling(24).mean()
        df['BBI'] = (ma3 + ma6 + ma12 + ma24) / 4
        
        # 波动率
        df['Log_Ret'] = np.log(df['收盘'] / df['收盘'].shift(1))
        df['波动率%'] = df['Log_Ret'].rolling(20).std() * np.sqrt(252) * 100

        # ==========================================
        # 策略1：历史大底 (Deep Bottom) - 保持不变
        # ==========================================
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
        
        df['RA'] = (df['R7']*3 + df['R8']*2 + df['R9']) / 6 * 1.738
        df['RA'] = df['RA'].rolling(21).mean()
        
        df['RB'] = df['最低'].shift(1)
        df['ABS_LOW_RB'] = (df['最低'] - df['RB']).abs()
        df['MAX_LOW_RB'] = (df['最低'] - df['RB']).clip(lower=0)
        df['SMA_ABS'] = sma(df['ABS_LOW_RB'], 3, 1)
        df['SMA_MAX'] = sma(df['MAX_LOW_RB'], 3, 1)
        
        with np.errstate(divide='ignore', invalid='ignore'):
            df['RC'] = np.where(df['SMA_MAX'] != 0, (df['SMA_ABS'] / df['SMA_MAX']) * 100, 0)
        
        df['RD'] = np.where(df['收盘']*1.35 <= df['RA'], df['RC']*10, df['RC']/10)
        df['RD'] = df['RD'].rolling(3).mean()
        df['RE'] = df['最低'].rolling(30).min()
        df['RF'] = df['RD'].rolling(30).max()
        
        df['R10'] = df['收盘'].rolling(58).mean().notna().astype(int)
        raw_signal = np.where(df['最低'] <= df['RE'], (df['RD'] + df['RF']*2)/2, 0)
        df['S1_Raw_Val'] = raw_signal * df['R10']
        df['S1_Trigger'] = (df['S1_Raw_Val'] > 0).astype(int)
        df['S1_Final_Flag'] = df['S1_Trigger'].rolling(window=3, min_periods=1).max()
        df['策略1_大底信号'] = np.where(df['S1_Final_Flag'] > 0, 'Y', '')

        # ==========================================
        # 策略2：波段回调 (EMA Pullback) - 保持不变
        # ==========================================
        df['VAR1'] = (df['收盘'] + df['最高'] + df['开盘'] + df['最低']) / 4
        df['S2_BuyLine'] = calculate_xma(df['VAR1'], 32) * (1 - 4/100)
        df['策略2_波段信号'] = np.where(df['收盘'] < df['S2_BuyLine'], 'Y', '')

        # ==========================================
        # 策略3：右侧强趋势 (RIGHT_SIDE_PRO) - 【新增】
        # ==========================================
        # 1. 计算 MA20 角度 (斜率)
        # 用 atan 计算弧度，再转角度。为了让斜率更明显，通常会 * 100
        df['MA20_Slope'] = (df['MA20'] / df['MA20'].shift(1) - 1) * 100
        df['MA20_Angle'] = np.degrees(np.arctan(df['MA20_Slope']))
        
        # 2. 计算 MACD
        exp12 = df['收盘'].ewm(span=12, adjust=False).mean()
        exp26 = df['收盘'].ewm(span=26, adjust=False).mean()
        df['DIF'] = exp12 - exp26
        df['DEA'] = df['DIF'].ewm(span=9, adjust=False).mean()
        
        # 3. 计算成交量均线
        df['VOL_MA5'] = df['成交量'].rolling(5).mean()
        df['MA5'] = df['收盘'].rolling(5).mean()
        df['MA10'] = df['收盘'].rolling(10).mean()

        # 4. 组合条件
        # 条件A: 角度 > 25度 (强趋势)
        cond_angle = df['MA20_Angle'] > 25
        # 条件B: 多头排列 (股价 > 20日线 > 60日线，且60日线向上)
        cond_trend = (df['收盘'] > df['MA10']) & \
                     (df['MA5'] > df['MA20']) & \
                     (df['MA20'] > df['MA60']) & \
                     (df['MA60'] > df['MA60'].shift(1))
        # 条件C: 动能 (涨幅 > 3% 且 阳线)
        cond_power = (df['收盘'] / df['收盘'].shift(1) > 1.03) & (df['收盘'] > df['开盘'])
        # 条件D: 放量
        cond_vol = df['成交量'] > df['VOL_MA5']
        # 条件E: MACD 水上金叉或多头
        cond_macd = (df['DIF'] > 0) & (df['DIF'] > df['DEA'])

        # 最终信号：用 '🔥' 标识
        df['策略3_主升浪'] = np.where(cond_angle & cond_trend & cond_power & cond_vol & cond_macd, '🔥', '')
        # 【新增】连续信号标记逻辑 (Streak Counter)
        # ==========================================
        # 定义需要统计连续天数的列和对应的原始标记
        check_list = {
            '策略1_大底信号': 'Y',
            '策略2_波段信号': 'Y',
            '策略3_主升浪': '🔥'
        }

        for col, marker in check_list.items():
            # 1. 找出满足条件的行 (True/False)
            condition = df[col] == marker
            
            # 2. 利用 Pandas 分组计算连续出现的次数
            # (condition != condition.shift()) 用于判断状态是否切换
            # .cumsum() 给每一段连续的状态分配一个唯一的组ID
            groups = (condition != condition.shift()).cumsum()
            
            # .cumcount() + 1 计算组内的累积计数
            df['temp_count'] = df.groupby(groups).cumcount() + 1
            
            # 3. 只修改连续出现天数 > 1 的行
            # 逻辑：如果是第2天及以上，且当前确实有信号，就改为 "标记 xN"
            mask = condition & (df['temp_count'] > 1)
            df.loc[mask, col] = marker + ' x' + df['temp_count'].astype(str)

        # 清理临时列
        if 'temp_count' in df.columns:
            del df['temp_count']
        # --- 截取结果 ---
        result_df = df[start_date:end_date].copy()
        if result_df.empty:
            return None

        output_list = []
        for date, row in result_df.iterrows():
            output_list.append({
                '股票代码': stock_info['code'],
                '股票简称': stock_info['name'],
                '主营行业': stock_info['industry'],
                '地区': stock_info['area'],
                '类型': stock_info['type'],
                '日期': date.strftime('%Y-%m-%d'),
                '收盘价': round(row['收盘'], 2),
                '策略1': row['策略1_大底信号'], 
                '策略2': row['策略2_波段信号'],
                '策略3': row['策略3_主升浪'], # 新增这一列
                'BBI': round(row['BBI'], 2),
                'MA60': round(row['MA60'], 2),
                '波动率': round(row['波动率%'], 2)
            })
            
        return output_list

    except Exception as e:
        # print(f"处理 {symbol} 时发生错误: {str(e)}") 
        return None

# ==========================================
# 3. HTML 生成器
# ==========================================

def generate_html_report(df, filename, date_str):
    """
    生成包含 DataTables 和 Plotly 图表的静态 HTML
    """
    
    # --- 0. [关键步骤] 强制指定列顺序 ---
    # 这一步是为了确保 HTML 表格中的列顺序固定
    # 对应 JS 中的索引：0-代码, 6-收盘, 7-策略1, 8-策略2, 9-策略3, 10-BBI, 11-MA60, 12-波动率
    target_columns = [
        '股票代码', '股票简称', '主营行业', '地区', '类型', '日期', '收盘价',
        '策略1', '策略2', '策略3', 
        'BBI', 'MA60', '波动率'
    ]
    
    # 确保只取存在的列，防止报错
    existing_cols = [c for c in target_columns if c in df.columns]
    df = df[existing_cols]
    
    # 填充空值（例如 MA60 刚开始计算可能为空），避免页面显示 NaN
    df = df.fillna('-')

    # --- 1. 生成统计图表 (Plotly) ---
    
    # 统计 1: 三个策略的信号总数
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

    # 统计 2: 触发“大底”的行业分布 (策略1)
    trigger_df_1 = df[df['策略1'].str.contains('Y', na=False)]
    if not trigger_df_1.empty:
        industry_counts_1 = trigger_df_1['主营行业'].value_counts().reset_index()
        industry_counts_1.columns = ['主营行业', '数量']
        fig_ind_1 = px.pie(industry_counts_1, values='数量', names='主营行业', title='策略1(大底) 行业分布')
        fig_ind_1.update_layout(height=400)
        ind_chart_html_1 = pio.to_html(fig_ind_1, full_html=False, include_plotlyjs=False)
    else:
        ind_chart_html_1 = "<p class='text-center mt-5'>策略1今日无信号</p>"

    # 统计 3: 触发“主升浪”的行业分布 (策略3)
    trigger_df_3 = df[df['策略3'].str.contains('🔥', na=False)]
    if not trigger_df_3.empty:
        industry_counts_3 = trigger_df_3['主营行业'].value_counts().reset_index()
        industry_counts_3.columns = ['主营行业', '数量']
        fig_ind_3 = px.pie(industry_counts_3, values='数量', names='主营行业', title='🔥策略3(主升浪) 行业分布')
        fig_ind_3.update_layout(height=400)
        ind_chart_html_3 = pio.to_html(fig_ind_3, full_html=False, include_plotlyjs=False)
    else:
        ind_chart_html_3 = "<p class='text-center mt-5'>策略3今日无信号</p>"

    # --- 2. 构建 HTML 模板 ---
    
    # 转换表格 (现在 df 已经包含了 BBI, MA60, 波动率)
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
            .highlight-red {{ color: white; background-color: #dc3545 !important; font-weight: bold; }}
            .highlight-green {{ color: white; background-color: #28a745 !important; font-weight: bold; }}
            .highlight-fire {{ color: white; background-color: #fd7e14 !important; font-weight: bold; }} 
            .strategy-box {{ background-color: #e9ecef; padding: 15px; border-radius: 5px; margin-bottom: 20px; border-left: 5px solid #0d6efd; }}
            .strategy-title {{ font-weight: bold; color: #0d6efd; }}
        </style>
    </head>
    <body>

    <div class="container-fluid">
        <h1 class="text-center mb-4">📈 量化交易信号日报 ({date_str})</h1>
        
        <div class="card">
            <div class="card-header">策略说明书 (左侧抄底 + 右侧强攻)</div>
            <div class="card-body">
                <div class="row">
                    <div class="col-md-4">
                        <div class="strategy-box" style="border-left-color: #dc3545;">
                            <div class="strategy-title" style="color: #dc3545;">策略1：大底 (Deep Bottom，左侧)</div>
                            <p><strong>信号：</strong> <span class="badge bg-danger">Y</span></p>
                            <p><strong>逻辑：</strong> 股票处于极度深跌状态，且近期（近3日）刚刚创下30日新低。</p>
                            <p><strong>建议：</strong> 这是一个左侧长线建仓信号。长线胜率较高，但需忍受短期波动。</p>
                        </div>
                    </div>
                    <div class="col-md-4">
                        <div class="strategy-box" style="border-left-color: #28a745;">
                            <div class="strategy-title" style="color: #28a745;">策略2：波段(EMA Pullback,左侧)</div>
                            <p><strong>信号：</strong> <span class="badge bg-success">Y</span></p>
                            <p><strong>逻辑：</strong> 股票短期向下调整，股价跌穿了 EMA 通道（类似BOLL）的下轨道。</p>
                            <p><strong>建议：</strong> 如果股价和通道拟合度高，这是一个短线/波段反弹信号，适合快进快出。</p>
                        </div>
                    </div>
                    <div class="col-md-4">
                        <div class="strategy-box" style="border-left-color: #fd7e14;">
                            <div class="strategy-title" style="color: #fd7e14;">策略3：主升浪 (Main Wave Pro,右侧)</div>
                            <p><strong>信号：</strong> <span class="badge bg-warning text-dark">🔥</span></p>
                            <p><strong>逻辑：</strong> 20日线大角度上扬(>25度) + 多头排列 + 放量大阳。</p>
                            <p><strong>建议：</strong> <strong>确定性较高</strong>。趋势确认，适合追涨。结束：5/10日均线跌破。</p>
                        </div>
                    </div>
                </div>
                
                <div class="row mt-2">
                    <div class="col-12">
                        <div class="strategy-box" style="border-left-color: #6c757d; background-color: #f1f3f5;">
                            <div class="strategy-title" style="color: #6c757d;">🛠️ 辅助指标参考</div>
                            <div class="row">
                                <div class="col-md-6">
                                    <p><strong>🔵 BBI & MA60：</strong> 多空分界线。如果股价在 BBI和MA60 下方很远，属于逆势抄底（左侧）；如果站上 BBI，说明短期趋势转强（右侧）。可结合BBI上穿和下插MA60线进行判断。</p>
                                </div>
                                <div class="col-md-6">
                                    <p><strong>⚡ 波动率(%)：</strong> 蓝筹股通常在 20-30%，妖股 >50%。<strong>注意：</strong>如果大底信号出现时波动率极高(>60%)，说明市场极度恐慌，风险较大，仓位要轻。</p>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <div class="row">
            <div class="col-md-4">
                <div class="card">
                    <div class="card-body">
                        {summary_chart_html}
                    </div>
                </div>
            </div>
            <div class="col-md-4">
                <div class="card">
                    <div class="card-body">
                        {ind_chart_html_1}
                    </div>
                </div>
            </div>
            <div class="col-md-4">
                <div class="card">
                    <div class="card-body">
                        {ind_chart_html_3}
                    </div>
                </div>
            </div>
        </div>

        <div class="card">
            <div class="card-header">
                全市场扫描结果 (支持搜索与排序)
            </div>
            <div class="card-body">
                <div class="table-responsive">
                    {table_html}
                </div>
            </div>
        </div>
        
        <footer class="text-center text-muted mt-4">
            <small>AI Application Engineer:Saviour | Data Source: AkShare | Strategy: Left & Right Side Combo</small>
        </footer>

    </div>

    <script src="https://cdn.bootcdn.net/ajax/libs/jquery/3.7.1/jquery.min.js"></script>
    <script src="https://cdn.datatables.net/1.13.6/js/jquery.dataTables.min.js"></script>
    <script src="https://cdn.datatables.net/1.13.6/js/dataTables.bootstrap5.min.js"></script>

    <script>
        $(document).ready(function() {{
            // 初始化 DataTable
            var table = $('#stockTable').DataTable({{
                "order": [[ 9, "desc" ], [ 8, "desc" ], [ 7, "desc" ]], // 默认优先按 策略3(第10列，索引9) 排序
                "pageLength": 25,
                "language": {{
                    "search": "🔍 搜索股票/行业:",
                    "lengthMenu": "每页显示 _MENU_ 条",
                    "info": "显示第 _START_ 至 _END_ 条，共 _TOTAL_ 条",
                    "paginate": {{
                        "first": "首页",
                        "last": "末页",
                        "next": "下一页",
                        "previous": "上一页"
                    }}
                }},
                // 回调函数：用于高亮显示信号行
                "rowCallback": function( row, data, index ) {{
                    // 列索引对照（已在 Python 中强制对齐）：
                    // 0:代码, 1:简称, 2:行业, 3:地区, 4:类型, 5:日期, 6:收盘
                    // 7:策略1(大底), 8:策略2(波段), 9:策略3(主升)
                    // 10:BBI, 11:MA60, 12:波动率
                    
                    var s1 = data[7]; 
                    var s2 = data[8];
                    var s3 = data[9];
                    
                    // 修改判断逻辑：只要包含 'Y' 或 '🔥' 就算命中
                    
                    if (s1.includes('Y')) {{
                        // 这里可以把显示的文字直接设为 s1 (这样网页上就会显示 "Y x3")
                        $('td:eq(7)', row).html('<span class="badge bg-danger">' + s1 + '</span>');
                        $(row).addClass('table-warning');
                    }}
                    
                    if (s2.includes('Y')) {{
                        $('td:eq(8)', row).html('<span class="badge bg-success">' + s2 + '</span>');
                        $(row).addClass('table-warning');
                    }}

                    if (s3.includes('🔥')) {{
                        // 如果是 "🔥 x3"，s3 变量里本身就已经有了，直接显示即可
                        // 可以加个判断，如果是 x2, x3... 换个颜色更深的 badge
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
# 4. 主程序入口
# ==========================================
if __name__ == '__main__':
    # ================= 配置区 =================
    input_file = 'stock_list.xlsx'
    output_html = 'index.html'    # Gitee Pages 默认入口通常是 index.html
    today_str = today_str = datetime.date.today().strftime('%Y-%m-%d')
    start_date = today_str     # 你的日期
    end_date = today_str
    print(f"自动设定分析日期为: {today_str}")
    # =========================================

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

    print(f"开始计算，时间范围: {start_date} 至 {end_date} ...")
    all_results = []
    
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {executor.submit(process_stock, stock, start_date, end_date): stock['code'] for stock in stock_list}
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
                print(f"进度: {count}/{total}", end='\r')

    if all_results:
        final_df = pd.DataFrame(all_results)
        cols_order = ['股票代码', '股票简称', '主营行业', '地区', '类型', '日期', '收盘价', '策略1', '策略2','策略3', 'BBI', 'MA60', '波动率']
        cols_order = [c for c in cols_order if c in final_df.columns]
        final_df = final_df[cols_order]
        final_df.sort_values(by=['日期', '股票代码'], inplace=True)
        
        # === 生成 HTML 报告 ===
        print(f"\n正在生成 HTML 交互式报告...")
        generate_html_report(final_df, output_html, start_date)
        
        print(f"\n========================================")
        print(f"成功！请在浏览器打开: {output_html}")
        print(f"包含交互式表格和策略说明，可直接部署到 Gitee Pages。")
        print(f"========================================")
    else:
        print("\n未生成任何有效结果。")