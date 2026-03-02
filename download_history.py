import os
import pandas as pd
import baostock as bs
import datetime

# --- 配置 ---
INPUT_FILE = 'stock_list.xlsx'
SAVE_DIR = './history_data'  # 数据保存在当前目录的 history_data 文件夹下
YEARS_TO_FETCH = 5           # 下载过去5年的数据

if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

def convert_to_baostock_code(code):
    """将 6 位纯数字代码转换为 Baostock 需要的 sh. / sz. 格式"""
    code_str = str(code).zfill(6)
    if code_str.startswith('6'):
        return f"sh.{code_str}"
    elif code_str.startswith('0') or code_str.startswith('3'):
        return f"sz.{code_str}"
    elif code_str.startswith('4') or code_str.startswith('8'):
        return f"bj.{code_str}"
    return f"sh.{code_str}" # 默认回退

if __name__ == '__main__':
    print("=== 开始使用 Baostock 下载历史前复权数据 (单线程稳定版) ===")
    end_date = datetime.date.today()
    start_date = end_date - datetime.timedelta(days=365 * YEARS_TO_FETCH)
    
    # Baostock 的日期格式要求带横杠 YYYY-MM-DD
    start_str = start_date.strftime('%Y-%m-%d')
    end_str = end_date.strftime('%Y-%m-%d')
    
    # 1. 读取股票列表
    meta_df = pd.read_excel(INPUT_FILE, usecols=[0])
    raw_codes = meta_df.iloc[:, 0].dropna().astype(str).str.replace(r'\.0$', '', regex=True).str.zfill(6).tolist()
    
    print(f"共发现 {len(raw_codes)} 只股票，准备下载 {start_str} 至 {end_str} 的数据...")
    
    # 2. 登录 Baostock 系统
    lg = bs.login()
    if lg.error_code != '0':
        print(f"Baostock 登录失败: {lg.error_msg}")
        exit()
        
    success_count = 0
    
    # 3. 稳妥的单线程循环下载（绝不报 Connection aborted）
    for i, raw_code in enumerate(raw_codes, 1):
        bs_code = convert_to_baostock_code(raw_code)
        
        # adjustflag="2" 代表前复权 (极其重要)
        rs = bs.query_history_k_data_plus(
            bs_code,
            "date,open,close,high,low,volume",
            start_date=start_str, 
            end_date=end_str,
            frequency="d", 
            adjustflag="2"
        )
        
        if rs.error_code == '0':
            data_list = []
            while (rs.error_code == '0') & rs.next():
                data_list.append(rs.get_row_data())
            
            if data_list:
                df = pd.DataFrame(data_list, columns=rs.fields)
                
                # 重命名列以完美适配您的回测系统
                df.rename(columns={
                    'date': '日期', 
                    'open': '开盘', 
                    'close': '收盘', 
                    'high': '最高', 
                    'low': '最低', 
                    'volume': '成交量'
                }, inplace=True)
                
                # 将字符串转换为数字格式，防止回测系统计算报错
                for col in ['开盘', '收盘', '最高', '最低', '成交量']:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                    
                save_path = os.path.join(SAVE_DIR, f"{raw_code}.csv")
                df.to_csv(save_path, index=False, encoding='utf-8-sig')
                success_count += 1
        else:
            print(f"\n[{raw_code}] 下载异常: {rs.error_msg}")
            
        # 实时打印进度条
        if i % 5 == 0 or i == len(raw_codes):
            print(f"📡 下载进度: {i}/{len(raw_codes)} (已成功: {success_count})", end='\r')
            
    # 4. 登出系统
    bs.logout()
    print(f"\n\n✅ 下载彻底完成！成功保存 {success_count} 个 CSV 文件到 {SAVE_DIR} 目录。")
    print("👉 现在您可以运行 python3 stock_backtest_pro.py 进行策略回测了！")