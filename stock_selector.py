import pandas as pd
import numpy as np
import os
import glob
from datetime import datetime
from multiprocessing import Pool, cpu_count, freeze_support
import pytz 

# --- 配置常量：包含所有优化后的参数 (进一步放宽) ---
DATA_DIR = 'stock_data'
NAME_MAP_FILE = 'stock_names.csv' # 股票名称映射文件路径
TIMEZONE_SH = pytz.timezone('Asia/Shanghai')

MA_SHORT = 5
MA_MEDIUM = 10
N_DAYS = 20      # MA20
M_DAYS = 60      # MA60 
P_DAYS = 250     # MA250 

WINDOW_DAYS = 60 # 平台整理观察期
# 平台整理宽度 (已放宽到 20%)
PRICE_RANGE_PCT = 0.20 

# 极低波动率过滤器参数 (前 5 日)
LOW_VOLATILITY_DAYS = 5          
# **进一步放宽：由 7.0 ➡️ 10.0**
MAX_AMPLITUDE_PCT = 10.0         # 突破前 5 日 K 线总振幅最大值限制

# 底部堆量确认参数
ACCUMULATION_DAYS = 30          
# **进一步放宽：由 25.0 ➡️ 20.0**
MIN_ACCUMULATION_TURNOVER = 20.0 # 30日累积换手率要求 (20%)

# --- 【精细过滤参数】 ---
# 1. 突破前极小实体波动率 (形态扁平化)
# **进一步放宽：由 4.0 ➡️ 5.0**
MAX_BODY_AMPLITUDE_PCT = 5.0    # 要求前5日K线实体平均波动低于 5.0%
# 2. 启动日实体饱满度 
MIN_BODY_TO_RANGE_RATIO = 0.55  # 要求启动日实体占比至少 55%

# --- 启动参数 ---
MIN_LAUNCH_PCT = 2.0            # 最小启动涨幅
VOLUME_MULTIPLIER = 1.1         # 启动量比均量要求
TURNOVER_THRESHOLD = 1.5        # 启动日换手率要求
# 均线粘合度 (虽然条件被移除，但仍保留指标计算和 COIL_PCT 常量)
COIL_PCT = 0.05                 

# --- 核心修正：列名映射字典 (不变) ---
COLUMN_MAP = {
    '日期': 'Date', '股票代码': 'Code', '开盘': 'Open', '收盘': 'Close', 
    '最高': 'High', '最低': 'Low', '成交量': 'Volume', '成交额': 'Amount', 
    '振幅': 'Amplitude', '涨跌幅': 'ChangePct', '涨跌额': 'ChangeAmt', '换手率': 'Turnover'
}

def standardize_columns(df):
    """标准化列名，并过滤掉无效数据。"""
    df.rename(columns=COLUMN_MAP, inplace=True)
    if 'Close' in df.columns:
        df = df[df['Close'] > 0].copy()
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df.dropna(subset=['Date'], inplace=True)
    return df

def calculate_indicators(df):
    """计算所需的均线、成交量、粘合度和波动率指标"""
    
    # 确保所有涉及计算的列都是数值类型
    for col in ['Close', 'Volume', 'ChangePct', 'Turnover', 'Amplitude', 'Open', 'High', 'Low']: 
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    df = df.sort_values(by='Date').reset_index(drop=True)
    
    # --- 均线计算 ---
    df[f'MA{MA_SHORT}'] = df['Close'].rolling(window=MA_SHORT, min_periods=1).mean()
    df[f'MA{MA_MEDIUM}'] = df['Close'].rolling(window=MA_MEDIUM, min_periods=1).mean()
    df[f'MA{N_DAYS}'] = df['Close'].rolling(window=N_DAYS, min_periods=1).mean() # MA20
    df['MA_P'] = df['Close'].rolling(window=P_DAYS, min_periods=1).mean() # MA250
    
    # --- 成交量和整理期指标 ---
    df['VOL_MEAN'] = df['Volume'].rolling(window=WINDOW_DAYS, min_periods=1).mean()
    df['MAX_CLOSE'] = df['Close'].rolling(window=WINDOW_DAYS, min_periods=1).max()
    df['MIN_CLOSE'] = df['Close'].rolling(window=WINDOW_DAYS, min_periods=1).min()
    
    # --- 粘合度指标 ---
    df['MA_COIL_DIFF'] = abs(df[f'MA{MA_SHORT}'] - df[f'MA{MA_MEDIUM}']) / df['Close']
    COIL_DAYS = 10 
    df['COILED_PREV'] = (df['MA_COIL_DIFF'].shift(1).rolling(window=COIL_DAYS, min_periods=COIL_DAYS).max() < COIL_PCT)
    
    # --- 突破前低波动率指标 (K线总振幅) ---
    df['LOW_VOL_PREV'] = df['Amplitude'].shift(1).rolling(window=LOW_VOLATILITY_DAYS, min_periods=LOW_VOLATILITY_DAYS).max()

    # --- 新增：突破前 K 线实体波动率 (用于形态扁平化过滤) ---
    df['DAILY_BODY_AMPLITUDE'] = (df['Close'] - df['Open']).abs() / df['Close'].shift(1)
    df['LOW_BODY_AMPLITUDE_PREV'] = df['DAILY_BODY_AMPLITUDE'].shift(1).rolling(window=LOW_VOLATILITY_DAYS, min_periods=LOW_VOLATILITY_DAYS).max()

    # --- 底部温和堆量指标：观察期内换手率总和 ---
    df['ACCUM_TURNOVER'] = df['Turnover'].shift(1).rolling(window=ACCUMULATION_DAYS, min_periods=ACCUMULATION_DAYS).sum()

    return df

def select_stock(df, stock_name):
    """根据优化的条件筛选符合启动特征的股票"""
    
    required_days = max(P_DAYS, ACCUMULATION_DAYS) + LOW_VOLATILITY_DAYS
    if len(df) < required_days:
        return None
    
    df = calculate_indicators(df)
    
    # 过滤掉计算指标后产生的NaN值
    df.dropna(subset=['Close', 'ChangePct', 'Turnover', 'MA_P', 'COILED_PREV', 'VOL_MEAN', 
                      'LOW_VOL_PREV', 'ACCUM_TURNOVER', 'LOW_BODY_AMPLITUDE_PREV', 
                      'Open', 'High', 'Low'], inplace=True)
    
    if df.empty:
        return None

    latest_data = df.iloc[-1]
    stock_code = latest_data['Code']
    
    # --- 最终策略选股条件 (总共 9 个过滤条件，移除了均线粘合) ---
    
    # 1. 启动信号：捕捉强劲的、非涨停的启动日
    if latest_data['ChangePct'] < MIN_LAUNCH_PCT:
        return None
        
    # 2. 换手率确认
    if latest_data['Turnover'] < TURNOVER_THRESHOLD:
        return None
        
    # 3. 量能确认条件 
    if latest_data['Volume'] < (latest_data['VOL_MEAN'] * VOLUME_MULTIPLIER):
        return None
    
    # 4. 整理平台确认条件 
    price_range = (latest_data['MAX_CLOSE'] - latest_data['MIN_CLOSE']) / latest_data['Close']
    if price_range > PRICE_RANGE_PCT:
        return None
        
    # 5. 均线粘合启动 (***此条件已移除***)
    # if latest_data['COILED_PREV'] == False:
    #     return None

    # 6. 突破前极低波动率确认 (K线总振幅) 
    if latest_data['LOW_VOL_PREV'] > MAX_AMPLITUDE_PCT:
        return None

    # 7. 突破前极小实体波动 (保证形态的扁平化) 
    if latest_data['LOW_BODY_AMPLITUDE_PREV'] * 100 > MAX_BODY_AMPLITUDE_PCT:
        return None
        
    # 8. 底部堆量确认 
    if latest_data['ACCUM_TURNOVER'] < MIN_ACCUMULATION_TURNOVER:
        return None
        
    # 9. 均线启动条件 (灵活多头排列)
    ma_short = latest_data[f'MA{MA_SHORT}']
    ma_medium = latest_data[f'MA{MA_MEDIUM}']
    ma_long = latest_data[f'MA{N_DAYS}'] # MA20
    
    # 要求：股价站上所有均线，且短期均线多头 (MA5 > MA10)
    if not (latest_data['Close'] > ma_short and 
                 latest_data['Close'] > ma_long and 
                 ma_short > ma_medium): 
        return None
        
    # 10. 启动日实体饱满度 (确保启动阳线实体强劲)
    total_range = latest_data['High'] - latest_data['Low']
    body_length = latest_data['Close'] - latest_data['Open']
    
    if total_range <= 0.0 or (body_length / total_range < MIN_BODY_TO_RANGE_RATIO):
        return None

    # 满足所有条件，返回结果
    body_ratio = body_length / total_range
    
    result = {
        '日期': latest_data['Date'].strftime('%Y-%m-%d'),
        '股票代码': stock_code,
        '股票名称': stock_name, 
        '收盘价': f"{latest_data['Close']:.2f}",
        '涨跌幅': f"{latest_data['ChangePct']:.2f}%",
        f'MA{MA_SHORT}': f"{ma_short:.2f}",
        f'MA{MA_MEDIUM}': f"{ma_medium:.2f}", 
        f'MA{N_DAYS}': f"{ma_long:.2f}", 
        '成交量比均量': f"{latest_data['Volume'] / latest_data['VOL_MEAN']:.2f}",
        '换手率%': f"{latest_data['Turnover']:.2f}%",
        '整理期波动%': f"{price_range * 100:.2f}%",
        '突破前最大振幅%': f"{latest_data['LOW_VOL_PREV']:.2f}%",
        '前5日实体波动%': f"{latest_data['LOW_BODY_AMPLITUDE_PREV'] * 100:.2f}%",
        '实体饱满度': f"{body_ratio * 100:.2f}%",
        '30日累积换手率%': f"{latest_data['ACCUM_TURNOVER']:.2f}%",
        '是否低于MA250': '是' if latest_data['Close'] < latest_data['MA_P'] else '否'
    }
    
    return result

def process_file(file_info):
    """多进程工作函数：读取文件，标准化列名，计算指标并筛选股票"""
    file_path, name_map = file_info
    
    try:
        stock_code_raw = os.path.basename(file_path).replace('.csv', '')
        stock_code = stock_code_raw.zfill(6)
        stock_name = name_map.get(stock_code, "未知")

        # 尝试使用 gbk 编码读取，防止中文文件名或内容乱码
        try:
            df = pd.read_csv(file_path, encoding='gbk')
        except UnicodeDecodeError:
            df = pd.read_csv(file_path, encoding='utf-8')

        df = standardize_columns(df) 
        df['Code'] = stock_code
        
        result = select_stock(df, stock_name)
        if result:
            return result
    except Exception as e:
        # print(f"处理文件 {file_path} 失败: {e}")
        pass
    return None

def main():
    """主函数：并行扫描所有文件并执行选股"""
    # 确保数据目录存在
    os.makedirs(DATA_DIR, exist_ok=True)
    
    # --- 加载名称映射文件 ---
    name_map = {}
    
    if os.path.exists(NAME_MAP_FILE):
        print(f"正在加载名称映射文件 '{NAME_MAP_FILE}'...")
        
        delimiters = [',', '\t', ';']
        encodings = ['utf-8', 'utf-8-sig', 'gbk'] 
        found_map = False
        
        for enc in encodings:
            for delim in delimiters:
                if found_map: break
                try:
                    df_names = pd.read_csv(NAME_MAP_FILE, 
                                           dtype=str,
                                           encoding=enc, 
                                           sep=delim,
                                           header=None, skiprows=1, names=['code', 'name']) 
                                           
                    if 'code' in df_names.columns and 'name' in df_names.columns:
                        
                        df_names['code'] = df_names['code'].astype(str).str.zfill(6)
                        
                        print(f"成功使用编码'{enc}'和分隔符'{repr(delim)}'加载名称映射。")
                        name_map = df_names.set_index('code')['name'].to_dict()
                        found_map = True
                        break 
                except Exception:
                    continue

        if not name_map:
            print("警告：无法正确解析名称映射文件，名称将显示为'未知'。")
    else:
        print(f"警告：名称映射文件 '{NAME_MAP_FILE}' 未找到，名称将显示为'未知'。")

    # 获取所有股票文件路径
    file_paths = glob.glob(os.path.join(DATA_DIR, '*.csv'))
    
    if not file_paths:
        print(f"错误：在 '{DATA_DIR}' 目录下未找到任何 CSV 文件。")
        return

    # 准备并行任务列表 (将 name_map 传给每个任务)
    file_info_list = [(path, name_map) for path in file_paths]
    
    # 限制进程数，防止资源耗尽
    num_processes = min(cpu_count(), 8) 
    print(f"🚀 启动并行处理，使用 {num_processes} 个进程处理 {len(file_paths)} 个文件。")

    with Pool(num_processes) as pool:
        raw_results = pool.map(process_file, file_info_list)

    selected_stocks = [r for r in raw_results if r is not None]
    
    if selected_stocks:
        results_df = pd.DataFrame(selected_stocks)
        
        # 结果排序和输出
        results_df = results_df.sort_values(by=['日期', '股票代码'], ascending=[False, True])
        
        current_time = datetime.now(TIMEZONE_SH)
        output_dir_name = current_time.strftime('%Y%m')
        
        os.makedirs(output_dir_name, exist_ok=True)
        timestamp = current_time.strftime('%Y%m%d%H%M%S')
        # 更改文件名以反映进一步放宽
        output_filename = f'selected_stocks_Final_Loose_{timestamp}.csv'
        output_path = os.path.join(output_dir_name, output_filename)
        
        # 整理最终列顺序
        final_cols = ['日期', '股票代码', '股票名称', '收盘价', '涨跌幅', '换手率%', '成交量比均量', 
                      '整理期波动%', '突破前最大振幅%', '前5日实体波动%', '实体饱满度', '30日累积换手率%', 
                      f'MA{MA_SHORT}', f'MA{MA_MEDIUM}', f'MA{N_DAYS}', '是否低于MA250']
        
        results_df = results_df.reindex(columns=final_cols)
        
        # 使用带 BOM 的 UTF-8 编码方便 Excel 打开
        results_df.to_csv(output_path, index=False, encoding='utf-8-sig') 
        print(f"\n🎉 选股结果已保存到: {output_path}")
        print(f"✅ 选股数量: {len(selected_stocks)} 支")
    else:
        print("\n😔 没有股票满足筛选条件。")

if __name__ == '__main__':
    if os.name == 'nt':
        freeze_support()
    main()
