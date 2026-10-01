# short_ambush_screener.py (最终优化版 V2.2 - 加入 MACD 动能确认)

import pandas as pd
import numpy as np
import os
import glob
from datetime import datetime, timedelta
from multiprocessing import Pool, cpu_count
import pytz

# --- 配置参数 (V2.2 优化版) ---
STOCK_DATA_DIR = 'stock_data'
STOCK_NAMES_FILE = 'stock_names.csv'
OUTPUT_BASE_DIR = 'screener_results'

# 筛选条件参数
N_DAYS_ADJ_LIMIT = 25      # 调整周期限制 (交易日): 25 天
M_AVERAGE_DAYS = 30        # 中期支撑均线: M30
SUPPORT_TOLERANCE = 0.05   # 支撑容忍度: M30 上下 5% 范围内
MAX_VOLUME_RATIO = 0.7     # 缩量标准: 当前成交量 <= 近10日平均量的 70%
MIN_FALL_PERCENT = 0.10    # 调整跌幅下限: 10%
MAX_FALL_PERCENT = 0.50    # 调整跌幅上限: 50%

# 强势上涨验证参数
MIN_RISE_PERCENT = 0.30    # 前期上涨必须达到的最小涨幅 (30%)
RISE_VOLUME_MULTIPLIER = 1.5 # 上涨周期平均量必须 >= 长期平均量的 1.5 倍
MIN_SINGLE_DAY_GAIN = 0.07 # 上涨周期内必须至少有一天涨幅达到 7% (大阳线)
MIN_YANG_YIN_RATIO = 1.5   # 上涨周期内阳线数量/阴线数量 >= 1.5

# 企稳验证参数
MAX_AMPLITUDE_RECENT = 0.03 # 最近3天平均日振幅 <= 3%

# --- 数据处理和筛选逻辑 ---

def load_stock_data(filepath):
    """加载单个股票数据文件，进行数据清洗、列名映射和均线计算"""
    try:
        basename = os.path.basename(filepath)
        stock_code = os.path.splitext(basename)[0]

        df = pd.read_csv(filepath)
        df.columns = df.columns.str.strip() 
        
        # 【中文列名映射】
        column_map = {
            '日期': 'Date',
            '开盘': 'Open',
            '收盘': 'Close',
            '最高': 'High',
            '最低': 'Low',
            '成交量': 'Volume'
        }
        df = df.rename(columns=column_map)
        
        # 统一日期格式并排序
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values(by='Date').reset_index(drop=True)
        
        # 确保有足够的历史数据 (需要 MA30 和 MACD)
        if len(df) < M_AVERAGE_DAYS + 60:
            return None

        # 计算日振幅 (Range / Close_prev)
        df['Close_Prev'] = df['Close'].shift(1)
        df['DailyAmplitude'] = (df['High'] - df['Low']) / df['Close_Prev'] 

        # 计算均线 M30
        df['MA30'] = df['Close'].rolling(window=M_AVERAGE_DAYS).mean()
        
        # 计算近10日平均成交量（用于缩量判断）
        df['AvgVolume10D'] = df['Volume'].rolling(window=10).mean()
        
        # 标记阳线/阴线: 阳线(Close > Open) = 1, 阴线(Close < Open) = -1
        df['CandleType'] = np.select(
            [df['Close'] > df['Open'], df['Close'] < df['Open']],
            [1, -1],
            default=0  # 平盘
        )
        
        # V2.2 新增：计算 MACD 指标 (用于动能确认)
        exp12 = df['Close'].ewm(span=12, adjust=False).mean()
        exp26 = df['Close'].ewm(span=26, adjust=False).mean()
        df['DIFF'] = exp12 - exp26
        df['DEA'] = df['DIFF'].ewm(span=9, adjust=False).mean()
        df['MACD_Bar'] = df['DIFF'] - df['DEA'] # MACD 柱体

        df['StockCode'] = stock_code
        return df
    except Exception as e:
        print(f"Error processing file {filepath}: {e}")
        return None

def check_stock_conditions(df):
    """对单个股票DataFrame应用筛选逻辑"""
    stock_code = df['StockCode'].iloc[0]
    
    # 获取最新数据
    latest = df.iloc[-1]
    # 确保关键指标有值
    if pd.isna(latest['MA30']) or pd.isna(latest['MACD_Bar']): return None 
    
    # --- 1. 调整周期与跌幅限制 ---
    recent_period = N_DAYS_ADJ_LIMIT * 2 
    recent_data = df.tail(recent_period)
    
    if len(recent_data) < 2: return None

    # 找到最高点和其索引
    max_close_idx = recent_data['Close'].idxmax()
    peak_close = recent_data.loc[max_close_idx]['Close']
    
    # a. 调整周期限制 (交易日数)
    adjustment_days_count = latest.name - max_close_idx
    if adjustment_days_count <= 0 or adjustment_days_count > N_DAYS_ADJ_LIMIT:
        return None
        
    # b. 调整跌幅限制 (MIN_FALL_PERCENT 到 50%)
    fall_percent = (peak_close - latest['Close']) / peak_close
    if not (MIN_FALL_PERCENT <= fall_percent <= MAX_FALL_PERCENT): 
        return None

    # --- 2. 【硬性要求】验证前期强势上涨形态 (代码保持不变) ---
    
    # 查找最高点前30个交易日的数据
    start_index = max(0, max_close_idx - 30)
    pre_peak_data = df.loc[start_index : max_close_idx].copy()
    
    if len(pre_peak_data) < 10: return None 
    
    # 2a. 验证涨幅：从周期内最低点到最高点的涨幅必须超过 MIN_RISE_PERCENT
    initial_low = pre_peak_data['Close'].min()
    required_increase = (peak_close - initial_low) / initial_low
    if required_increase < MIN_RISE_PERCENT: 
        return None
        
    # 2b. 验证量能：上涨周期内的平均成交量必须显著放大
    long_term_avg_vol = df['Volume'].iloc[start_index - 60 : start_index].mean()
    if not pd.isna(long_term_avg_vol) and long_term_avg_vol > 0:
        current_rise_avg_vol = pre_peak_data['Volume'].mean()
        if current_rise_avg_vol < long_term_avg_vol * RISE_VOLUME_MULTIPLIER:
            return None

    # 2c. 验证大阳线：上涨周期内必须至少有一天涨幅超过 MIN_SINGLE_DAY_GAIN (7%)
    pre_peak_data['DailyGain'] = (pre_peak_data['Close'] - pre_peak_data['Close'].shift(1)) / pre_peak_data['Close'].shift(1)
    if (pre_peak_data['DailyGain'].dropna() >= MIN_SINGLE_DAY_GAIN).sum() == 0:
        return None
        
    # 2d. 验证阳线数量优势
    yang_count = (pre_peak_data['CandleType'] == 1).sum()
    yin_count = (pre_peak_data['CandleType'] == -1).sum()
    if yin_count > 0 and (yang_count / yin_count) < MIN_YANG_YIN_RATIO:
        return None


    # --- 3. 支撑有效性确认：回踩 M30 且在容忍范围内 (修复 V2.1 逻辑) ---
    ma_support = latest['MA30']
    
    support_low_limit = ma_support * (1 - SUPPORT_TOLERANCE)
    support_high_limit = ma_support * (1 + SUPPORT_TOLERANCE)
    
    # 要求收盘价必须位于支撑带内 (排除破位股)
    if not (latest['Close'] >= support_low_limit and 
            latest['Close'] <= support_high_limit):
        return None

    # --- 4. 量能极致萎缩（启动前夜） (<= 70% 平均量) ---
    if latest['Volume'] > latest['AvgVolume10D'] * MAX_VOLUME_RATIO:
        return None

    # --- 5. 企稳迹象：最近 3 天平均日振幅 <= 3% ---
    if len(df) < 3: return None
    recent_amplitude = df['DailyAmplitude'].tail(3).mean()
    if recent_amplitude > MAX_AMPLITUDE_RECENT:
        return None
        
    # --- 6. V2.2 新增：动能转换确认 (要求 MACD 柱体开始变大) ---
    if latest.name < 1: return None # 确保有前一天数据
    
    latest_macd_bar = latest['MACD_Bar']
    # 注意：MACD 柱体在 0 轴下方为负值。要求最新柱体 > 前一日柱体，即柱体收缩或开始变长。
    prev_macd_bar = df.loc[latest.name - 1, 'MACD_Bar']
    
    if latest_macd_bar <= prev_macd_bar:
        return None

    # 满足所有条件
    return {
        'Code': stock_code,
        'Name': None, 
        'Close': latest['Close'],
        'MA_Used': f'MA{M_AVERAGE_DAYS}',
        'MA_Value': ma_support,
        'MaxClose': peak_close,
        'Fall_%': f"{fall_percent * 100:.2f}%",
        'AdjustmentDays': adjustment_days_count, 
        'LatestDate': latest['Date'].strftime('%Y-%m-%d')
    }

# --- 后续主函数和并行处理函数保持不变 ---

def process_all_stocks(stock_files):
    """并行处理所有股票文件"""
    num_processes = min(cpu_count(), 8) 
    
    with Pool(num_processes) as pool:
        data_frames = pool.map(load_stock_data, stock_files)
        data_frames = [df for df in data_frames if df is not None]
        
        results = pool.map(check_stock_conditions, data_frames)
        
    return [r for r in results if r is not None]

def load_stock_names():
    """加载股票代码和名称的映射"""
    try:
        names_df = pd.read_csv(STOCK_NAMES_FILE)
        if names_df.shape[1] < 2: 
             names_df = pd.read_csv(STOCK_NAMES_FILE, header=None)
        
        names_df.columns = ['Code', 'Name']
        names_df['Code'] = names_df['Code'].astype(str).str.zfill(6) 
        return names_df.set_index('Code')['Name'].to_dict()
    except Exception as e:
        print(f"Warning: Could not load stock names file {STOCK_NAMES_FILE}: {e}")
        return {}

def main():
    """主函数，负责执行、匹配和保存结果"""
    print(f"Starting stock screening at {datetime.now()}")
    
    data_path = os.path.join(STOCK_DATA_DIR, '*.csv')
    stock_files = glob.glob(data_path)
    
    if not stock_files:
        print(f"Error: No CSV files found in {STOCK_DATA_DIR}. Exiting.")
        return

    print(f"Processing {len(stock_files)} stock files...")

    # 并行处理数据
    screened_results = process_all_stocks(stock_files)
    
    if not screened_results:
        print("No stocks matched the 'Short-term Ambush' criteria.")
        return

    # 转换为 DataFrame
    results_df = pd.DataFrame(screened_results)
    
    # --- 匹配股票名称 ---
    stock_names_map = load_stock_names()
    results_df['Code'] = results_df['Code'].astype(str).str.zfill(6) 
    results_df['Name'] = results_df['Code'].map(stock_names_map).fillna('N/A')
    
    # 调整列顺序
    results_df = results_df[['Code', 'Name', 'LatestDate', 'Close', 'MA_Used', 'MA_Value', 'MaxClose', 'Fall_%', 'AdjustmentDays']]
    
    print(f"Found {len(results_df)} stocks matching the criteria.")
    print("\nMatched Stocks:")
    print(results_df)

    # --- 结果保存和推送配置 ---
    
    shanghai_tz = pytz.timezone('Asia/Shanghai')
    now_shanghai = datetime.now(shanghai_tz)
    
    output_dir = os.path.join(
        OUTPUT_BASE_DIR, 
        now_shanghai.strftime('%Y'), 
        now_shanghai.strftime('%m')
    )
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = now_shanghai.strftime('%Y%m%d_%H%M%S')
    output_filename = f"{timestamp}_screener.csv"
    output_filepath = os.path.join(output_dir, output_filename)
    
    results_df.to_csv(output_filepath, index=False, encoding='utf-8')
    print(f"\nResults saved successfully to: {output_filepath}")

if __name__ == '__main__':
    main()
