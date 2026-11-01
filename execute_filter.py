import os
import pandas as pd
import numpy as np
from datetime import datetime
import pytz
from concurrent.futures import ProcessPoolExecutor, as_completed

# --- 配置 ---
STOCK_DATA_DIR = 'stock_data'
STOCK_NAMES_FILE = 'stock_names.csv'
OUTPUT_BASE_DIR = 'screener_results'
TIMEZONE = 'Asia/Shanghai'
RSI_PERIOD = 21       # RSI 计算周期
MAX_PRICE = 20.0      # 价格上限配置

# 强化配置
RSI_LOWER_BOUND = 60.0 # RSI 黄金区下限
RSI_UPPER_BOUND = 75.0 # RSI 黄金区上限

# MACD 参数
MACD_SHORT_PERIOD = 12
MACD_LONG_PERIOD = 26
MACD_SIGNAL_PERIOD = 9

# 操作建议分类阈值
RSI_HIGH_THRESHOLD = 70.5 # RSI 高爆发阈值
MACD_ABS_THRESHOLD = 0.25  # MACD 绝对强度阈值

# 假设您的股票名称文件 stock_names.csv 包含两列：Code, Name
STOCK_NAME_COLUMNS = ['Code', 'Name'] 
# 您的历史数据CSV列名
DATA_COLUMNS = {
    'Date': '日期',
    'Open': '开盘',
    'High': '最高',
    'Low': '最低',
    'Close': '收盘',
    'Volume': '成交量'
}

# --- 辅助函数：计算 RSI (已修复警告) ---
def calculate_rsi(df, period=RSI_PERIOD):
    """计算相对强弱指数 (RSI)"""
    if 'Close' not in df.columns or not pd.api.types.is_numeric_dtype(df['Close']):
        return pd.Series([float('nan')] * len(df))

    delta = df['Close'].diff()
    gain = delta.clip(lower=0)
    loss = delta.clip(upper=0).abs() 

    ewm_params = {
        'com': period - 1, 
        'min_periods': period, 
        'adjust': False
    }

    avg_gain = gain.ewm(**ewm_params).mean()
    avg_loss = loss.ewm(**ewm_params).mean()

    # 计算相对强度 (RS)，并处理无穷大
    rs = avg_gain / avg_loss
    rs = rs.replace([float('inf'), float('-inf')], float('nan'))
    
    rsi = 100 - (100 / (1 + rs))
    rsi.loc[(avg_loss == 0) & (avg_gain > 0)] = 100.0

    return rsi

# --- 辅助函数：计算 MACD ---
def calculate_macd(df, short_period=MACD_SHORT_PERIOD, long_period=MACD_LONG_PERIOD, signal_period=MACD_SIGNAL_PERIOD):
    """计算 MACD 指标 (DIFF, DEA, MACD 柱线)"""
    if 'Close' not in df.columns or not pd.api.types.is_numeric_dtype(df['Close']):
        df['DIFF'] = df['DEA'] = df['MACD'] = pd.Series([float('nan')] * len(df))
        return df

    ema_short = df['Close'].ewm(span=short_period, adjust=False).mean()
    ema_long = df['Close'].ewm(span=long_period, adjust=False).mean()
    
    df['DIFF'] = ema_short - ema_long
    df['DEA'] = df['DIFF'].ewm(span=signal_period, adjust=False).mean()
    df['MACD'] = (df['DIFF'] - df['DEA']) * 2
    
    return df

# --- 筛选逻辑函数 (带操作建议) ---
def screen_stock(file_path):
    try:
        df = pd.read_csv(file_path)
        
        df = df.rename(columns={v: k for k, v in DATA_COLUMNS.items()})
        df = df.sort_values(by='Date').reset_index(drop=True)
        
        if len(df) < 250: 
            return None

        # 计算技术指标
        df['MA5'] = df['Close'].rolling(window=5).mean()
        df['MA10'] = df['Close'].rolling(window=10).mean()
        df['MA20'] = df['Close'].rolling(window=20).mean()
        df['MA250'] = df['Close'].rolling(window=250).mean() 
        df['VOL_MA5'] = df['Volume'].rolling(window=5).mean()
        df['VOL_MA10'] = df['Volume'].rolling(window=10).mean() 
        df['RSI'] = calculate_rsi(df, period=RSI_PERIOD)
        df = calculate_macd(df)
        
        if len(df) < 2:
            return None
            
        latest = df.iloc[-1]
        prev_day = df.iloc[-2]
        stock_code = os.path.basename(file_path).replace('.csv', '')
        
        if pd.isna(latest['MA250']) or pd.isna(latest['RSI']) or pd.isna(latest['DIFF']):
            return None

        # --- 筛选条件 (与增强版相同) ---
        is_uptrend = (latest['Close'] > latest['MA5'] and latest['MA5'] > latest['MA10'] and latest['MA10'] > latest['MA20'] and latest['MA20'] > latest['MA250'])
        
        high_20_days_excl_today = df['High'].iloc[-20:-1].max() 
        high_60_days_excl_today = df['High'].iloc[-60:-1].max() 
        is_breaking = (latest['Close'] > high_20_days_excl_today and latest['Close'] > high_60_days_excl_today)
        
        is_volume_up_today = latest['Volume'] > latest['VOL_MA5'] * 1.5 
        is_vol_uptrend = latest['VOL_MA5'] > latest['VOL_MA10']
        is_volume_ok = is_volume_up_today and is_vol_uptrend

        is_price_ok = (latest['Close'] >= 5.0) and (latest['Close'] <= MAX_PRICE)
        
        is_rsi_golden_zone = (latest['RSI'] >= RSI_LOWER_BOUND and latest['RSI'] <= RSI_UPPER_BOUND)

        body_ratio = (latest['Close'] - latest['Open']) / np.fmax(latest['High'] - latest['Low'], 1e-9)
        is_strong_k_bar = (latest['Close'] > latest['Open'] and body_ratio > 0.6)
        if pd.isna(is_strong_k_bar): is_strong_k_bar = False

        is_macd_golden_cross = (latest['DIFF'] > latest['DEA']) and (prev_day['DIFF'] <= prev_day['DEA'])
        is_macd_bar_enlarging = (latest['MACD'] > 0) and (latest['MACD'] > prev_day['MACD'])
        is_macd_signal = is_macd_golden_cross or is_macd_bar_enlarging
        
        # 组合所有增强条件
        if (is_uptrend and is_breaking and is_volume_ok and 
            is_price_ok and is_rsi_golden_zone and 
            is_strong_k_bar and is_macd_signal): 
            
            # --- 新增：操作建议分类逻辑 ---
            macd_avg_strength = (latest['DIFF'] + latest['DEA']) / 2
            
            if (latest['RSI'] >= RSI_HIGH_THRESHOLD or 
                (latest['RSI'] > (RSI_HIGH_THRESHOLD - 5) and macd_avg_strength >= MACD_ABS_THRESHOLD)):
                
                # 满足 RSI > 70 或 RSI > 65 且 MACD 绝对值强度 > 0.2
                suggestion = '高爆发/警惕型 (追涨)'
            else:
                suggestion = '稳健启动型 (回踩潜伏)'

            return {
                'Code': stock_code,
                'Name': None, 
                'Latest_Close': latest['Close'],
                'Latest_Date': latest['Date'],
                'RSI': latest['RSI'],
                'MACD_DIFF': latest['DIFF'],
                'MACD_DEA': latest['DEA'],
                'Operation_Suggestion': suggestion # 新增列
            }
        
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        
    return None

# --- 主运行逻辑 (调整最终列顺序) ---
def main():
    print("Starting stock screening process (Final Version: Technical + Suggestions)...")
    
    if not os.path.exists(STOCK_DATA_DIR):
        print(f"Error: Stock data directory '{STOCK_DATA_DIR}' not found. Exiting.")
        return

    all_files = [os.path.join(STOCK_DATA_DIR, f) for f in os.listdir(STOCK_DATA_DIR) if f.endswith('.csv')]
    
    if not all_files:
        print(f"No CSV files found in {STOCK_DATA_DIR}. Exiting.")
        return

    print(f"Found {len(all_files)} files. Processing in parallel...")
    results = []
    max_workers = min(os.cpu_count() or 4, 32) 
    with ProcessPoolExecutor(max_workers=max_workers) as executor: 
        future_to_file = {executor.submit(screen_stock, f): f for f in all_files}
        
        processed_count = 0
        for future in as_completed(future_to_file):
            processed_count += 1
            if processed_count % 1000 == 0 or processed_count == len(all_files):
                 print(f"Processed {processed_count}/{len(all_files)} files...")

            try:
                result = future.result()
                if result:
                    results.append(result)
            except Exception as exc:
                file_path = future_to_file[future]
                print(f'Error processing {file_path}: {exc}')
            
    
    if not results:
        print("No stocks matched the strict screening criteria.")
        return

    # 3. 结果合并与匹配
    screened_df = pd.DataFrame(results)
    
    try:
        # 最终输出列顺序：新增了 Operation_Suggestion
        results_cols = ['Code', 'Name', 'Latest_Close', 'Latest_Date', 'RSI', 'MACD_DIFF', 'MACD_DEA', 'Operation_Suggestion']
        
        names_df = pd.read_csv(STOCK_NAMES_FILE, names=STOCK_NAME_COLUMNS, header=None, dtype={'Code': str})
        names_df.columns = STOCK_NAME_COLUMNS
        
        names_df['Code'] = names_df['Code'].astype(str).str.zfill(6)
        screened_df['Code'] = screened_df['Code'].astype(str).str.zfill(6)
        
        final_df = pd.merge(screened_df, names_df, on='Code', how='left')
        
        final_df = final_df.drop(columns=['Name_x'], errors='ignore')
        if 'Name_y' in final_df.columns:
            final_df.rename(columns={'Name_y': 'Name'}, inplace=True)
        final_df['Name'] = final_df['Name'].fillna('N/A')
        
    except Exception as e:
        print(f"Error matching stock names from {STOCK_NAMES_FILE}: {e}. Outputting Code only.")
        final_df = screened_df
        final_df['Name'] = 'N/A'
        
    # 4. 排序：RSI小的排在前面 (升序排列)
    final_df = final_df.sort_values(by='RSI', ascending=True)

    # 5. 调整最终输出列顺序
    final_df = final_df.reindex(columns=results_cols, fill_value='N/A')
    
    print(f"Successfully matched, screened, and sorted {len(final_df)} stocks.")


    # 6. 生成带上海时区时间戳的文件名和目录
    try:
        tz_shanghai = pytz.timezone(TIMEZONE)
        now_shanghai = datetime.now(tz_shanghai)
    except pytz.exceptions.UnknownTimeZoneError:
        now_shanghai = datetime.now()
    
    output_dir = os.path.join(OUTPUT_BASE_DIR, now_shanghai.strftime('%Y-%m'))
    output_filename = now_shanghai.strftime('screened_stocks_%Y%m%d_%H%M%S.csv')
    output_path = os.path.join(output_dir, output_filename)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 7. 保存结果
    final_df.to_csv(output_path, index=False, encoding='utf-8')
    print(f"Screening results saved successfully to: {output_path}")

if __name__ == '__main__':
    main()
