# -*- coding: utf-8 -*-
"""
终极版选股脚本（2025-11-29 RSI & MACD复合版 V7：新增RSI动能确认）
专抓：放量大阳线突破 + MACD低位金叉（宽松） + RSI动能确认
目的：在数据修正后，加入RSI指标来进一步验证突破动能。
"""

import pandas as pd
import numpy as np
import os
import glob
from datetime import datetime
from multiprocessing import Pool, cpu_count, freeze_support
import pytz

# --- 配置常量（RSI & MACD复合版 V7）---
DATA_DIR = 'stock_data'
NAME_MAP_FILE = 'stock_names.csv'
TIMEZONE_SH = pytz.timezone('Asia/Shanghai')

MA_SHORT = 5
MA_MEDIUM = 10
P_DAYS = 250
WINDOW_DAYS = 60

# 启动条件（保持不变）
MIN_LAUNCH_PCT = 5.0              # 启动涨幅 ≥5.0%
VOLUME_MULTIPLIER = 1.8           # 放量倍数 ≥1.8倍均量
MIN_BODY_TO_RANGE_RATIO = 0.65    # 实体占比 ≥65%
MIN_TURNOVER_LAUNCH = 4.0         # 启动日换手率 ≥4%

# MACD低位金叉（沿用 V6 的宽松参数）
MACD_EMA_SHORT = 12
MACD_EMA_LONG = 26
MACD_EMA_DIFF = 9
MACD_MAX_FOR_LOW = 2.5            # MACD值 ≤2.5算低位

# RSI 动能确认 (新增)
RSI_WINDOW = 14
RSI_MIN = 50.0                    # RSI ≥ 50，确认多头趋势
RSI_MAX = 80.0                    # RSI ≤ 80，避免短期过度超买

# --- 列名映射 ---
COLUMN_MAP = {
    '日期': 'Date', '股票代码': 'Code', '开盘': 'Open', '收盘': 'Close',
    '最高': 'High', '最低': 'Low', '成交量': 'Volume', '成交额': 'Amount',
    '振幅': 'Amplitude', '涨跌幅': 'ChangePct', '涨跌额': 'ChangeAmt', '换手率': 'Turnover'
}

def standardize_columns(df):
    df.rename(columns=COLUMN_MAP, inplace=True)
    if 'Close' in df.columns:
        df = df[df['Close'] > 0].copy()
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df.dropna(subset=['Date'], inplace=True)
    return df

def calculate_indicators(df):
    # 修正数据类型转换错误
    for col in ['Close', 'Volume', 'ChangePct', 'Turnover', 'Amplitude', 'Open', 'High', 'Low']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    df = df.sort_values(by='Date').reset_index(drop=True)
    
    # 均线
    df[f'MA{MA_SHORT}'] = df['Close'].rolling(window=MA_SHORT, min_periods=1).mean()
    df[f'MA{MA_MEDIUM}'] = df['Close'].rolling(window=MA_MEDIUM, min_periods=1).mean()
    df['MA250'] = df['Close'].rolling(window=P_DAYS, min_periods=1).mean()
    
    # 成交量与平台指标 (计算以供输出)
    df['VOL_MEAN_60'] = df['Volume'].rolling(window=WINDOW_DAYS, min_periods=1).mean()
    df['MAX_CLOSE_60'] = df['Close'].rolling(window=WINDOW_DAYS, min_periods=1).max()
    df['MIN_CLOSE_60'] = df['Close'].rolling(window=WINDOW_DAYS, min_periods=1).min()
    df['LOW_VOL_PREV_5'] = df['Amplitude'].shift(1).rolling(window=5, min_periods=5).max()
    df['DAILY_BODY_PCT'] = (df['Close'] - df['Open']).abs() / df['Close'].shift(1) * 100
    df['LOW_BODY_PREV_5'] = df['DAILY_BODY_PCT'].shift(1).rolling(window=5, min_periods=5).mean()
    df['ACCUM_TURNOVER_30D'] = df['Turnover'].shift(1).rolling(window=30, min_periods=30).sum()
    
    # MACD 
    ema12 = df['Close'].ewm(span=MACD_EMA_SHORT, adjust=False).mean()
    ema26 = df['Close'].ewm(span=MACD_EMA_LONG, adjust=False).mean()
    df['DIFF'] = ema12 - ema26
    df['DEA'] = df['DIFF'].ewm(span=MACD_EMA_DIFF, adjust=False).mean()
    df['MACD'] = 2 * (df['DIFF'] - df['DEA'])
    
    # RSI (新增计算)
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(com=RSI_WINDOW - 1, adjust=False).mean()
    avg_loss = loss.ewm(com=RSI_WINDOW - 1, adjust=False).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # KDJ (仅计算，不筛选)
    kdj_window = 9
    low_min = df['Low'].rolling(window=kdj_window).min()
    high_max = df['High'].rolling(window=kdj_window).max()
    rsv = (df['Close'] - low_min) / (high_max - low_min) * 100
    df['K'] = rsv.ewm(com=2, adjust=False).mean()
    df['D'] = df['K'].ewm(com=2, adjust=False).mean()
    df['J'] = 3 * df['K'] - 2 * df['D']
    
    return df

def select_stock(df, stock_name):
    required_days = max(P_DAYS, 60) + 30
    if len(df) < required_days:
        return None
    
    df = calculate_indicators(df)
    
    required_cols = ['Close', 'ChangePct', 'Turnover', 'MA250', 'VOL_MEAN_60', 'Open', 'High', 'Low', 
                     f'MA{MA_SHORT}', f'MA{MA_MEDIUM}', 'DIFF', 'DEA', 'RSI'] # 确保 RSI 在 required_cols
    df.dropna(subset=required_cols, inplace=True)
    
    if df.empty:
        return None
    
    latest = df.iloc[-1]
    prev = df.iloc[-2] if len(df) >= 2 else latest
    
    # 1. 启动日必须是大阳线（涨幅+实体饱满）
    if latest['ChangePct'] < MIN_LAUNCH_PCT:
        return None
    total_range = latest['High'] - latest['Low']
    if total_range <= 0:
        return None
    body_ratio = abs(latest['Close'] - latest['Open']) / total_range
    if body_ratio < MIN_BODY_TO_RANGE_RATIO:
        return None
    
    # 2. 放巨量 + 高换手确认
    if latest['Volume'] < latest['VOL_MEAN_60'] * VOLUME_MULTIPLIER:
        return None
    if latest['Turnover'] < MIN_TURNOVER_LAUNCH:
        return None
    
    # 6. MACD低位金叉（宽松条件）
    macd_gold_cross_recent = (prev['DIFF'] <= prev['DEA']) and (latest['DIFF'] > latest['DEA'])
    macd_gold_cross_recent = macd_gold_cross_recent or \
        (len(df) >= 3 and df.iloc[-3]['DIFF'] <= df.iloc[-3]['DEA'] and latest['DIFF'] > latest['DEA'])
    if not macd_gold_cross_recent:
        return None
    if latest['MACD'] > MACD_MAX_FOR_LOW: # MACD_MAX_FOR_LOW = 2.5 (宽松)
        return None

    # 7. RSI 动能确认 (新增筛选条件)
    if not (latest['RSI'] >= RSI_MIN and latest['RSI'] <= RSI_MAX): # RSI 介于 50.0 和 80.0 之间
        return None
    
    # 8. 均线多头 + 站上短期均线
    if not (latest['Close'] > latest[f'MA{MA_SHORT}'] > latest[f'MA{MA_MEDIUM}']):
        return None
    
    # 全部通过 → 返回结果
    
    # 重新计算输出字段
    price_range = (latest['MAX_CLOSE_60'] - latest['MIN_CLOSE_60']) / latest['Close'] if 'MAX_CLOSE_60' in latest and 'MIN_CLOSE_60' in latest and latest['Close'] > 0 else np.nan
    low_vol_prev_5 = latest['LOW_VOL_PREV_5'] if 'LOW_VOL_PREV_5' in latest else np.nan
    low_body_prev_5 = latest['LOW_BODY_PREV_5'] if 'LOW_BODY_PREV_5' in latest else np.nan
    accum_turnover_30d = latest['ACCUM_TURNOVER_30D'] if 'ACCUM_TURNOVER_30D' in latest else np.nan
    
    is_above_ma250 = latest['Close'] > latest['MA250']
    ma_deviation = (latest['Close'] / latest['MA250'] - 1) * 100 if is_above_ma250 and latest['MA250'] > 0 else 0
    
    result = {
        '日期': latest['Date'].strftime('%Y-%m-%d'),
        '股票代码': latest['Code'],
        '股票名称': stock_name,
        '收盘价': f"{latest['Close']:.2f}",
        '涨跌幅': f"{latest['ChangePct']:.2f}%",
        '换手率%': f"{latest['Turnover']:.2f}%",
        '成交量比均量': f"{latest['Volume']/latest['VOL_MEAN_60']:.2f}",
        '整理期波动%': f"{price_range*100:.2f}%" if not np.isnan(price_range) else 'N/A',
        '突破前5日最大振幅%': f"{low_vol_prev_5:.2f}%" if not np.isnan(low_vol_prev_5) else 'N/A',
        '前5日实体波动%': f"{low_body_prev_5:.2f}%" if not np.isnan(low_body_prev_5) else 'N/A',
        '实体饱满度': f"{body_ratio*100:.2f}%",
        '30日累积换手%': f"{accum_turnover_30d:.2f}%" if not np.isnan(accum_turnover_30d) else 'N/A',
        'MACD': f"{latest['MACD']:.3f}",
        'RSI': f"{latest['RSI']:.1f}", # 新增输出 RSI
        'J值': f"{latest['J']:.1f}" if 'J' in latest else 'N/A',
        f'MA{MA_SHORT}': f"{latest[f'MA{MA_SHORT}']:.2f}",
        f'MA{MA_MEDIUM}': f"{latest[f'MA{MA_MEDIUM}']:.2f}",
        '是否低于MA250': '否' if is_above_ma250 else '是',
        'MA250偏离%': f"{ma_deviation:.2f}%" if is_above_ma250 else '低于年线'
    }
    
    return result

# ============== 多进程处理部分（保持不变）==============
def process_file(file_info):
    file_path, name_map = file_info
    try:
        stock_code_raw = os.path.basename(file_path).replace('.csv', '')
        stock_code = stock_code_raw.zfill(6)
        stock_name = name_map.get(stock_code, "未知")
        try:
            df = pd.read_csv(file_path, encoding='gbk')
        except UnicodeDecodeError:
            df = pd.read_csv(file_path, encoding='utf-8')
        except Exception:
            try:
                df = pd.read_csv(file_path, encoding='utf-8', sep='\t')
            except Exception:
                try:
                    df = pd.read_csv(file_path, encoding='utf-8', sep=';')
                except Exception:
                    return None

        df = standardize_columns(df)
        df['Code'] = stock_code
        result = select_stock(df, stock_name)
        if result:
            return result
    except Exception as e:
        # print(f"Error processing file {file_path}: {e}")
        pass
    return None

def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    name_map = {}
    if os.path.exists(NAME_MAP_FILE):
        for enc in ['utf-8', 'utf-8-sig', 'gbk']:
            for delim in [',', '\t', ';']:
                try:
                    df_names = pd.read_csv(NAME_MAP_FILE, encoding=enc, sep=delim, header=None, skiprows=1, names=['code', 'name'], dtype=str)
                    df_names['code'] = df_names['code'].str.zfill(6)
                    name_map = df_names.set_index('code')['name'].to_dict()
                    break
                except:
                    continue
    file_paths = glob.glob(os.path.join(DATA_DIR, '*.csv'))
    if not file_paths:
        print(f"错误：在 '{DATA_DIR}' 目录下未找到任何 CSV 文件。")
        return
    
    file_info_list = [(path, name_map) for path in file_paths]
    num_processes = min(cpu_count(), 8)
    print(f"启动并行处理，使用 {num_processes} 个进程处理 {len(file_paths)} 个文件。")
    
    with Pool(num_processes) as pool:
        raw_results = pool.map(process_file, file_info_list)
    
    selected_stocks = [r for r in raw_results if r is not None]
    
    if selected_stocks:
        results_df = pd.DataFrame(selected_stocks)
        results_df = results_df.sort_values(by=['涨跌幅'], ascending=False)
        
        current_time = datetime.now(TIMEZONE_SH)
        output_dir_name = current_time.strftime('%Y%m')
        os.makedirs(output_dir_name, exist_ok=True)
        timestamp = current_time.strftime('%Y%m%d%H%M%S')
        # 文件名前缀更改为“妖票起爆_RSI&MACD复合版V7”
        output_filename = f'妖票起爆_RSI&MACD复合版V7_{timestamp}.csv'
        output_path = os.path.join(output_dir_name, output_filename)
        
        final_cols = ['日期', '股票代码', '股票名称', '收盘价', '涨跌幅', '换手率%', '成交量比均量',
                      '整理期波动%', '突破前5日最大振幅%', '前5日实体波动%', '实体饱满度', '30日累积换手%',
                      'MACD', 'RSI', 'J值', f'MA{MA_SHORT}', f'MA{MA_MEDIUM}', '是否低于MA250', 'MA250偏离%']
        results_df = results_df.reindex(columns=final_cols)
        results_df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"\n选股完成！共选出 {len(selected_stocks)} 支符合RSI&MACD复合版V7条件的票")
        print(f"结果已保存：{output_path}")
    else:
        print("\n今日无票满足RSI&MACD复合版V7条件")

if __name__ == '__main__':
    if os.name == 'nt':
        freeze_support()
    main()
