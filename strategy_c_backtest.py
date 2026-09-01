import pandas as pd
import numpy as np
import os
import glob
from datetime import datetime
from multiprocessing import Pool, cpu_count

# --- 配置 ---
DATA_DIR = "stock_data"
NAME_MAP_FILE = 'stock_names.csv'
LOOKBACK_WINDOW = 120  # 回测过去120天的数据
HOLD_DAYS = 10         # 追踪信号发出后10天的表现

def calculate_indicators(df):
    close = df['Close']
    df['MA5'] = close.rolling(5).mean()
    df['MA20'] = close.rolling(20).mean()
    df['MA5V'] = df['Volume'].rolling(5).mean()
    df['MA3V'] = df['Volume'].rolling(3).mean()
    # RSI6
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(6).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(6).mean()
    df['RSI6'] = 100 - (100 / (1 + gain/loss))
    # KDJ
    low_9 = df['Low'].rolling(9).min()
    high_9 = df['High'].rolling(9).max()
    rsv = (close - low_9) / (high_9 - low_9) * 100
    df['K'] = rsv.ewm(com=2).mean()
    df['D'] = df['K'].ewm(com=2).mean()
    # MACD
    exp1 = close.ewm(span=12, adjust=False).mean()
    exp2 = close.ewm(span=26, adjust=False).mean()
    df['DIF'] = exp1 - exp2
    df['DEA'] = df['DIF'].ewm(span=9, adjust=False).mean()
    df['DIF_MA60'] = df['DIF'].rolling(60).mean()
    df['OBV'] = (np.sign(close.diff()) * df['Volume']).fillna(0).cumsum()
    return df

def run_backtest_on_file(file_path):
    code = os.path.basename(file_path).split('.')[0]
    if not (code.startswith('60') or code.startswith('00')): return None
    
    try:
        df = pd.read_csv(file_path)
        if len(df) < 100: return None
        df.columns = df.columns.str.strip()
        df.rename(columns={'日期':'Date','收盘':'Close','开盘':'Open','最高':'High','最低':'Low','成交量':'Volume'}, inplace=True)
        df = calculate_indicators(df)
        
        results = []
        # 在回测窗口内滑动寻找信号
        start_idx = len(df) - LOOKBACK_WINDOW
        if start_idx < 65: start_idx = 65
        
        for i in range(start_idx, len(df) - 1):
            curr = df.iloc[i]
            prev = df.iloc[i-1]
            
            # C 策略核心逻辑
            ma5_slope = np.polyfit(np.arange(5), df['MA5'].iloc[i-4:i+1].values, 1)[0]
            is_trend = (ma5_slope > 0) and (curr['Close'] > curr['MA20'])
            prev_high_40 = df['High'].iloc[i-40:i].max()
            is_breakout = (curr['Close'] > prev_high_40 * 1.01)
            is_vol = (2.0 * curr['MA5V'] < curr['Volume'] < 4.0 * curr['MA5V']) and (curr['MA3V'] >= curr['MA5V'])
            is_rsi = (curr['RSI6'] > 60) and (curr['RSI6'] > prev['RSI6'])
            is_kdj = (curr['K'] > curr['D']) and (prev['K'] <= prev['D']) and (curr['K'] < 70)
            is_macd = (curr['DIF'] > curr['DEA']) and (curr['DIF'] > -0.05) and (curr['DIF'] > curr['DIF_MA60'])

            if is_trend and is_breakout and is_vol and is_rsi and is_kdj and is_macd:
                # 信号触发，计算后续 HOLD_DAYS 表现
                post_df = df.iloc[i+1 : i+1+HOLD_DAYS]
                if post_df.empty: continue
                
                max_reach = ((post_df['High'].max() - curr['Close']) / curr['Close']) * 100
                final_ret = ((post_df['Close'].iloc[-1] - curr['Close']) / curr['Close']) * 100
                
                results.append({
                    "代码": code, "触发日期": curr['Date'], "触发价": curr['Close'],
                    "T+1开盘涨幅": round(((post_df['Open'].iloc[0] - curr['Close']) / curr['Close']) * 100, 2),
                    "最高冲击%": round(max_reach, 2), "持有10日收益%": round(final_ret, 2)
                })
        return results
    except: return None

def main():
    names_df = pd.read_csv(NAME_MAP_FILE, dtype={'code': str})
    names_dict = dict(zip(names_df['code'].str.zfill(6), names_df['name']))
    files = glob.glob(os.path.join(DATA_DIR, "*.csv"))
    
    with Pool(cpu_count()) as pool:
        all_results = pool.map(run_backtest_on_file, files)
    
    flattened = [item for sublist in all_results if sublist for item in sublist]
    if not flattened:
        print("回测期间未发现信号。")
        return

    res_df = pd.DataFrame(flattened)
    res_df['名称'] = res_df['代码'].apply(lambda x: names_dict.get(x, "未知"))
    
    # 按照月份存入仓库
    now = datetime.now()
    dir_name = "backtest_reports/" + now.strftime("%Y-%m")
    os.makedirs(dir_name, exist_ok=True)
    save_path = os.path.join(dir_name, f"C_Strategy_Backtest_{now.strftime('%Y%m%d_%H%M')}.csv")
    res_df.to_csv(save_path, index=False, encoding='utf-8-sig')
    
    print(f"回测完成！信号总数: {len(res_df)}")
    print(f"平均最高冲击: {res_df['最高冲击%'].mean():.2f}%")
    print(f"T+1 均价入场难度(开盘涨幅): {res_df['T+1开盘涨幅'].mean():.2f}%")

if __name__ == "__main__":
    main()
