import pandas as pd
import numpy as np
import os
import glob
from datetime import datetime
from multiprocessing import Pool, cpu_count

# --- 核心配置 ---
DATA_DIR = "stock_data"
NAME_MAP_FILE = 'stock_names.csv'

def calculate_indicators(df):
    """手写核心指标，确保在 GitHub Actions 环境中 100% 运行成功"""
    close = df['Close']
    high = df['High']
    low = df['Low']
    vol = df['Volume']

    # 1. 均线系统
    df['MA5'] = close.rolling(5).mean()
    df['MA20'] = close.rolling(20).mean()
    df['MA5V'] = vol.rolling(5).mean()
    df['MA3V'] = vol.rolling(3).mean()

    # 2. RSI6 (判断强弱与超买)
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(6).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(6).mean()
    df['RSI6'] = 100 - (100 / (1 + gain/loss))

    # 3. KDJ (9,3,3)
    low_9 = low.rolling(9).min()
    high_9 = high.rolling(9).max()
    rsv = (close - low_9) / (high_9 - low_9) * 100
    df['K'] = rsv.ewm(com=2, adjust=False).mean()
    df['D'] = df['K'].ewm(com=2, adjust=False).mean()
    
    # 4. MACD (12,26,9)
    exp1 = close.ewm(span=12, adjust=False).mean()
    exp2 = close.ewm(span=26, adjust=False).mean()
    df['DIF'] = exp1 - exp2
    df['DEA'] = df['DIF'].ewm(span=9, adjust=False).mean()
    df['DIF_MA60'] = df['DIF'].rolling(60).mean()

    # 5. OBV (能量潮)
    df['OBV'] = (np.sign(close.diff()) * vol).fillna(0).cumsum()
    return df

def is_strategy_c_pro(df):
    """C策略 Pro版：基于 195 个样本统计优化后的实战逻辑"""
    if len(df) < 65: return False
    
    df = calculate_indicators(df)
    latest = df.iloc[-1]
    prev = df.iloc[-2]

    # --- [优化 A: 资金与价格过滤] ---
    # 1. 价格区间：5.0 - 20.0 元
    if not (5.0 <= latest['Close'] <= 20.0): return False
    # 2. 流动性过滤：成交额必须 > 8000万 (避开僵尸股)
    if latest['Amount'] < 80000000: return False

    # --- [优化 B: 风险与位置控制] ---
    # 3. 乖离率控制：股价距离 MA20 不能超过 12% (防止买在短线顶点)
    bias_20 = (latest['Close'] - latest['MA20']) / latest['MA20']
    if bias_20 > 0.12: return False
    # 4. RSI黄金区间：60-80 (超过 80 视为过度超买，极易次日低开)
    if not (60 < latest['RSI6'] < 80): return False

    # --- [优化 C: 趋势与突破确认] ---
    # 5. 趋势向上：MA5 斜率 > 0 且 站在 MA20 之上
    ma5_tail = df['MA5'].tail(5).values
    slope = np.polyfit(np.arange(5), ma5_tail, 1)[0]
    if slope <= 0 or latest['Close'] <= latest['MA20']: return False

    # 6. 40日平台突破：收盘价站上过去40日最高点 1% 以上
    prev_high_40 = df['High'].iloc[-41:-1].max()
    if latest['Close'] <= prev_high_40 * 1.01: return False

    # --- [优化 D: 量价共振确认] ---
    # 7. 量能健康：2.0 - 5.0 倍放量，且 OBV 向上
    if not (2.0 * latest['MA5V'] < latest['Volume'] < 5.0 * latest['MA5V']): return False
    if latest['OBV'] <= prev['OBV']: return False

    # 8. 指标金叉共振
    is_kdj_ok = (latest['K'] > latest['D']) and (prev['K'] <= prev['D']) and (latest['K'] < 70)
    is_macd_ok = (latest['DIF'] > latest['DEA']) and (latest['DIF'] > -0.05) and (latest['DIF'] > latest['DIF_MA60'])

    return is_kdj_ok and is_macd_ok

def process_file(file_path):
    # 只处理 60 和 00 开头的股票
    code = os.path.basename(file_path).split('.')[0]
    if not (code.startswith('60') or code.startswith('00')): return None
    
    try:
        df = pd.read_csv(file_path)
        if len(df) < 65: return None
        df.columns = df.columns.str.strip()
        # 兼容处理列名
        df.rename(columns={'成交额':'Amount','收盘':'Close','开盘':'Open','最高':'High','最低':'Low','成交量':'Volume'}, inplace=True)
        
        if is_strategy_c_pro(df):
            return {
                "代码": code, 
                "价格": round(df.iloc[-1]['Close'], 2), 
                "成交额(万)": round(df.iloc[-1]['Amount']/10000, 0),
                "RSI6": round(df.iloc[-1]['RSI6'], 1)
            }
    except Exception as e:
        return None

def main():
    # 匹配股票名称
    names_df = pd.read_csv(NAME_MAP_FILE, dtype={'code': str})
    names_dict = dict(zip(names_df['code'].str.zfill(6), names_df['name']))

    files = glob.glob(os.path.join(DATA_DIR, "*.csv"))
    print(f"🚀 启动 C 策略 Pro 深度扫描: {len(files)} 只个股...")

    # 并行处理
    with Pool(cpu_count()) as pool:
        results = pool.map(process_file, files)

    found = [r for r in results if r is not None]
    
    # 二次过滤 ST 和 退市
    final_list = []
    if found:
        for item in found:
            name = names_dict.get(item['代码'], "未知")
            if "ST" in name or "退" in name: continue
            item['名称'] = name
            final_list.append(item)
            
    if final_list:
        df_res = pd.DataFrame(final_list)
        now = datetime.now()
        dir_name = now.strftime("%Y-%m")
        os.makedirs(dir_name, exist_ok=True)
        
        filename = os.path.join(dir_name, f"C_Sniper_Pro_{now.strftime('%Y%m%d_%H%M%S')}.csv")
        df_res.to_csv(filename, index=False, encoding='utf-8-sig')
        print(f"✅ 筛选完成! 发现 {len(df_res)} 只满足黄金区间的个股，结果已存档至: {filename}")
    else:
        print("📭 今日未发现符合“黄金共振”条件的个股。")

if __name__ == "__main__":
    main()
