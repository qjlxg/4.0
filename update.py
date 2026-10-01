import pandas as pd
import os
import time
import sys
import socket
from pytdx.hq import TdxHq_API
from pytdx.params import TDXParams

# --- 核心配置 ---
DATA_DIR = 'stock_data'
PROGRESS_DIR = 'results_data_update'
PROGRESS_FILE = os.path.join(PROGRESS_DIR, 'progress.txt')
STOCK_LIST_FILE = '列表.txt'
BATCH_SIZE = 1500

# 12 个核心 IP 阵列 (稳如泰山)
TDX_SERVERS = [
    ('101.227.73.20', 7709), ('101.227.77.254', 7709),
    ('122.192.35.44', 7709), ('221.231.141.60', 7709),
    ('59.173.18.140', 7709), ('14.215.128.18', 7709),
    ('60.28.23.80', 7709), ('218.60.29.136', 7709),
    ('119.147.212.81', 7709), ('183.60.224.178', 7709),
    ('115.238.90.165', 7709), ('123.125.108.14', 7709)
]

def get_best_server():
    best_ip, min_latency = None, float('inf')
    for ip, port in TDX_SERVERS:
        try:
            start = time.time()
            with socket.create_connection((ip, port), timeout=1) as conn:
                latency = time.time() - start
                if latency < min_latency:
                    min_latency, best_ip = latency, ip
        except: continue
    return best_ip

def is_st_stock(code, api):
    """通过行情接口判断是否为 ST 股票"""
    market = TDXParams.MARKET_SH if code.startswith('6') else TDXParams.MARKET_SZ
    try:
        quote = api.get_security_quotes([(market, code)])
        if quote:
            name = quote[0].get('name', '')
            return "ST" in name.upper()
    except: pass
    return False

def fetch_and_sync(code, api):
    # 1. 过滤主板：只保留 60 (沪市) 和 00 (深市)
    if not (code.startswith('60') or code.startswith('00')):
        return

    # 2. 排除 ST
    if is_st_stock(code, api):
        print(f"S({code})", end=' ', flush=True) # S 代表跳过 ST
        return

    market = TDXParams.MARKET_SH if code.startswith('6') else TDXParams.MARKET_SZ
    file_path = os.path.join(DATA_DIR, f"{code}.csv")
    
    last_date_str = "1990-01-01"
    if os.path.exists(file_path):
        try:
            existing_df = pd.read_csv(file_path)
            if not existing_df.empty:
                last_date_str = pd.to_datetime(existing_df['日期'].iloc[-1]).strftime('%Y-%m-%d')
        except: pass

    try:
        # 补齐断层，回溯 30 天
        bars = api.get_security_bars(9, market, code, 0, 30)
        if not bars: return
        
        f_info = api.get_finance_info(market, code)
        liutong = f_info.get('liutongguben', 0) if f_info else 0
        
        new_rows = []
        for i in range(len(bars)):
            curr = bars[i]
            p_close = bars[i-1]['close'] if i > 0 else curr['open']
            d_str = pd.to_datetime(curr['datetime']).strftime('%Y-%m-%d')
            
            if d_str <= last_date_str: continue

            row = {
                '日期': d_str, '股票代码': code,
                '开盘': float(curr['open']), '收盘': float(curr['close']),
                '最高': float(curr['high']), '最低': float(curr['low']),
                '成交量': int(curr['vol']), '成交额': float(curr['amount']),
                '振幅': round((curr['high'] - curr['low']) / p_close * 100, 2) if p_close != 0 else 0,
                '涨跌幅': round((curr['close'] - p_close) / p_close * 100, 2) if p_close != 0 else 0,
                '涨跌额': round(curr['close'] - p_close, 2),
                '换手率': round((curr['vol'] * 100 / liutong) * 100, 2) if liutong > 0 else 0.0
            }
            new_rows.append(row)
        
        if new_rows:
            df_append = pd.DataFrame(new_rows)
            df_append.to_csv(file_path, mode='a', index=False, header=not os.path.exists(file_path))
            print(f"√{code}", end=' ', flush=True)

    except: pass

def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(PROGRESS_DIR, exist_ok=True)

    # 1. 读列表并初步过滤沪深主板代码
    stock_df = pd.read_csv(STOCK_LIST_FILE, sep='\t')
    stock_df.columns = stock_df.columns.str.strip().str.lower()
    code_col = '代码' if '代码' in stock_df.columns else 'code'
    stock_df[code_col] = stock_df[code_col].astype(str).str.zfill(6)
    
    # 代码过滤逻辑：60开头(沪主板), 00开头(深主板)
    codes = [c for c in stock_df[code_col].tolist() if c.startswith('60') or c.startswith('00')]

    # 2. 进度控制
    start_idx = 0
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, 'r') as f:
            try: start_idx = int(f.read().strip())
            except: start_idx = 0

    if start_idx >= len(codes):
        print("\n主板非ST股票已全部补齐。"); 
        with open(PROGRESS_FILE, 'w') as f: f.write('0')
        return

    best_ip = get_best_server()
    if not best_ip: sys.exit(1)

    api = TdxHq_API(heartbeat=True, multithread=True)
    with api.connect(best_ip, 7709):
        end_idx = min(start_idx + BATCH_SIZE, len(codes))
        print(f"🚀 沪深主板(排除ST)同步中: {start_idx} -> {end_idx}")
        for code in codes[start_idx:end_idx]:
            fetch_and_sync(code, api)
    
    with open(PROGRESS_FILE, 'w') as f:
        f.write(str(end_idx))
    
    sys.exit(99 if end_idx < len(codes) else 0)

if __name__ == "__main__":
    main()
