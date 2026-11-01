import pandas as pd
import numpy as np
import os
import glob
from multiprocessing import Pool, cpu_count
from datetime import datetime, timedelta

# --- 路径配置 ---
RESULTS_DIR = 'results/2025/12'
DATA_DIR = 'stock_data'
NAMES_FILE = 'stock_names.csv'
DETAIL_REPORT = '股票信号追踪明细.csv'
SUMMARY_REPORT = '策略逻辑胜率看板.csv'

def process_single_stock(args):
    code, sig_date, sig_price, sig_rsi, sig_k, sig_vol = args
    stock_code_str = str(code).zfill(6)
    file_path = os.path.join(DATA_DIR, f"{stock_code_str}.csv")
    if not os.path.exists(file_path): return None
    
    try:
        df = pd.read_csv(file_path)
        df['日期'] = pd.to_datetime(df['日期'])
        sig_dt = pd.to_datetime(sig_date)
        
        # 提取信号日之后的数据
        after_sig = df[df['日期'] >= sig_dt].copy()
        if after_sig.empty: return None
        
        # 1. 计算3天表现 (信号日+2个交易日)
        price_3d = after_sig.iloc[min(2, len(after_sig)-1)]['收盘']
        win_3d = 1 if price_3d > sig_price else 0
        
        # 2. 计算7天表现 (信号日+6个交易日)
        price_7d = after_sig.iloc[min(6, len(after_sig)-1)]['收盘']
        win_7d = 1 if price_7d > sig_price else 0
        
        # 3. 计算至今表现
        latest_price = after_sig.iloc[-1]['收盘']
        total_change = (latest_price - sig_price) / sig_price * 100
        
        # --- 翻译逻辑名称 ---
        conds = []
        if sig_rsi < 35: conds.append("超跌")
        if sig_k < 30: conds.append("筑底")
        if sig_vol > 1.8: conds.append("放量")
        if sig_vol < 0.6: conds.append("缩量")
        logic_name = "+".join(conds) if conds else "常规"
        
        return {
            "代码": stock_code_str,
            "信号日期": sig_date,
            "选股逻辑": logic_name,
            "初始价": sig_price,
            "当前价": latest_price,
            "累计涨跌%": round(total_change, 2),
            "3天胜": win_3d,
            "7天胜": win_7d,
            "至今胜": 1 if total_change > 0 else 0
        }
    except: return None

def main():
    # 1. 加载名称
    name_map = {}
    if os.path.exists(NAMES_FILE):
        n_df = pd.read_csv(NAMES_FILE, dtype={'code': str})
        name_map = dict(zip(n_df['code'].str.zfill(6), n_df['name']))

    # 2. 读取任务
    tasks = []
    for f in glob.glob(os.path.join(RESULTS_DIR, "轮动低吸_本地数据_*.csv")):
        tmp = pd.read_csv(f, dtype={'代码': str})
        for _, row in tmp.iterrows():
            tasks.append((row['代码'], row['最新日期'], float(row['现价']), 
                          float(row['RSI6']), float(row['K值']), float(row['今日量比'])))

    # 3. 计算
    with Pool(cpu_count()) as pool:
        results = [r for r in pool.map(process_single_stock, tasks) if r is not None]

    df = pd.DataFrame(results)
    df.insert(1, '名称', df['代码'].map(name_map).fillna("未知"))

    # 4. 生成多维度汇总看板
    summary = df.groupby('选股逻辑').agg({
        '代码': 'count',
        '3天胜': 'sum',
        '7天胜': 'sum',
        '至今胜': 'sum',
        '累计涨跌%': 'mean'
    }).rename(columns={'代码': '总样本'})

    summary['3天胜率%'] = round(summary['3天胜'] / summary['总样本'] * 100, 1)
    summary['7天胜率%'] = round(summary['7天胜'] / summary['总样本'] * 100, 1)
    summary['至今胜率%'] = round(summary['至今胜'] / summary['总样本'] * 100, 1)
    summary['平均收益%'] = round(summary['累计涨跌%'], 2)

    # 只保留核心汇总列
    sum_cols = ['总样本', '3天胜率%', '7天胜率%', '至今胜率%', '平均收益%']
    summary[sum_cols].sort_values(by='7天胜率%', ascending=False).to_csv(SUMMARY_REPORT, encoding='utf_8_sig')
    
    # 保存明细
    df.to_csv(DETAIL_REPORT, index=False, encoding='utf_8_sig')
    print("对比报告已生成！")

if __name__ == "__main__":
    main()
