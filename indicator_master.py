import os
import glob
import pandas as pd
import numpy as np
from multiprocessing import Pool, cpu_count

# 配置路径
# 适配多种可能的路径：screened_results/2025/12/ 或根目录下 2025/12 开头的文件
SCREENED_DIR = "screened_results"
STOCK_DATA_DIR = "stock_data"
NAMES_FILE = "stock_names.csv"
OUTPUT_FILE = "indicator_report.csv"

def calculate_rsi(series, period=6):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_indicators(file_info):
    file_path, code = file_info
    try:
        df = pd.read_csv(file_path)
        if len(df) < 60: return None
        
        # 统一列名引用 (根据上传样本)
        # 列名: 日期, 股票代码, 开盘, 收盘, 最高, 最低, 成交量, 成交额, 振幅, 涨跌幅, 涨跌额, 换手率
        latest = df.iloc[-1]
        
        # 1. 30日均换手
        avg_turnover_30 = df['换手率'].tail(30).mean()
        
        # 2. 今日量比 (今日成交量 / 前5日平均成交量)
        vol_5 = df['成交量'].iloc[-6:-1].mean()
        vol_ratio = latest['成交量'] / vol_5 if vol_5 > 0 else 0
        
        # 3. RSI6
        rsi_series = calculate_rsi(df['收盘'], 6)
        rsi6 = rsi_series.iloc[-1]

        # 4. KDJ-K值 (周期9)
        low_9 = df['最低'].rolling(window=9).min()
        high_9 = df['最高'].rolling(window=9).max()
        rsv = (df['收盘'] - low_9) / (high_9 - low_9) * 100
        k_value = rsv.ewm(com=2, adjust=False).mean().iloc[-1]

        # 5. 距60日线空间
        ma60 = df['收盘'].rolling(window=60).mean().iloc[-1]
        space_60 = ((latest['收盘'] - ma60) / ma60) * 100

        # 6. 状态
        status = "涨" if latest['涨跌额'] > 0 else "跌"

        return {
            "code": str(code).zfill(6),
            "当时价格": latest['收盘'],
            "涨跌": status,
            "30日均换手": round(avg_turnover_30, 2),
            "今日量比": round(vol_ratio, 2),
            "RSI6": round(rsi6, 2),
            "K值": round(k_value, 2),
            "距60日线空间%": round(space_60, 2)
        }
    except Exception as e:
        return None

def main():
    # 1. 搜寻目标代码文件 (兼容 screened 和 screener 拼写)
    search_patterns = [
        os.path.join(SCREENED_DIR, "2025", "12", "*.csv"),
        os.path.join(SCREENED_DIR, "screener_results_202512*.csv"),
        "screener_results_202512*.csv"
    ]
    
    target_files = []
    for pattern in search_patterns:
        target_files.extend(glob.glob(pattern))
    
    if not target_files:
        print("未找到筛选结果文件，请检查目录结构。")
        return

    target_codes = set()
    for f in target_files:
        try:
            tmp_df = pd.read_csv(f)
            if 'code' in tmp_df.columns:
                target_codes.update(tmp_df['code'].astype(str).str.zfill(6).tolist())
        except:
            continue
    
    if not target_codes:
        print("筛选文件中没有有效的股票代码。")
        return

    # 2. 匹配 stock_data 中的 CSV
    data_tasks = []
    for code in target_codes:
        csv_path = os.path.join(STOCK_DATA_DIR, f"{code}.csv")
        if os.path.exists(csv_path):
            data_tasks.append((csv_path, code))
    
    if not data_tasks:
        print(f"在 {STOCK_DATA_DIR} 目录下未匹配到任何历史数据文件。")
        return

    # 3. 并行处理
    print(f"开始并行处理 {len(data_tasks)} 个文件...")
    with Pool(cpu_count()) as p:
        results = p.map(calculate_indicators, data_tasks)
    
    results = [r for r in results if r is not None]
    
    if not results:
        print("没有产生任何有效统计数据。")
        return

    # 4. 合并名称并保存
    res_df = pd.DataFrame(results)
    if os.path.exists(NAMES_FILE):
        names_df = pd.read_csv(NAMES_FILE, dtype={'code': str})
        names_df['code'] = names_df['code'].str.zfill(6)
        res_df = pd.merge(res_df, names_df[['code', 'name']], on='code', how='left')
    
    # 调整列顺序
    cols = ['code', 'name', '当时价格', '涨跌', '30日均换手', '今日量比', 'RSI6', 'K值', '距60日线空间%']
    res_df = res_df[[c for c in cols if c in res_df.columns]]
    
    res_df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')
    print(f"成功！分析结果已保存至 {OUTPUT_FILE}，共 {len(res_df)} 行。")

if __name__ == "__main__":
    main()
