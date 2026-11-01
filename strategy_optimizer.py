import os
import pandas as pd
import numpy as np
import glob
from multiprocessing import Pool, cpu_count

# 配置
STOCK_DATA_DIR = "stock_data"
OUTPUT_FILE = "strategy_insight_report.csv"
SUMMARY_FILE = "insight_summary.txt"

def analyze_stock_vectorized(file_path):
    try:
        # 1. 读取并基础清洗
        df = pd.read_csv(file_path, usecols=['日期', '股票代码', '收盘', '最高', '最低', '成交量', '换手率'])
        # 剔除股价异常或数据太少的票
        df = df[df['收盘'] > 0.1].copy()
        if len(df) < 70: return None
        
        # 2. 指标计算
        delta = df['收盘'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(6).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(6).mean()
        df['RSI6'] = 100 - (100 / (1 + (gain / loss.replace(0, np.nan))))
        df['RSI6'] = df['RSI6'].fillna(50)
        
        low_9 = df['最低'].rolling(9).min()
        high_9 = df['最高'].rolling(9).max()
        rsv = (df['收盘'] - low_9) / (high_9 - low_9).replace(0, np.nan) * 100
        df['K'] = rsv.ewm(com=2, adjust=False).mean()
        
        df['MA60'] = df['收盘'].rolling(60).mean()
        df['Space60'] = (df['收盘'] - df['MA60']) / df['MA60'].replace(0, np.nan) * 100
        df['Turnover30'] = df['换手率'].rolling(30).mean()
        df['VolRatio'] = df['成交量'] / df['成交量'].shift(1).rolling(5).mean().replace(0, np.nan)
        
        # 3. 寻找起涨点
        # 未来3天内最高价对比当前收盘价涨幅 >= 8%
        future_3d_max = df['最高'].shift(-3).rolling(3, min_periods=1).max()
        df['Growth'] = (future_3d_max - df['收盘']) / df['收盘'].replace(0, np.nan)
        
        # 4. 筛选逻辑修复
        # 过滤掉距60日线空间异常的脏数据 (例如 > 200% 的通常是除权未复权)
        mask = (df['Growth'] >= 0.08) & \
               (df.index < len(df) - 10) & \
               (df.index > 60) & \
               (df['Space60'].abs() < 100) 
        
        df['is_start'] = mask
        # 消除 FutureWarning: 显式转换 bool 类型
        df['is_start'] = df['is_start'] & ~(df['is_start'].shift(1).fillna(False).astype(bool))
        
        success_moments = df[df['is_start']].copy()
        if success_moments.empty: return None
            
        success_moments['Next5D_Max'] = (df['最高'].shift(-5).rolling(5, min_periods=1).max() - success_moments['收盘']) / success_moments['收盘'] * 100
        
        return success_moments[['股票代码', '日期', 'RSI6', 'K', 'Space60', 'Turnover30', 'VolRatio', 'Next5D_Max']].to_dict('records')
    except:
        return None

def main():
    files = glob.glob(os.path.join(STOCK_DATA_DIR, "*.csv"))
    if not files:
        print("未找到数据文件")
        return
    
    print(f"🚀 开始稳健性分析，处理 {len(files)} 个文件...")
    with Pool(cpu_count()) as p:
        all_results = p.map(analyze_stock_vectorized, files, chunksize=25)
    
    flat_results = [item for sublist in all_results if sublist for item in sublist]
    report_df = pd.DataFrame(flat_results)
    
    if not report_df.empty:
        # 数据量控制：保留最近 50 万条
        if len(report_df) > 500000:
            report_df = report_df.sort_values('日期').tail(500000)
            
        report_df.columns = ['code', 'date', 'RSI6_pre', 'K_pre', 'Space60_pre', 'Turnover30_pre', 'VolRatio_pre', 'Next5D_MaxPct']
        report_df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')

        # 5. 生成修正后的摘要 (使用中位数防止脏数据干扰)
        with open(SUMMARY_FILE, 'w', encoding='utf-8') as f:
            f.write("--- 修正版：起涨特征统计摘要 ---\n")
            f.write(f"有效样本总数: {len(report_df)}\n")
            f.write(f"RSI6 核心区间: {report_df['RSI6_pre'].quantile(0.25):.2f} - {report_df['RSI6_pre'].quantile(0.75):.2f}\n")
            f.write(f"K值 核心区间: {report_df['K_pre'].quantile(0.25):.2f} - {report_df['K_pre'].quantile(0.75):.2f}\n")
            f.write(f"量比 中位数: {report_df['VolRatio_pre'].median():.2f}\n")
            f.write(f"距60日线空间 中位数: {report_df['Space60_pre'].median():.2f}%\n")
            f.write("\n提示：若空间中位数恢复正常（如 -5% 到 10%），说明脏数据已剔除。\n")
        
        print(f"✅ 分析完成。摘要已更新至 {SUMMARY_FILE}")

if __name__ == "__main__":
    main()
