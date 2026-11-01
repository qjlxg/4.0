import pandas as pd
import numpy as np
import glob
import os
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
import time

# --- 配置 ---
STOCK_DATA_DIR = 'stock_data'  # 股票数据目录
STOCK_NAMES_FILE = 'stock_names.csv' # 股票名称文件
OUTPUT_DIR = 'screened_results' # 输出结果目录
NUM_DAYS_LOOKBACK = 20 # 观察最近 N 个交易日的数据

# --- 核心技术指标计算 ---
def calculate_kdj(df, n=9, m1=3, m2=3):
    """计算 KDJ 指标"""
    df['收盘'] = pd.to_numeric(df['收盘'], errors='coerce')
    df['最高'] = pd.to_numeric(df['最高'], errors='coerce')
    df['最低'] = pd.to_numeric(df['最低'], errors='coerce')

    # 计算 RSV (未成熟随机值)
    low_list = df['最低'].rolling(window=n, min_periods=n).min()
    high_list = df['最高'].rolling(window=n, min_periods=n).max()
    
    # 避免除以零
    denominator = high_list - low_list
    denominator[denominator == 0] = 1e-6 
    
    df['RSV'] = (df['收盘'] - low_list) / denominator * 100
    
    # 计算 K、D、J
    df['K'] = df['RSV'].ewm(com=m1 - 1, adjust=False, min_periods=n).mean()
    df['D'] = df['K'].ewm(com=m2 - 1, adjust=False, min_periods=n).mean()
    df['J'] = 3 * df['K'] - 2 * df['D']
    
    return df

def calculate_indicators(df):
    """计算 MACD, KDJ, 均线和成交量指标 - 新增 Volume Ratio 计算"""
    # 确保有足够的数据计算 MA60 (60天)
    if len(df) < 60:
        return None

    # 统一数据类型
    df['收盘'] = pd.to_numeric(df['收盘'], errors='coerce')
    df['成交量'] = pd.to_numeric(df['成交量'], errors='coerce')
    df['最高'] = pd.to_numeric(df['最高'], errors='coerce')
    df['最低'] = pd.to_numeric(df['最低'], errors='coerce')

    # 1. MACD 计算
    ema_short = df['收盘'].ewm(span=12, adjust=False).mean()
    ema_long = df['收盘'].ewm(span=26, adjust=False).mean()
    df['DIFF'] = ema_short - ema_long
    df['DEA'] = df['DIFF'].ewm(span=9, adjust=False).mean()
    df['MACD'] = (df['DIFF'] - df['DEA']) * 2 

    # 2. KDJ 计算
    df = calculate_kdj(df)

    # 3. 均线 (MA)
    df['MA5'] = df['收盘'].rolling(window=5).mean()
    df['MA10'] = df['收盘'].rolling(window=10).mean()
    df['MA30'] = df['收盘'].rolling(window=30).mean()
    df['MA60'] = df['收盘'].rolling(window=60).mean()

    # 4. 成交量 (VOL) - 新增 VOL_MA5 和 Vol_Ratio
    df['VOL_MA5'] = df['成交量'].rolling(window=5).mean()
    df['VOL_MA10'] = df['成交量'].rolling(window=10).mean()
    # 新增: Vol_Ratio = 最新成交量 / 近5日均量
    df['Vol_Ratio'] = df['成交量'] / df['VOL_MA5']

    # 确保数据完整，去掉 NaN 行
    return df.dropna()

# --- 筛选逻辑 ---

def check_mode_1(df_recent, df_full):
    """
    模式一：底部反转启动型 (买入机会) - 优化：新增量能和MACD金叉严格确认
    """
    if len(df_recent) < NUM_DAYS_LOOKBACK:
        return False

    last = df_recent.iloc[-1]
    prev = df_recent.iloc[-2]

    # 1. MACD 零轴下方或附近首次金叉
    is_golden_cross = (last['DIFF'] > last['DEA'])
    is_recent_cross = (prev['DIFF'] <= prev['DEA'])
    is_low_macd = (last['DIFF'] < 0.1) 
    
    # **MACD动量过滤：必须是金叉**
    macd_is_confirmed = is_golden_cross and is_recent_cross and is_low_macd
    
    # **优化点 1: 要求 MACD 红柱极小 (启动初期)**
    is_early_stage = (last['MACD'] > 0) and (last['MACD'] < 0.05) 

    # 2. 地量蓄势 + 放量突破 (优化为：突破日放量确认)
    # **量能放大确认：最新成交量高于近5日均量1.25倍**
    is_volume_confirmed = last['Vol_Ratio'] > 1.25 

    # 3. 突破 (收盘价突破 MA30 且当日涨幅大于 5%)
    is_breakout = last['收盘'] > last['MA30'] and last['收盘'] > prev['收盘'] * 1.05 
    
    # 结合所有优化条件
    return macd_is_confirmed and is_early_stage and is_volume_confirmed and is_breakout

def check_mode_2(df_recent, df_full):
    """
    模式二：强势股整理再加速型 (买入机会) - 维持原有严格要求
    """
    if len(df_recent) < NUM_DAYS_LOOKBACK:
        return False

    last = df_recent.iloc[-1]
    prev_2 = df_recent.iloc[-3]

    # 1. MACD 零轴上方二次金叉或红柱再次放大
    is_strong_macd = (last['DIFF'] > 0.1 and last['DEA'] > 0.1)
    is_macd_reaccelerate = (last['MACD'] > df_recent.iloc[-2]['MACD'] * 1.1) and (df_recent.iloc[-2]['MACD'] < prev_2['MACD'])
    macd_signal = is_strong_macd and is_macd_reaccelerate

    # 2. 严格的多头排列
    is_bullish = last['MA10'] > last['MA30'] and last['MA30'] > last['MA60'] 

    # 优化点 2: 严格紧贴 10 日均线 (股价与 MA10 差距在 1%以内)
    is_tight_ma10 = (last['收盘'] > last['MA10']) and \
                    (last['收盘'] / last['MA10'] < 1.01)
    
    # 4. 量能确认 (当日收阳且放量) - 维持严格的 1.5x VOL_MA10 要求
    is_rebound = last['收盘'] > df_recent['收盘'].iloc[-2] 
    is_vol_confirm = last['成交量'] > last['VOL_MA10'] * 1.5

    return macd_signal and is_bullish and is_tight_ma10 and is_rebound and is_vol_confirm

def check_mode_3(df_recent, df_full):
    """
    模式三：高风险预警型 (提前跑路信号) - KDJ 高位死叉 + 跌破 MA10
    """
    if len(df_recent) < 10:
        return False
        
    last = df_recent.iloc[-1]
    prev = df_recent.iloc[-2]

    # 1. KDJ 死亡预警 (高位死叉)
    is_kdj_death_cross = (last['K'] < last['D']) and (prev['K'] >= prev['D'])
    is_high_kdj = (last['K'] > 80) or (last['D'] > 80) # 发生在超买区
    
    kdj_signal = is_kdj_death_cross and is_high_kdj

    # 2. 价格确认 (跌破短期生命线 MA10)
    is_price_breakdown = last['收盘'] < last['MA10'] 
    
    # 3. MACD 尚未死叉 (确保这是提前预警)
    is_macd_not_death = (last['DIFF'] > last['DEA'])

    return kdj_signal and is_price_breakdown and is_macd_not_death

# --- 工具函数：获取MACD信号文本 ---
def get_macd_signal_text(last, prev):
    if last['DIFF'] > last['DEA'] and prev['DIFF'] <= prev['DEA']:
        return 'Golden Cross'
    elif last['DIFF'] < last['DEA'] and prev['DIFF'] >= prev['DEA']:
        return 'Death Cross'
    else:
        return 'No Cross'

# --- 并行处理函数 ---
def process_stock_file(file_path):
    """处理单个股票文件，计算指标并筛选"""
    try:
        df = pd.read_csv(file_path, encoding='utf-8')
        df = df.sort_values(by='日期')

        df.rename(columns={
            '成交量': '成交量',
            '收盘': '收盘',
            '最高': '最高',
            '最低': '最低'
        }, inplace=True)

        df_calc = calculate_indicators(df)
        if df_calc is None:
            return None

        # 筛选只看最近 NUM_DAYS_LOOKBACK 天的数据
        df_recent = df_calc.tail(NUM_DAYS_LOOKBACK)
        if len(df_recent) < 2: # 至少需要两天来判断交叉和涨跌
             return None

        stock_code = os.path.basename(file_path).split('.')[0]
        
        last = df_recent.iloc[-1]
        prev = df_recent.iloc[-2]
        
        # --- 新增风险管理和指标输出字段 ---
        ma30_value = last['MA30']
        close_price = last['收盘']
        
        # 建议止损价：跌破 MA30 后再跌 2%
        stop_loss_price = ma30_value * 0.98 
        # 建议止盈价：短期目标盈利 5%
        take_profit_price = close_price * 1.05
        
        result = {
            'code': stock_code,
            'Close': f"{close_price:.2f}",
            'MA30_Value': f"{ma30_value:.2f}",
            'Vol_Ratio': f"{last['Vol_Ratio']:.2f}",
            'MACD_Signal': get_macd_signal_text(last, prev),
            'Stop_Loss': f"{stop_loss_price:.2f}",
            'Take_Profit': f"{take_profit_price:.2f}",
        }
        
        # 优先级：风险预警 > 买入机会
        if check_mode_3(df_recent, df_calc):
            result['mode'] = '模式三：高风险预警型 (提前跑路)'
            result['type'] = 'Warning'
        elif check_mode_1(df_recent, df_calc):
            result['mode'] = '模式一：底部反转启动型 (买入机会)'
            result['type'] = 'Buy'
        elif check_mode_2(df_recent, df_calc):
            result['mode'] = '模式二：强势股整理再加速型 (买入机会)'
            result['type'] = 'Buy'
        else:
            return None

        return result

    except Exception as e:
        # 实际运行中可以打印错误信息进行调试
        # print(f"处理文件 {file_path} 出错: {e}") 
        return None

# --- 主函数 ---
def main():
    start_time = time.time()
    
    all_files = glob.glob(os.path.join(STOCK_DATA_DIR, '*.csv'))
    
    if not all_files:
        print(f"未找到 {STOCK_DATA_DIR} 目录下的 CSV 文件。")
        return

    print(f"开始扫描 {len(all_files)} 个股票文件，使用并行处理...")
    results = []
    # 根据 CPU 核心数设置最大工作线程数
    max_workers = os.cpu_count() * 2 if os.cpu_count() else 4
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_stock_file, file) for file in all_files]
        for future in futures:
            result = future.result()
            if result:
                results.append(result)

    if not results:
        print("未筛选出符合条件的股票。")
        return

    # 3. 匹配股票名称 
    try:
        names_df = pd.read_csv(STOCK_NAMES_FILE, encoding='utf-8')
        names_df.rename(columns={'code': 'code', 'name': 'name'}, inplace=True)
        names_df['code'] = names_df['code'].astype(str).str.zfill(6)
        
        results_df = pd.DataFrame(results)
        results_df['code'] = results_df['code'].astype(str).str.zfill(6)
        
        final_df = pd.merge(results_df, names_df, on='code', how='left')
        final_df['name'] = final_df['name'].fillna('名称未知')

    except Exception as e:
        print(f"加载或匹配股票名称文件出错: {e}")
        final_df = pd.DataFrame(results)
        final_df['name'] = '名称未知'
        # 如果名称匹配失败，也要确保其他指标能输出
        for col in ['Close', 'MA30_Value', 'Vol_Ratio', 'MACD_Signal', 'Stop_Loss', 'Take_Profit']:
             if col not in final_df.columns:
                 final_df[col] = ''

    # 4. 输出到指定目录 
    now = datetime.now()
    timestamp = now.strftime('%Y%m%d_%H%M%S')
    year_month = now.strftime('%Y/%m')

    output_sub_dir = os.path.join(OUTPUT_DIR, year_month)
    os.makedirs(output_sub_dir, exist_ok=True)
    
    output_filename = f'screener_results_{timestamp}.csv'
    output_path = os.path.join(output_sub_dir, output_filename)
    
    # 最终输出列：包含名称、模式、指标和风险管理
    final_df = final_df[['code', 'name', 'mode', 'Close', 'MA30_Value', 'Vol_Ratio', 'MACD_Signal', 'Stop_Loss', 'Take_Profit']]
    final_df.to_csv(output_path, index=False, encoding='utf-8')
    
    end_time = time.time()
    duration = end_time - start_time
    
    print(f"\n筛选完成！总耗时: {duration:.2f} 秒。结果已保存到: {output_path}")

if __name__ == '__main__':
    main()