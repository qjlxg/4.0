import pandas as pd
import os
import glob
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

# --- 配置参数 ---
DATA_DIR = 'stock_data'
NAMES_FILE = 'stock_names.csv'
OUTPUT_ROOT = 'scan_results'

# 均线周期 (新增 MA10)
MA_SHORT = 5
MA_TEN = 10 
MA_MEDIUM = 20  # 核心均线
MA_LONG = 60
ANALYSIS_DAYS = 120
VOLUME_RATIO = 1.5

# --- 排除和形态配置 ---
UPPER_LIMIT_DAYS = 10  # 10日内有涨停
MAIN_BOARD_PREFIXES = ('60', '00') # 沪深主板
TOTAL_CONDITIONS = 6 # 总条件数更新为 6 个

def calculate_indicators(df):
    """计算所需的移动平均线和涨跌幅"""
    df[f'MA{MA_SHORT}'] = df['Close'].rolling(window=MA_SHORT).mean()
    df[f'MA{MA_TEN}'] = df['Close'].rolling(window=MA_TEN).mean() # 新增 MA10
    df[f'MA{MA_MEDIUM}'] = df['Close'].rolling(window=MA_MEDIUM).mean()
    df[f'MA{MA_LONG}'] = df['Close'].rolling(window=MA_LONG).mean()
    df['VOL_MA5'] = df['Volume'].rolling(window=5).mean()
    
    # 确保涨跌幅列被映射或计算
    if 'Daily_Change_Pct' not in df.columns:
        if '涨跌幅' in df.columns:
            df['Daily_Change_Pct'] = df['涨跌幅']
        else:
            df['Daily_Change_Pct'] = df['Close'].pct_change() * 100
        
    return df

def scan_stock(file_path, stock_names):
    """
    单个股票的筛选逻辑。
    检查 6 个技术条件 + 2 个排除条件。
    """
    stock_code = os.path.basename(file_path).split('.')[0]
    stock_name = stock_names.get(stock_code, 'N/A')
    
    # 1. 排除条件
    if not stock_code.startswith(MAIN_BOARD_PREFIXES):
        return None
    if 'ST' in stock_name or '*ST' in stock_name:
        return None

    try:
        # 读取数据并标准化列名
        df = pd.read_csv(file_path, parse_dates=['日期'], encoding='utf-8')
        df.rename(columns={'日期': 'Date', '收盘': 'Close', '成交量': 'Volume'}, inplace=True)
        # 处理涨跌幅列名
        if '涨跌幅' in df.columns:
            df.rename(columns={'涨跌幅': 'Daily_Change_Pct'}, inplace=True)
            
        df.sort_values(by='Date', inplace=True)
        df.set_index('Date', inplace=True)

        if len(df) < ANALYSIS_DAYS:
            return None
            
        df = calculate_indicators(df)
        
        df_analysis = df.tail(max(ANALYSIS_DAYS, 10)).copy()
        
        if len(df_analysis) < 3:
             return None

        latest = df_analysis.iloc[-1]
        previous_1 = df_analysis.iloc[-2]
        previous_2 = df_analysis.iloc[-3]
        
        # ----------------------------------------------------
        # 定义并检查 6 个条件 (OR 逻辑)
        matched_conditions = []

        # 1. 多头排列条件 (MA5 > MA20 > MA60)
        ma_condition_met = (latest[f'MA{MA_SHORT}'] > latest[f'MA{MA_MEDIUM}']) and \
                           (latest[f'MA{MA_MEDIUM}'] > latest[f'MA{MA_LONG}'])
        if ma_condition_met:
            matched_conditions.append('趋势: 多头排列 (5>20>60)')

        # 2. 放量上涨条件 (当日成交量 > 5日均量 * VOLUME_RATIO)
        volume_ratio = latest['Volume'] / latest['VOL_MA5'] if latest['VOL_MA5'] > 0 else 0
        volume_condition_met = volume_ratio > VOLUME_RATIO
        if volume_condition_met:
            matched_conditions.append(f'量能: 放量 ({volume_ratio:.2f}X)')

        # 3. 10日内有涨停 (日涨跌幅 >= 9.9%)
        recent_data = df_analysis.tail(UPPER_LIMIT_DAYS)
        has_upper_limit = (recent_data['Daily_Change_Pct'] >= 9.9).any()
        if has_upper_limit:
            matched_conditions.append(f'形态: {UPPER_LIMIT_DAYS}日内有涨停')

        # 4. 两连阴 (连续两天日涨跌幅 < 0)
        is_previous_1_negative = previous_1['Daily_Change_Pct'] < 0
        is_previous_2_negative = previous_2['Daily_Change_Pct'] < 0
        
        two_negative_days = is_previous_1_negative and is_previous_2_negative
        if two_negative_days:
            p1_pct = previous_1['Daily_Change_Pct']
            p2_pct = previous_2['Daily_Change_Pct']
            matched_conditions.append(f'形态: 两连阴 ({p2_pct:.2f}%, {p1_pct:.2f}%)')
        
        # 5. 20日均线上 (收盘价 > MA20)
        ma20_up_condition = latest['Close'] > latest[f'MA{MA_MEDIUM}']
        if ma20_up_condition:
            matched_conditions.append('趋势: 高于MA20')
            
        # 6. 新增条件：突破转强 (一根阳线上穿五日十日线)
        # 确保 MA10 有效
        if not pd.isna(latest[f'MA{MA_TEN}']):
            breakout_condition = (latest['Close'] > latest[f'MA{MA_SHORT}']) and \
                                 (latest['Close'] > latest[f'MA{MA_TEN}']) and \
                                 (latest['Daily_Change_Pct'] > 0)
            if breakout_condition:
                matched_conditions.append('动量: 突破转强 (阳线上穿5/10)')
            
        # ----------------------------------------------------

        if matched_conditions:
            return {
                'Code': stock_code,
                'Name': stock_name, 
                'Latest_Date': latest.name.strftime('%Y-%m-%d'),
                'Latest_Close': latest['Close'],
                'Daily_Change_Pct': f"{latest['Daily_Change_Pct']:.2f}%",
                'MA20': f"{latest[f'MA{MA_MEDIUM}']:.2f}",
                'MA60': f"{latest[f'MA{MA_LONG}']:.2f}",
                'Volume_Ratio': f"{volume_ratio:.2f}X",
                'Matched_Conditions': ' | '.join(matched_conditions)
            }
        
        return None

    except Exception as e:
        print(f"Error scanning {file_path} (Code: {stock_code}): {e}")
        return None

def main():
    """主函数：并行扫描，合并结果，保存输出（同时输出 OR 逻辑和 AND 逻辑的结果）"""
    print("Starting comprehensive pattern finder...")
    
    # 1. 加载股票名称字典
    try:
        names_df = pd.read_csv(NAMES_FILE, dtype={'code': str}, encoding='utf-8-sig')
        stock_names = names_df.set_index('code')['name'].to_dict()
    except Exception as e:
        print(f"Error loading stock names: {e}. Excluding stock names from results.")
        stock_names = {}
    
    # 2. 获取所有数据文件列表
    data_files = glob.glob(os.path.join(DATA_DIR, '*.csv'))
    if not data_files:
        print(f"No CSV files found in {DATA_DIR}. Exiting.")
        return

    # 3. 并行处理 (运行 OR 逻辑扫描)
    results = []
    MAX_WORKERS = os.cpu_count() * 2 if os.cpu_count() else 4
    print(f"Using {MAX_WORKERS} workers to scan {len(data_files)} files...")
    
    scan_func = lambda f: scan_stock(f, stock_names)
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        results_iter = executor.map(scan_func, data_files)
        
        for result in results_iter:
            if result:
                results.append(result)

    if not results:
        print("No stocks matched any bullish criteria after exclusion filters.")
        return
        
    # 4. 整理 OR 逻辑结果
    results_df_or = pd.DataFrame(results)
    
    # 调整 OR 结果的列顺序
    results_df_or = results_df_or[['Code', 'Name', 'Latest_Date', 'Latest_Close', 'Daily_Change_Pct',
                                   'Matched_Conditions', 'MA20', 'MA60', 'Volume_Ratio']]

    # 5. 生成 AND 逻辑结果 (现在是 6 个条件)
    
    # 检查是否同时满足所有 6 个条件的函数
    def check_all_conditions(conditions_str):
        # 6 个条件检查的关键词
        is_ma_met = '多头排列' in conditions_str
        is_volume_met = '放量' in conditions_str
        is_limit_met = f'{UPPER_LIMIT_DAYS}日内有涨停' in conditions_str
        is_negative_met = '两连阴' in conditions_str
        is_ma20_met = '高于MA20' in conditions_str
        is_breakout_met = '突破转强' in conditions_str # 新增：突破转强
        
        return is_ma_met and is_volume_met and is_limit_met and is_negative_met and is_ma20_met and is_breakout_met

    # 筛选出满足所有 6 个条件的行
    results_df_and = results_df_or[results_df_or['Matched_Conditions'].apply(check_all_conditions)].copy()

    # 6. 确定输出路径和文件名
    now = datetime.now()
    output_dir = os.path.join(OUTPUT_ROOT, now.strftime('%Y-%m'))
    timestamp_str = now.strftime('%Y%m%d_%H%M%S')
    os.makedirs(output_dir, exist_ok=True)
    
    # 7. 保存 OR 逻辑结果
    output_filename_or = f'bullish_scan_OR_logic_COMPREHENSIVE_{timestamp_str}.csv' 
    output_path_or = os.path.join(output_dir, output_filename_or)
    results_df_or.to_csv(output_path_or, index=False, encoding='utf-8')
    print(f"\nScan complete. {len(results_df_or)} stocks matched at least one of {TOTAL_CONDITIONS} criteria (OR Logic).")
    print(f"OR Logic results saved to: {output_path_or}")
    
    # 8. 保存 AND 逻辑结果 
    if not results_df_and.empty:
        output_filename_and = f'bullish_scan_AND_logic_COMPREHENSIVE_{timestamp_str}.csv'
        output_path_and = os.path.join(output_dir, output_filename_and)
        results_df_and.to_csv(output_path_and, index=False, encoding='utf-8')
        print(f"Found {len(results_df_and)} stocks matching ALL {TOTAL_CONDITIONS} criteria (AND Logic).")
        print(f"AND Logic results saved to: {output_path_and}")
    else:
        print(f"No stocks matched ALL {TOTAL_CONDITIONS} comprehensive criteria (AND Logic).")

if __name__ == '__main__':
    main()
