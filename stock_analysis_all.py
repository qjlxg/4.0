import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import matplotlib
from matplotlib.font_manager import FontProperties
from tqdm import tqdm
import time
import os
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

# --- 新增常量：名称映射文件路径 ---
NAME_MAP_FILE = 'stock_names.csv' 

# --- 修复 1: 解决中文字体问题 (修正 Matplotlib 错误) ---
# 保留字体配置部分
fontprop = None
try:
    font_path = '/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc'
    if os.path.exists(font_path):
        fontprop = FontProperties(fname=font_path)
        matplotlib.rcParams['font.family'] = fontprop.get_name() 
        chinese_font = {'fontproperties': fontprop}
        print("中文字体加载成功。")
    else:
        matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'WenQuanYi Zen Hei', 'Arial Unicode MS']
        chinese_font = {'family': 'sans-serif'}
        print("警告: 无法找到指定中文字体，已尝试设置备用字体。")
        
    matplotlib.rcParams['axes.unicode_minus'] = False
except Exception as e:
    print(f"致命警告: 字体配置失败: {e}，将使用 Matplotlib 默认字体。")
    chinese_font = {}
# ------------------------------------

# 定义常量
DATA_DIR = 'stock_data'
RESULTS_DIR = 'results'

PROGRESS_FILE = os.path.join(RESULTS_DIR, 'progress.txt')
FINAL_RESULTS_FILE = f'{RESULTS_DIR}/buy_signals_final_summary.csv'

# 创建结果保存目录和数据保存目录
if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# --- 新增：加载名称映射的函数 ---
def load_name_map():
    """从 stock_names.csv 文件加载股票代码到名称的映射字典。"""
    name_map = {}
    if os.path.exists(NAME_MAP_FILE):
        print(f"正在加载名称映射文件 '{NAME_MAP_FILE}'...")
        delimiters = [',', '\t', ';']
        encodings = ['utf-8', 'utf-8-sig', 'gbk'] 
        found_map = False
        
        for enc in encodings:
            for delim in delimiters:
                if found_map: break
                try:
                    # 读取时将 code 列作为字符串处理
                    df_names = pd.read_csv(NAME_MAP_FILE, 
                                            dtype={'code': str}, 
                                            encoding=enc, 
                                            sep=delim)
                    
                    if 'code' in df_names.columns and 'name' in df_names.columns:
                        # 统一股票代码格式为 6 位带前导零
                        df_names['code'] = df_names['code'].astype(str).str.zfill(6) 
                        
                        print(f"成功使用编码'{enc}'和分隔符'{repr(delim)}'加载名称映射。")
                        name_map = df_names.set_index('code')['name'].to_dict()
                        found_map = True
                        break 
                except Exception:
                    continue
        if not name_map:
            print("警告：无法正确解析名称映射文件，名称将显示为'本地数据_[代码]'。")
    else:
        print(f"警告：名称映射文件 '{NAME_MAP_FILE}' 未找到，名称将显示为'本地数据_[代码]'。")
    return name_map

# 【修改】：获取股票列表 -> 从本地文件列表构造 (接受 name_map)
def get_local_stock_list(name_map):
    """遍历 DATA_DIR 目录下的所有 CSV 文件，构造股票列表 DataFrame"""
    print(f"正在扫描 {DATA_DIR} 目录下的本地股票数据...")
    
    stock_data = []
    
    # 获取 DATA_DIR 中所有以 .csv 结尾的文件
    csv_files = [f for f in os.listdir(DATA_DIR) if f.endswith('.csv')]
    
    for file_name in csv_files:
        stock_code = file_name.replace('.csv', '')
        # 使用名称映射，如果找不到则使用默认名称
        stock_name = name_map.get(stock_code, f"本地数据_{stock_code}") 
        
        stock_data.append({'代码': stock_code, '名称': stock_name})
        
    if not stock_data:
        print("致命错误：未在 stock_data 目录下找到任何 .csv 文件。")
        return pd.DataFrame() 

    stock_list_df = pd.DataFrame(stock_data)
    print(f"扫描完成，找到 {len(stock_list_df)} 个本地股票数据文件。")
    return stock_list_df

# 计算技术指标
def calculate_technical_indicators(df):
    """计算各种技术指标"""
    # 确保列类型为数字
    for col in ['开盘', '收盘', '最高', '最低', '成交量']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # 【数据清洗】：成交量缺失值处理
    if '成交量' in df.columns:
        df['成交量'] = df['成交量'].fillna(0)
    
    # MA
    df['MA5'] = df['收盘'].rolling(window=5).mean()
    df['MA10'] = df['收盘'].rolling(window=10).mean()
    df['MA20'] = df['收盘'].rolling(window=20).mean()
    df['MA60'] = df['收盘'].rolling(window=60).mean()
    
    # RSI
    delta = df['收盘'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    # 避免除以零的错误
    rs = gain / (loss + 1e-9) 
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # MACD
    df['EMA12'] = df['收盘'].ewm(span=12, adjust=False).mean()
    df['EMA26'] = df['收盘'].ewm(span=26, adjust=False).mean()
    df['DIF'] = df['EMA12'] - df['EMA26']
    df['DEA'] = df['DIF'].ewm(span=9, adjust=False).mean()
    df['MACD'] = 2 * (df['DIF'] - df['DEA'])
    
    # KDJ
    low_min = df['最低'].rolling(window=9).min()
    high_max = df['最高'].rolling(window=9).max()
    df['RSV'] = 100 * ((df['收盘'] - low_min) / (high_max - low_min + 1e-9))
    df['K'] = df['RSV'].ewm(com=2, adjust=False).mean()
    df['D'] = df['K'].ewm(com=2, adjust=False).mean()
    df['J'] = 3 * df['K'] - 2 * df['D']
    
    return df

# 分析股票的买入信号 (新增 close_price 到返回结果)
def analyze_buy_signals(df, stock_code, stock_name, max_close_price):
    """
    分析股票的买入信号。
    排除收盘价 < 5.0 或 >= max_close_price (即 20.0) 元的股票
    """
    signals = []
    score = 0
    
    if len(df) < 60:
        return {"code": stock_code, "name": stock_name, "signals": ["数据不足 (少于60天)"], "score": 0, "close_price": None}
    
    required_cols = ['收盘', 'MA5', 'MA20', 'RSI', 'MACD', 'DIF', 'DEA', 'K', 'D', 'J']
    for col in required_cols:
        if col not in df.columns or df[col].isnull().all():
            return {"code": stock_code, "name": stock_name, "signals": [f"数据计算错误: {col} 列缺失或全为空"], "score": 0, "close_price": None}
            
    current_close = df['收盘'].iloc[-1]
    
    # 下限过滤
    LOWER_BOUND = 5.0
    if current_close < LOWER_BOUND:
        return {"code": stock_code, "name": stock_name, "signals": [f"收盘价{current_close:.2f}元，低于{LOWER_BOUND}元，排除"], "score": 0, "close_price": current_close}
    
    # 上限过滤
    if current_close >= max_close_price:
        return {"code": stock_code, "name": stock_name, "signals": [f"收盘价{current_close:.2f}元，高于或等于{max_close_price}元，排除"], "score": 0, "close_price": current_close}

    
    try:
        # MA金叉
        if df['MA5'].iloc[-1] > df['MA20'].iloc[-1] and df['MA5'].iloc[-2] <= df['MA20'].iloc[-2]:
            signals.append("MA金叉形成")
            score += 20
        
        # RSI 低位
        current_rsi = df['RSI'].iloc[-1]
        if 30 <= current_rsi <= 50:
            signals.append(f"RSI值为{current_rsi:.2f}，处于低位回升阶段")
            score += 15
        elif current_rsi < 30:
            signals.append(f"RSI值为{current_rsi:.2f}，股票可能超卖")
            score += 10
        
        # MACD金叉
        if df['DIF'].iloc[-1] > df['DEA'].iloc[-1] and df['DIF'].iloc[-2] <= df['DEA'].iloc[-2]:
            signals.append("MACD金叉形成")
            score += 20
        
        # KDJ金叉
        if df['K'].iloc[-1] > df['D'].iloc[-1] and df['K'].iloc[-2] <= df['D'].iloc[-2]:
            if df['K'].iloc[-1] < 50:
                signals.append("KDJ低位金叉")
                score += 15
            else:
                signals.append("KDJ金叉")
                score += 10
        
        # 放量上涨
        if len(df['成交量']) >= 6:
            # 避免除以零的错误
            avg_volume = df['成交量'].iloc[-6:-1].mean()
            current_volume = df['成交量'].iloc[-1]
            price_change = (df['收盘'].iloc[-1] / df['收盘'].iloc[-2] - 1) * 100
            
            if avg_volume > 1e-9 and current_volume > 1.5 * avg_volume and price_change > 0:
                signals.append(f"放量上涨: 量比{current_volume/avg_volume:.2f}, 涨幅{price_change:.2f}%")
                score += 15
        
        # 突破20日前高
        if len(df['最高']) >= 20:
            recent_high = df['最高'].iloc[-20:-1].max()
            if df['收盘'].iloc[-1] > recent_high:
                signals.append("突破20日前高")
                score += 15
            
    except Exception as e:
        signals.append(f"分析过程出错: {str(e)}")
        score = 0
    
    return {
        "code": stock_code,
        "name": stock_name,
        "signals": signals,
        "score": score,
        "close_price": current_close
    }

# 【修改】：仅加载本地数据并分析
def analyze_single_stock(stock_code, stock_name, max_close_price):
    """从本地文件加载数据，并进行分析 (不生成图表)"""
    try:
        df = load_local_stock_data(stock_code)
        
        if df.empty:
            return {
                "code": stock_code,
                "name": stock_name,
                "signals": ["数据加载失败/为空"],
                "score": 0,
                "close_price": None
            }

        df = calculate_technical_indicators(df)
        result = analyze_buy_signals(df, stock_code, stock_name, max_close_price)
        
        return result
        
    except Exception as e:
        return {
            "code": stock_code,
            "name": stock_name,
            "signals": [f"分析失败: {str(e)}"],
            "score": 0,
            "close_price": None
        }

# 【修改】：主函数
def main():
    # 价格过滤参数定义
    MAX_CLOSE_PRICE = 20.0 
    # 并发线程数
    MAX_WORKERS = 8 

    # ----------------------------------------
    # NEW: 0. 加载名称映射
    # ----------------------------------------
    name_map = load_name_map()
    
    # ----------------------------------------
    # 1. 检查和加载进度
    # ----------------------------------------
    start_index = 0
    
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, 'r') as f:
            try:
                start_index = int(f.read().strip())
                print(f"检测到未完成的进度，将从第 {start_index + 1} 只股票开始分析。")
            except ValueError:
                os.remove(PROGRESS_FILE)
                print("进度文件损坏，将从头开始分析。")
                start_index = 0

    # 【修改】：传入 name_map
    stock_list = get_local_stock_list(name_map)
    if stock_list.empty:
        print("致命错误：无法获取股票列表 (本地无 CSV 文件)，程序退出。")
        return

    # ----------------------------------------
    # 2. 排除 ST 股票 和 创业板股票 (300/301开头) 和科创板 (688开头) 股票
    # ----------------------------------------
    
    stock_list['代码'] = stock_list['代码'].astype(str)
    
    # 排除 创业板(300/301开头) 和 科创板(688开头) 股票
    gem_and_star_excluded_df = stock_list[
        stock_list['代码'].str.startswith('300') | 
        stock_list['代码'].str.startswith('301') |
        stock_list['代码'].str.startswith('688')
    ]
    stock_list = stock_list.drop(gem_and_star_excluded_df.index)
    excluded_gem_star_count = len(gem_and_star_excluded_df)
    
    print(f"⚠️ 警告: 仅根据本地文件名，无法排除 ST/退市 股票。")
    print(f"✅ 已排除 {excluded_gem_star_count} 只 创业板 (300/301开头) 和科创板 (688开头) 股票。")

    total_stocks = len(stock_list)
    
    # ----------------------------------------
    # 3. 设置分段处理范围
    # ----------------------------------------
    BATCH_SIZE = 880
    
    END_INDEX = min(start_index + BATCH_SIZE, total_stocks)
    
    if start_index >= total_stocks:
        print("所有股票已分析完毕。任务结束。")
        merge_and_deduplicate_results()
        if os.path.exists(PROGRESS_FILE):
             os.remove(PROGRESS_FILE)
        exit(0) 

    print(f"共获取到 {total_stocks} 只股票 (排除创业板、科创板后)")
    print(f"本次任务范围: 分析 {start_index + 1} 到 {END_INDEX} 只股票 (价格过滤: >= 5.0 元 且 < {MAX_CLOSE_PRICE} 元)。")
    print(f"使用 {MAX_WORKERS} 个线程并发处理...")
    
    current_batch = stock_list.iloc[start_index:END_INDEX]
    results = []
    
    # ----------------------------------------
    # 4. 使用 ThreadPoolExecutor 进行并发处理
    # ----------------------------------------
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_stock = {
            # 名称已在 stock_list 中正确映射
            executor.submit(analyze_single_stock, row['代码'], row['名称'], MAX_CLOSE_PRICE): row 
            for _, row in current_batch.iterrows()
        }
        
        for future in tqdm(as_completed(future_to_stock), total=len(current_batch), desc="分析进度"):
            result = future.result()
            
            # 只将评分 >= 30 的结果添加到 results 列表中
            if result and result.get('score', 0) >= 30:
                results.append(result)

    # ----------------------------------------
    # 5. 结果处理和进度保存
    # ----------------------------------------
    
    if results:
        csv_data = []
        for result in results:
            signals_str = '; '.join(result['signals'])
            
            # 从分析结果中获取收盘价，无需再次读取文件
            close_price_raw = result.get('close_price')
            close_price = f"{close_price_raw:.2f}" if isinstance(close_price_raw, (int, float)) else 'N/A'
            
            row = {
                '股票代码': result['code'],
                '股票名称': result['name'], # 股票名称已正确映射
                '收盘价': close_price,
                '买入信号': signals_str,
                '评分': result['score'],
            }
            csv_data.append(row)
            
        df_result = pd.DataFrame(csv_data)
        df_result.to_csv(f'{RESULTS_DIR}/buy_signals_{start_index}_to_{END_INDEX}.csv', index=False, encoding='utf-8-sig')
        print(f"\n批次结果已保存到 {RESULTS_DIR}/buy_signals_{start_index}_to_{END_INDEX}.csv")

    
    next_start_index = END_INDEX
    
    if next_start_index < total_stocks:
        with open(PROGRESS_FILE, 'w') as f:
            f.write(str(next_start_index))
        print(f"本次任务完成。进度已保存到 {PROGRESS_FILE}，下次将从 {next_start_index + 1} 只股票继续。")
        # 退出码 99 通知工作流重启
        exit(99) 
    else:
        # 任务全部完成后的处理
        merge_and_deduplicate_results()
        if os.path.exists(PROGRESS_FILE):
             os.remove(PROGRESS_FILE)
        print("所有股票已分析完毕。进度文件已清除。")
        # 退出码 0 通知工作流完成
        exit(0)

def merge_and_deduplicate_results():
    """合并所有批次结果文件，去重并生成最终汇总文件"""
    all_results = []
    
    batch_files = [f for f in os.listdir(RESULTS_DIR) if f.startswith('buy_signals_') and f.endswith('.csv')]
    
    if not batch_files:
        print("未找到任何批次结果文件，跳过汇总。")
        return
        
    for filename in batch_files:
        try:
            file_path = os.path.join(RESULTS_DIR, filename)
            df = pd.read_csv(file_path, encoding='utf-8-sig') 
            all_results.append(df)
        except Exception as e:
            print(f"读取批次文件 {filename} 失败: {e}")
            
    if not all_results:
        return
        
    combined_df = pd.concat(all_results, ignore_index=True)
    
    # 根据股票代码去重 (保留最新的记录)
    final_df = combined_df.drop_duplicates(subset=['股票代码'], keep='last')
    
    # 整理列顺序
    final_cols = ['股票代码', '股票名称', '收盘价', '评分', '买入信号']
    if all(col in final_df.columns for col in final_cols):
        final_df = final_df[final_cols]
    
    final_df.to_csv(FINAL_RESULTS_FILE, index=False, encoding='utf-8-sig')
    print(f"\n✨ 最终汇总结果已保存到 {FINAL_RESULTS_FILE} (共 {len(final_df)} 条记录)。")


if __name__ == "__main__":
    main()
