# stock_analyzer_ultimate_final_V4.0_NakedK_Volume_System.py
# 五策略集成 V4.0 (C>A>B>E>D) 极简量价交易系统 (裸K+量价核心驱动)

import pandas as pd
import pandas_ta as ta
import os
from datetime import datetime, timedelta
import pytz
import numpy as np
import re

# --- 配置 (V4.0 固化) ---
STOCK_DATA_DIR = "stock_data"
OUTPUT_DIR_BASE = "combined_results"
MAX_DAYS_TO_LOOK_BACK = 7 # 策略分析不再使用此变量，但保留配置
TOP_CANDIDATES_COUNT = 5

# --- 名称映射配置 ---
NAME_MAP_FILE = 'stock_names.csv'

# 输入文件列名标准化映射
CHINESE_TO_ENGLISH_MAP = {
    '日期': 'Date', '开盘': 'Open', '收盘': 'Close', '最高': 'High', '最低': 'Low',
    '成交量': 'Volume', '成交额': 'Amount', '换手率': 'TurnoverRate'
}

# 修正：定义缺失的 STANDARDIZED_CHINESE_MAP，用于 analyze_and_filter_stocks 函数内部
# 注意：这里假设输入CSV中的列名是标准的中文，所以 STANDARDIZED_CHINESE_MAP 直接等于 CHINESE_TO_ENGLISH_MAP
STANDARDIZED_CHINESE_MAP = CHINESE_TO_ENGLISH_MAP

# --- 辅助函数：加载名称映射 (同前一个脚本的健壮加载逻辑) ---
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
                    df_names = pd.read_csv(NAME_MAP_FILE,
                                           dtype={'code': str},
                                           encoding=enc,
                                           sep=delim)
                    
                    if 'code' in df_names.columns and 'name' in df_names.columns:
                        # 统一股票代码格式为 6 位带前导零
                        df_names['code'] = df_names['code'].astype(str).str.zfill(6)
                        name_map = df_names.set_index('code')['name'].to_dict()
                        found_map = True
                        print(f"✅ 成功加载 {len(name_map)} 条股票名称映射。")
                        break
                except Exception:
                    continue
        if not name_map:
            print("⚠️ 警告：无法正确解析名称映射文件，将跳过名称映射。")
    else:
        print(f"⚠️ 警告：名称映射文件 '{NAME_MAP_FILE}' 未找到，将跳过名称映射。")
    return name_map

# --- 辅助函数：严格路径查找 (不再用于查找输入信号，仅保留 get_current_shanghai_time) ---
def get_current_shanghai_time():
    """获取当前上海时间"""
    shanghai_tz = pytz.timezone('Asia/Shanghai')
    # 使用 2025 年 11 月 11 日 15:01:00 作为当前时间参考
    return datetime.now(shanghai_tz)

def find_input_signal_path():
    """原严格查找最新的信号文件函数。在 V4.0 全量分析中不再使用，但为保持结构完整保留。"""
    # 此函数已不再执行信号文件查找任务
    print("⚠️ 注意: 信号文件查找逻辑已停用，脚本将全量分析 'stock_data' 目录下的所有数据。")
    return None

def calculate_slope(series, periods):
    """计算序列的斜率"""
    if len(series) < periods: return 0
    return (series.iloc[-1] - series.iloc[-periods]) / periods

# --- 辅助函数：通用市值自适应换手率函数 ---
def get_cap_adapted_turnover(code, base_turnover):
    """根据股票代码前缀进行板块/市值自适应调整换手率阈值。"""
    code_str = str(code)

    if code_str.startswith('688') or code_str.startswith('300'):
        return base_turnover
    elif code_str.startswith('60'):
        return base_turnover * 0.3
    elif code_str.startswith('00'):
        return base_turnover * 0.6
    else:
        return base_turnover

# --- 指标计算和基础判断函数 ---
def calculate_all_indicators(df):
    """计算所有必要的技术指标"""
    if df.empty: return df

    # MA
    df.ta.sma(length=5, append=True, col_names=('MA5',)); df.ta.sma(length=10, append=True, col_names=('MA10',))
    df.ta.sma(length=20, append=True, col_names=('MA20',)); df.ta.sma(length=60, append=True, col_names=('MA60',))
    # RSI
    df.ta.rsi(length=6, append=True, col_names=('RSI6',))
    # KDJ
    kdj_df = df.ta.stoch(k=14, d=3, smooth_k=3, append=True)
    df['K'] = kdj_df.iloc[:, 0]; df['D'] = kdj_df.iloc[:, 1]; df['J'] = (3 * df['K']) - (2 * df['D'])
    # MACD
    macd_df = df.ta.macd(fast=12, slow=26, signal=9, append=True)
    df['DIF'] = macd_df.iloc[:, 0]; df['DEA'] = macd_df.iloc[:, 2]; df['MACDh'] = macd_df.iloc[:, 1]
    df['Prev_DIF'] = macd_df.iloc[:, 0].shift(1); df['Prev_DEA'] = macd_df.iloc[:, 2].shift(1)

    # 长期趋势判断 MACD DIF MA60
    df.ta.sma(close=df['DIF'], length=60, append=True, col_names=('DIF_MA60',))

    # Volume MA (V4.0：MA3V 用于梯量判断)
    df.ta.sma(close=df['Volume'], length=3, append=True, col_names=('MA3V',))
    df.ta.sma(close=df['Volume'], length=5, append=True, col_names=('MA5V',))
    df['OBV'] = ta.obv(df['Close'], df['Volume'], append=True); df['Prev_OBV'] = df['OBV'].shift(1)

    # BBands
    bbands_df = df.ta.bbands(length=20, std=2, append=True)
    df['BBL'] = bbands_df.iloc[:, 0]; df['BBM'] = bbands_df.iloc[:, 1]
    # 20日内前高
    df['Max_High_Prev_20'] = df['High'].rolling(window=21, min_periods=1).max().shift(1)

    return df

def is_limit_up(df):
    """判断是否涨停"""
    if len(df) < 2: return False
    latest = df.iloc[-1]; prev = df.iloc[-2]
    if pd.isna(latest['Close']) or pd.isna(prev['Close']): return False
    price_up_ratio = (latest['Close'] - prev['Close']) / prev['Close']
    is_price_at_high = latest['Close'] >= latest['High'] * 0.999
    target_ratio = 0.10
    code = str(latest.get('code', ''))
    if code.startswith('688') or code.startswith('300'): target_ratio = 0.20
    is_up_limit = is_price_at_high and abs(price_up_ratio - target_ratio) < 0.015
    return is_up_limit


# --- V4.0 (极简量价交易系统) 策略函数 (保持不变) ---

def enhanced_leader_restart_strategy(df):
    """
    策略 E (V4.0 维持): 龙头股二次启动 - 严格高位风险过滤。
    """
    if len(df) < 20: return False
    latest = df.iloc[-1]
    code = str(latest.get('code', ''))

    # V4.0 风险优化 1: KDJ/RSI 极限高位钝化过滤 (排除极端高位)
    if latest['J'] > 95 or latest['RSI6'] > 85: return False

    # 1. 近5日内涨停
    had_limit_up_recently = False
    for i in range(max(0, len(df)-6), len(df)-1):
        if i >= 1:
            prev_close = df['Close'].iloc[i-1]; current_close = df['Close'].iloc[i]
            current_high = df['High'].iloc[i]; ratio = (current_close - prev_close) / prev_close
            target_ratio = 0.199 if code.startswith('688') or code.startswith('300') else 0.099
            is_at_high = current_close >= current_high * 0.998
            if ratio >= target_ratio * 0.98 and is_at_high:
                had_limit_up_recently = True
                break
    if not had_limit_up_recently: return False

    # 2. 二次启动量能
    prev_5_volume_mean = df['Volume'].iloc[-6:-1].mean()
    restart_volume = latest['Volume'] > 2.0 * prev_5_volume_mean

    # V4.0 风险优化 2: 量能衰竭过滤
    is_volume_not_decaying = latest['MA3V'] >= 0.9 * latest['MA5V']
    if not is_volume_not_decaying: return False

    # 3. RSI 和 KDJ 严格限制上限
    rsi_strong = (latest['RSI6'] > 65) and (latest['RSI6'] < 80)
    kdj_strong = (latest['J'] > 50) and (latest['J'] < 95)

    # 4. 均线对齐 (保持趋势一致性)
    ma_alignment = (latest['MA5'] > latest['MA10'] > latest['MA20'])

    # 5. 趋势加强：DIF 必须高于 DIF_MA60
    is_macd_long_trend = latest['DIF'] > latest['DIF_MA60']

    return (had_limit_up_recently and restart_volume and rsi_strong and
            ma_alignment and kdj_strong and is_macd_long_trend and is_volume_not_decaying)

def enhanced_strong_breakout_strategy(df):
    """
    策略 D (V4.0 维持): 强势突破 - 严格高位风险过滤。
    """
    if len(df) < 61 or df['Max_High_Prev_20'].iloc[-1] is np.nan: return False
    latest = df.iloc[-1]; code = str(latest.get('code', ''))

    # V4.0 风险优化 1: KDJ/RSI 极限高位钝化过滤 (排除极端高位)
    if latest['J'] > 95 or latest['RSI6'] > 85: return False

    # 零轴上方确认主升浪初期
    is_dif_above_zero = latest['DIF'] > 0
    if not is_dif_above_zero: return False

    # 趋势加强：DIF 必须高于 DIF_MA60
    is_macd_long_trend = latest['DIF'] > latest['DIF_MA60']
    if not is_macd_long_trend: return False

    # 1. 通用自适应换手率
    base_turnover_min = 4.0
    adapted_turnover_min = get_cap_adapted_turnover(code, base_turnover_min)

    # 2. 成交量条件
    volume_condition = latest['Volume'] > 2.0 * latest['MA5V']

    # 3. 价格突破（限制突破上限）
    max_high_prev_20 = latest['Max_High_Prev_20']
    price_condition = (latest['Close'] > max_high_prev_20 * 1.005) and (latest['Close'] < max_high_prev_20 * 1.05)

    # 4. RSI条件收紧
    if code.startswith('688') or code.startswith('300'):
        rsi_condition = (latest['RSI6'] > 60) and (latest['RSI6'] < 80)
    else:
        rsi_condition = (latest['RSI6'] > 65) and (latest['RSI6'] < 75)

    # 5. 动量条件
    macd_momentum = latest['DIF'] > latest['DEA']; kdj_momentum = latest['J'] > 70
    rsi_momentum = latest['RSI6'] > 60; momentum_condition = (macd_momentum and rsi_momentum) or kdj_momentum

    # 6. 换手率条件
    turnover_upper_limit = 25.0 if code.startswith('688') or code.startswith('300') else 12.0
    turnover_condition = (adapted_turnover_min < latest['TurnoverRate'] < turnover_upper_limit)

    return (volume_condition and price_condition and rsi_condition and
            momentum_condition and turnover_condition)

def enhanced_pullback_strategy(df):
    """
    策略 A (V4.0 固化): 强势回踩接力 - 保持趋势判断，强化 K线和量能启动 (K线阳线要求更高)。
    """
    if len(df) < 61: return False
    latest = df.iloc[-1]; prev = df.iloc[-2]; code = str(latest.get('code', ''))

    # 1. 趋势判断：要求完美多头排列 (趋势仍是生命线)
    is_trend = (latest['Close'] > latest['MA5'] > latest['MA10'] > latest['MA20']) \
               and (calculate_slope(df['MA20'].tail(5), 5) > 0)

    # 2. V4.0 K线形态加强：阳线启动且涨跌幅大于 2.0%
    is_price_up = latest['Close'] > prev['Close']
    is_bullish_and_strong = is_price_up and (latest['Close'] / prev['Close'] - 1) > 0.020

    # 3. 均线支撑强化 (收盘价必须高于 MA5)
    is_close_above_ma5 = latest['Close'] > latest['MA5']

    # 4. 量能和换手率
    base_turnover = 3.0
    adapted_turnover = get_cap_adapted_turnover(code, base_turnover)
    turnover_condition = adapted_turnover < latest['TurnoverRate'] < 20.0
    volume_condition = latest['Volume'] > 1.5 * latest['MA5V'] # 量能爆发

    # 5. 辅助指标和风险控制
    rsi_condition = (latest['RSI6'] >= 45) and (latest['RSI6'] < 65)
    kdj_condition = latest['K'] < 80
    is_macd_long_trend = latest['DIF'] > latest['DIF_MA60'] # 长期趋势确认

    return (is_trend and is_bullish_and_strong and is_close_above_ma5 and
            rsi_condition and is_macd_long_trend and turnover_condition and volume_condition and kdj_condition)

def is_low_position_start_strategy(df):
    """
    策略 B (V4.0 固化): 裸K低位启动/提前埋伏 - 专注于缩量横盘后的放量突破阳线。
    """
    if len(df) < 61: return False
    latest_data = df.iloc[-1]; prev_data = df.iloc[-2]; code = str(latest_data.get('code', ''))

    # 1. 裸K形态：低位盘整/突破
    # 过去 10 日低点波动小于 5%
    n_days = 10
    low_range = df['Low'].iloc[-n_days:].max() - df['Low'].iloc[-n_days:].min()
    price_range_small = low_range / latest_data['Close'] < 0.05

    # 当日阳线突破盘整区：收盘价明确高于前 N 日收盘价高点 (1% 容错)
    prev_high_close = df['Close'].iloc[-n_days:-1].max()
    is_breakout_candle = (latest_data['Close'] > prev_high_close * 1.01) and (latest_data['Close'] > prev_data['Close'])
    
    # 2. 量价共振
    # V4.0 优化点：量能爆发和梯量 (当日放量，且MA3V开始抬头)
    is_volume_burst_B = (latest_data['Volume'] > 2.0 * latest_data['MA5V']) and \
                         (latest_data['Volume'] < 4.0 * latest_data['MA5V']) and \
                         (latest_data['MA3V'] >= 1.0 * latest_data['MA5V'])
    
    # 换手率：低位合理放量 (V4.0 换手率上限收紧至 8.0%)
    base_turnover_B = 0.8
    adapted_turnover_B = get_cap_adapted_turnover(code, base_turnover_B)
    is_turnover_active_B = (latest_data['TurnoverRate'] > adapted_turnover_B) and (latest_data['TurnoverRate'] < 8.0)
    
    # 3. 辅助指标和风险控制 (作为低位辅助确认)
    is_kdj_low_B = latest_data['K'] < 50
    is_macd_low_gold_B = (latest_data['DIF'] > latest_data['DEA']) and (latest_data['DIF'] < 0.05)
    is_macd_long_trend_B = latest_data['DIF'] > latest_data['DIF_MA60']

    return (price_range_small and is_breakout_candle and
            is_volume_burst_B and is_turnover_active_B and
            is_kdj_low_B and is_macd_low_gold_B and is_macd_long_trend_B)

def is_new_strategy_C(df):
    """
    策略 C (V4.0 固化)：裸K量价平台突破共振 - 强调平台突破形态和多动量共振。
    """
    if len(df) < 61: return False
    latest_data = df.iloc[-1]; prev_data = df.iloc[-2]
    
    # V4.0 趋势判断：MA5 向上且收盘价在 MA20 上方 (确认上升趋势)
    is_trend_up_C = (calculate_slope(df['MA5'].tail(5), 5) > 0.0) and (latest_data['Close'] > latest_data['MA20'])

    # 1. 裸K平台突破 (N=40 日)
    n_days_C = 40
    prev_high_C = df['High'].iloc[-n_days_C:-1].max()
    # 当日K线收盘价明确突破过去 40 日的高点（平台突破）
    is_platform_breakout_C = (latest_data['Close'] > prev_high_C * 1.01)

    # 2. 量价共振
    is_volume_confirm_C = (latest_data['Volume'] > 2.0 * latest_data['MA5V']) and \
                          (latest_data['Volume'] < 4.0 * latest_data['MA5V']) and \
                          (latest_data['MA3V'] >= 1.0 * latest_data['MA5V'])
    is_obv_up_C = latest_data['OBV'] > prev_data['OBV']

    # 3. 动量共振 (指标从弱转强)
    is_rsi_strong_C = (latest_data['RSI6'] > 60) and (latest_data['RSI6'] > prev_data['RSI6'])
    
    # KDJ 金叉且不在高位
    is_kdj_golden_C = (latest_data['K'] > latest_data['D']) and (prev_data['K'] <= prev_data['D']) and (latest_data['K'] < 70)

    # MACD 从负值区回零轴或在零轴上方金叉
    is_macd_turn_strong_C = (latest_data['DIF'] > latest_data['DEA']) and (latest_data['DIF'] > -0.05)
    
    # 4. 长期趋势确认
    is_macd_long_trend_C = latest_data['DIF'] > latest_data['DIF_MA60']

    return (is_trend_up_C and is_platform_breakout_C and
            is_volume_confirm_C and is_obv_up_C and
            is_rsi_strong_C and is_kdj_golden_C and is_macd_turn_strong_C and is_macd_long_trend_C)


def log_strategy_details(code, stock_name, strategy_results):
    """日志系统 (保留)"""
    details = []
    for strategy in ['C', 'A', 'B', 'E', 'D']:
        passed = strategy_results.get(strategy, False)
        status = "✅" if passed else "❌"
        details.append(f"{strategy}:{status}")
    print(f"🔍 {code} ({stock_name}) 策略详情: {', '.join(details)}")


# --- 核心分析函数 (已修改) ---

def analyze_and_filter_stocks(stock_data_path, name_map):
    """
    主分析函数：遍历 stock_data_path 目录下的所有 CSV 文件，计算指标，应用策略，并输出结果 DataFrame
    新增参数: name_map 用于填充或校正股票名称。
    """
    if not os.path.exists(stock_data_path):
        print(f"❌ 股票数据目录不存在: {stock_data_path}，流程终止。")
        return pd.DataFrame()

    all_files = [f for f in os.listdir(stock_data_path) if f.endswith('.csv')]
    if not all_files:
        print(f"❌ 股票数据目录 {stock_data_path} 中没有找到任何 CSV 文件，流程终止。")
        return pd.DataFrame()

    print(f"✅ 成功找到 {len(all_files)} 个股票数据文件，开始全量分析...")
    results = []
    REQUIRED_COLUMNS = ['Close', 'High', 'Low', 'Open', 'Volume', 'TurnoverRate']

    for stock_file_name in all_files:
        stock_file_path = os.path.join(stock_data_path, stock_file_name)
        
        # 1. 从文件名解析 code 并标准化
        match = re.match(r'(\d{6})\.csv$', stock_file_name)
        if match:
            code = str(match.group(1)).zfill(6)
        else:
            code = stock_file_name.replace('.csv', '')
            code = str(code).zfill(6) # Fallback and standardize

        # 2. 使用名称映射获取股票名称
        stock_name = name_map.get(code, 'N/A')

        try:
            history_df = pd.read_csv(stock_file_path)

            # 列名标准化
            rename_dict = {}
            for original_col in history_df.columns:
                standard_col_key = re.sub(r'[^\u4e00-\u9fa5]+', '', str(original_col).strip())
                # 修正：使用已定义的 STANDARDIZED_CHINESE_MAP
                if standard_col_key in STANDARDIZED_CHINESE_MAP:
                    rename_dict[original_col] = STANDARDIZED_CHINESE_MAP[standard_col_key]
                    continue
                stripped_lower_col = str(original_col).strip().lower()
                if stripped_lower_col in ['trade_date', 'date']:
                    rename_dict[original_col] = 'Date'

            history_df.rename(columns=rename_dict, inplace=True)

            missing_cols = [col for col in REQUIRED_COLUMNS if col not in history_df.columns]
            if missing_cols or history_df.empty or len(history_df) < 61:
                # print(f"⚠️ 跳过 {code}: 缺少所需列或数据不足 (需61行)，缺少列: {missing_cols}")
                continue

            # 3. 最终确认代码和名称 (以名称映射为准，除非名称映射结果为 N/A)
            latest_row = history_df.iloc[-1]
            
            # 如果名称映射是 N/A，则尝试使用 CSV 文件中的 '股票名称'
            if stock_name == 'N/A' and '股票名称' in history_df.columns and not pd.isna(latest_row['股票名称']):
                stock_name = str(latest_row['股票名称'])

            history_df['code'] = code # 确保 df 中有 code 列用于 is_limit_up 和 get_cap_adapted_turnover
            df_with_indicators = calculate_all_indicators(history_df.copy())

            # 确保最新数据行和关键指标不为空
            if len(df_with_indicators) < 2 or df_with_indicators.iloc[-1].isnull().any():
                # print(f"⚠️ 跳过 {code}: 指标计算后数据行不足或最新行有空值")
                continue

            # --- 策略调用 (V4.0 固化策略) ---
            is_limit_up_today = is_limit_up(df_with_indicators)
            is_Strategy_A_Pullback = enhanced_pullback_strategy(df_with_indicators)
            is_Strategy_B_LowStart = is_low_position_start_strategy(df_with_indicators)
            is_Strategy_C_NewStart = is_new_strategy_C(df_with_indicators)
            is_Strategy_D_Breakout = enhanced_strong_breakout_strategy(df_with_indicators)
            is_Strategy_E_Restart = enhanced_leader_restart_strategy(df_with_indicators)

            strategy_results = {
                'A': is_Strategy_A_Pullback, 'B': is_Strategy_B_LowStart,
                'C': is_Strategy_C_NewStart, 'D': is_Strategy_D_Breakout,
                'E': is_Strategy_E_Restart
            }
            log_strategy_details(code, stock_name, strategy_results)

            # --- 最终入选判断与优先级排序 (C > A > B > E > D) ---
            strategy_type = "None"
            if is_Strategy_C_NewStart:
                strategy_type = "C_New_Strategy (最高共振)"
            elif is_Strategy_A_Pullback:
                strategy_type = "A_Strong_Pullback (中风险接力)"
            elif is_Strategy_B_LowStart:
                strategy_type = "B_Low_Position_Start (低风险埋伏)"
            elif is_Strategy_E_Restart:
                strategy_type = "E_Leader_Restart (二次启动)"
            elif is_Strategy_D_Breakout:
                strategy_type = "D_Strong_Breakout (高风险追涨/优化)"

            if strategy_type != "None":
                print(f"✅ {code} ({stock_name}) 满足策略: {strategy_type}")

                latest_data = df_with_indicators.iloc[-1]
                result_row = {
                    'code': code, 'name': stock_name, 'Strategy_Type': strategy_type,
                    'Close': latest_data.get('Close'), 'TurnoverRate': latest_data.get('TurnoverRate'),
                    'RSI6': latest_data.get('RSI6'), 'KDJ_J': latest_data.get('J'),
                    'Breakout_Pattern': (df_with_indicators.iloc[-1]['Close'] > df_with_indicators.iloc[-1]['Max_High_Prev_20'] * 1.005) if 'Max_High_Prev_20' in df_with_indicators.columns else False,
                    'Limit_Up_Today': is_limit_up_today,
                }
                results.append(result_row)

        except Exception as e:
            # print(f"❌ 处理 {code} ({stock_name}) 时发生最终错误: {e}")
            continue

    return pd.DataFrame(results)

# --- save_results 函数 (V4.0 版本号更新) (保持不变) ---
def save_results(df, now):
    """保存结果，并按照策略优先级和得分进行排序"""
    output_dir_date = now.strftime("%Y%m%d")
    output_path_dir = os.path.join(OUTPUT_DIR_BASE, output_dir_date)
    timestamp = now.strftime("%Y%m%d_%H%M%S")

    # 评分逻辑 (V4.0 固化)
    score_A = (df['TurnoverRate'] * df['KDJ_J']) / (df['RSI6'] + 1)
    score_B = df['RSI6'] * df['TurnoverRate']
    score_C = df['RSI6'] * df['TurnoverRate'] * 1.6
    score_D = df['RSI6'] * df['TurnoverRate'] * 1.1
    score_E = df['RSI6'] * df['TurnoverRate'] * 1.7

    df['Final_Score'] = np.select(
        [df['Strategy_Type'].str.contains('C_New_Strategy'),
         df['Strategy_Type'].str.contains('E_Leader_Restart'),
         df['Strategy_Type'].str.contains('D_Strong_Breakout'),
         df['Strategy_Type'].str.contains('A_Strong_Pullback')],
        [score_C, score_E, score_D, score_A],
        default=score_B
    )

    # 风险优化：策略优先级 C (0) > A (1) > B (2) > E (3) > D (4)
    df['Strategy_Rank'] = np.select(
        [df['Strategy_Type'].str.contains('C_New_Strategy'),
         df['Strategy_Type'].str.contains('A_Strong_Pullback'),
         df['Strategy_Type'].str.contains('B_Low_Position_Start'),
         df['Strategy_Type'].str.contains('E_Leader_Restart'),
         df['Strategy_Type'].str.contains('D_Strong_Breakout')],
        [0, 1, 2, 3, 4],
        default=5
    )
    # 排序：先按等级升序 (0->4)，再按得分降序
    df.sort_values(by=['Strategy_Rank', 'Final_Score'], ascending=[True, False], inplace=True)

    output_filename_csv = f"combined_results_5strategy_V4_0_NakedK_Volume_System_{timestamp}.csv"
    output_full_path_csv = os.path.join(output_path_dir, output_filename_csv)

    os.makedirs(output_path_dir, exist_ok=True)
    df.to_csv(output_full_path_csv, index=False, encoding='utf-8')
    print(f"\n✨ 结果已成功保存到 CSV (五策略 V4.0 极简量价交易系统：C>A>B>E>D): {output_full_path_csv}")

    # --- 生成 TXT 候选股清单 (更新版本号) ---
    output_filename_txt = f"candidate_list_5strategy_V4_0_NakedK_Volume_System_{timestamp}.txt"
    output_full_path_txt = os.path.join(output_path_dir, output_filename_txt)

    def format_row(row):
        is_up = " (涨停!)" if row['Limit_Up_Today'] else ""
        breakout = " [形态突破✔]" if row['Breakout_Pattern'] else ""
        strategy_display = row['Strategy_Type'].split(' ')[0]
        return f"[{strategy_display}] {row['code']} - {row['name']}{is_up}{breakout} (收盘: {row['Close']:.2f}, 换手率: {row['TurnoverRate']:.2f}%, RSI6: {row['RSI6']:.1f}, J: {row['KDJ_J']:.1f}, 得分: {row['Final_Score']:.2f})"

    top_candidates = df.head(TOP_CANDIDATES_COUNT)
    top_list_str = "\n".join([format_row(row) for index, row in top_candidates.iterrows()])
    remaining_candidates = df.iloc[TOP_CANDIDATES_COUNT:]
    remaining_list_str = "\n" + "\n".join([format_row(row) for index, row in remaining_candidates.iterrows()])


    header = f"--- 📈 候选股清单 (五策略 V4.0 极简量价交易系统：C>A>B>E>D) ({now.strftime('%Y-%m-%d %H:%M:%S')}) ---\n"
    header += f"总计：{len(df)} 只股票符合任一策略信号。\n"
    header += f"C(共振):{len(df[df['Strategy_Type'].str.contains('C_New_Strategy')])} | A(接力):{len(df[df['Strategy_Type'].str.contains('A_Strong_Pullback')])} | B(埋伏):{len(df[df['Strategy_Type'].str.contains('B_Low_Position_Start')])} | E(二次):{len(df[df['Strategy_Type'].str.contains('E_Leader_Restart')])} | D(突破):{len(df[df['Strategy_Type'].str.contains('D_Strong_Breakout')])}\n\n"


    with open(output_full_path_txt, 'w', encoding='utf-8') as f:
        f.write(header)
        f.write(f"--- 🥇 核心候选股 (TOP {TOP_CANDIDATES_COUNT}：按策略优先级 C>A>B>E>D 排序) ---\n")
        f.write(top_list_str)

        if len(remaining_candidates) > 0:
            f.write("\n\n--- 🥈 补充候选股 (符合任一策略，可进一步观察) ---\n")
            f.write(remaining_list_str)

        f.write("\n\n--- 纪律口号 (V4.0 极简量价交易纪律) ---\n")
        f.write("入场前：聚焦 **裸K平台突破** 和 **量价共振**，只做高确定性信号。尽量把自己变成机器人！\n")
        f.write("入场后：浮盈不设上限，单只亏损止损纪律不超15%。严格过滤高位钝化风险。\n")

    print(f"📜 候选股清单已生成: {output_full_path_txt}")

    return output_full_path_csv


# --- 主程序逻辑 (已修改) ---
if __name__ == "__main__":

    print("--- 启动股票技术分析：五策略 (C>A>B>E>D) 终极 V4.0 极简量价交易系统 ---")
    
    # NEW: 加载名称映射
    name_map = load_name_map()

    # V4.0 全量分析模式：不再需要输入信号文件
    # input_file_path = find_input_signal_path() 
    # if input_file_path is None:
    #     print("\n⚠️ 严格路径检查失败：未在 'combined_results/{date}/' 结构中找到最近的 'combined_buy_signals.csv'，流程终止。")
    #     # exit(0)

    now_shanghai = get_current_shanghai_time()

    # 直接传入 STOCK_DATA_DIR 和 name_map 进行全量分析
    results_df = analyze_and_filter_stocks(STOCK_DATA_DIR, name_map)

    if results_df.empty:
        print("\n⚠️ 没有股票符合任何策略筛选条件，流程终止。")
        # exit(0)
    else:
        save_results(results_df, now_shanghai)
