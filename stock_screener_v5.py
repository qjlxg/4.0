import pandas as pd
import glob
import os
import numpy as np
from datetime import datetime
import pytz
import logging
import math
from concurrent.futures import ProcessPoolExecutor
import warnings
warnings.simplefilter(action='ignore')

# ============================= 配置区 (已调整) =============================
STOCK_DATA_DIR = 'stock_data'           # 所有历史数据的目录
STOCK_NAMES_FILE = 'stock_names.csv'  # 代码-名称映射表
MIN_MONTH_DRAWDOWN = 0.06             # 【收紧】近1月回撤 ≥6% (原 4%) 才进入候选池
HIGH_ELASTICITY_MIN_DRAWDOWN = 0.10   # 【收紧】高弹性回撤阈值 (原 0.15, 实际为了配合 4.0分这里改为 0.10)
MIN_DAILY_DROP_PERCENT = 0.03
EXTREME_RSI_THRESHOLD_P1 = 29.0         # 【收紧】RSI(14)极值阈值 (原 32.0)
STRONG_RSI_THRESHOLD_P2 = 32.0          # 【收紧】用于高亮RSI的阈值 (原 35.0)
SHORT_TERM_RSI_EXTREME = 18.0           # 【收紧】RSI(6)极值阈值 (原 20.0)
KDJ_J_EXTREME_THRESHOLD = 5.0           # 【收紧】KDJ(J)极值确认阈值 (原 10.0)
TREND_HEALTH_THRESHOLD = 0.90           # 【收紧】趋势健康度 (MA50/MA250) (原 0.85)
MIN_BUY_SIGNAL_SCORE = 4.0            # 【收紧】最低买入信号分数 (原 3.7)
TREND_SLOPE_THRESHOLD = 0.005
REPORT_BASE_NAME = 'stock_report_v5_strict'

# >>> 价格限制配置 (保持不变) <<<
MIN_STOCK_PRICE = 5.0                 # 股票最低价格限制
MAX_STOCK_PRICE = 28.0                # 股票最高价格限制
# ==================================================================

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('stock_screener.log', encoding='utf-8'),
            logging.StreamHandler()
        ]
    )

# 加载代码→名称映射表 (保持不变)
def load_stock_names():
    try:
        df = pd.read_csv(STOCK_NAMES_FILE, dtype=str)
        df.columns = df.columns.str.replace('\ufeff', '')
        return dict(zip(df['code'].str.zfill(6), df['name']))
    except Exception as e:
        logging.error(f"加载 stock_names.csv 失败: {e}")
        return {}

# 加载并预处理单个股票CSV (保持不变)
def load_and_preprocess_data(filepath, code):
    try:
        df = pd.read_csv(filepath, dtype={'股票代码': str})
        df = df.rename(columns={'日期': 'date', '收盘': 'value', '最高': 'high', '最低': 'low'})
        df['date'] = pd.to_datetime(df['date'])
        
        df = df[['date', 'value', 'high', 'low']].sort_values('date').reset_index(drop=True)
        
        if len(df) < 60:
            return None, "数据不足60天"
        if (df['value'] <= 0).any():
            return None, "存在无效价格"
        return df, "OK"
    except Exception as e:
        return None, f"读取错误: {e}"

# 布林带 (保持不变)
def calculate_bollinger_bands(series, window=20):
    if len(series) < window:
        return "数据不足"
    rolling_mean = series.rolling(window=window).mean()
    rolling_std  = series.rolling(window=window).std()
    upper = rolling_mean + rolling_std * 2
    lower = rolling_mean - rolling_std * 2
    latest_value = series.iloc[-1]
    latest_upper = upper.iloc[-1]
    latest_lower = lower.iloc[-1]
    
    if latest_value <= latest_lower:
        return "**下轨下方**"
    elif latest_value >= latest_upper:
        return "**上轨上方**"
    else:
        range_val = latest_upper - latest_lower
        if range_val <= 0:
            return "轨道中间"
        
        pos = (latest_value - latest_lower) / range_val
        if pos < 0.2:
            return "下轨附近"
        elif pos > 0.8:
            return "上轨附近"
        else:
            return "轨道中间"

# 技术指标 (保持不变)
def calculate_technical_indicators(df):
    if len(df) < 60:
        return {'RSI(14)': np.nan, 'RSI(6)': np.nan, 'MACD信号': '数据不足',
                '净值/MA50': np.nan, 'MA50/MA250': np.nan,
                'MA50/MA250趋势': '数据不足', '布林带位置': '数据不足',
                '最新价格': np.nan, '当日跌幅': np.nan, 'KDJ(J)': np.nan}
    
    close = df['value']
    delta = close.diff()
    
    # RSI
    for period in [14, 6]:
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.ewm(span=period, adjust=False, min_periods=period if len(gain) >= period else 0).mean()
        avg_loss = loss.ewm(span=period, adjust=False, min_periods=period if len(loss) >= period else 0).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        df[f'RSI_{period}'] = 100 - 100 / (1 + rs)
        df[f'RSI_{period}'] = df[f'RSI_{period}'].fillna(100.0)
    
    # KDJ
    kdj_j = np.nan
    if 'high' in df.columns and 'low' in df.columns:
        n_period = 9
        if len(close) >= n_period:
            low_min = df['low'].rolling(window=n_period, min_periods=n_period).min()
            high_max = df['high'].rolling(window=n_period, min_periods=n_period).max()
            rsv = 100 * ((close - low_min) / (high_max - low_min).replace(0, 1e-10))
            df['K'] = rsv.ewm(com=2, adjust=False, min_periods=n_period).mean() 
            df['D'] = df['K'].ewm(com=2, adjust=False, min_periods=n_period).mean()
            df['J'] = 3 * df['K'] - 2 * df['D']
            if not np.isnan(df['J'].iloc[-1]):
                kdj_j = round(df['J'].iloc[-1], 2)

    # MACD
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    macd_signal = '观察'
    if len(macd) >= 2:
        macd_prev, signal_prev = macd.iloc[-2], signal.iloc[-2]
        golden = (macd.iloc[-1] > signal.iloc[-1]) and (macd_prev <= signal_prev)
        death  = (macd.iloc[-1] < signal.iloc[-1]) and (macd_prev >= signal_prev)
        if golden:
            macd_signal = '强势金叉' if macd.iloc[-1] > 0 else '弱势金叉'
        elif death:
            macd_signal = '死叉'
    
    # 均线
    ma50  = close.rolling(50, min_periods=1).mean()
    ma250 = close.rolling(250, min_periods=1).mean()
    net_to_ma50 = close.iloc[-1] / ma50.iloc[-1]
    ma_ratio = ma50.iloc[-1] / ma250.iloc[-1] if len(df) >= 250 else np.nan
    
    # 趋势方向
    trend = '数据不足'
    if len(df) >= 250:
        recent_ratio = (ma50 / ma250).tail(50).dropna()
        if len(recent_ratio) >= 5:
            if len(recent_ratio) > 1:
                slope = np.polyfit(np.arange(len(recent_ratio)), recent_ratio.values, 1)[0]
                if slope > TREND_SLOPE_THRESHOLD:
                    trend = '向上'
                elif slope < -TREND_SLOPE_THRESHOLD:
                    trend = '向下'
                else:
                    trend = '平稳'
            else:
                 trend = '平稳' 
    
    daily_drop = (close.iloc[-1] - close.iloc[-2]) / close.iloc[-2] if len(close) >= 2 else 0
    
    return {
        'RSI(14)': round(df['RSI_14'].iloc[-1], 2),
        'RSI(6)':  round(df['RSI_6'].iloc[-1],  2),
        'KDJ(J)': kdj_j,
        'MACD信号': macd_signal,
        '净值/MA50': round(net_to_ma50, 3),
        'MA50/MA250': round(ma_ratio, 3) if not np.isnan(ma_ratio) else np.nan,
        'MA50/MA250趋势': trend,
        '布林带位置': calculate_bollinger_bands(close),
        '最新价格': round(close.iloc[-1], 3),
        '当日跌幅': round(daily_drop, 4)
    }

# 连续下跌天数 (保持不变)
def calculate_consecutive_drops(series_tail10):
    drops = (series_tail10.diff() < 0).values[1:]
    cnt = 0
    for d in reversed(drops):
        if d:
            cnt += 1
        else:
            break
    return cnt

# 最大回撤（近1月） (保持不变)
def calculate_recent_month_drawdown(df):
    latest_date = df['date'].iloc[-1]
    one_month_ago = latest_date - pd.DateOffset(months=1)
    recent = df[df['date'] >= one_month_ago]['value']
    if len(recent) < 2:
        return 0.0
    peak = recent.cummax()
    drawdown = (peak - recent) / peak
    return drawdown.max()

# V5.0 行动信号 (已修改：信号名称和逻辑)
def generate_v5_action_signal(row):
    signals = []
    rsi14 = row['RSI(14)']
    rsi6  = row['RSI(6)']
    kdj_j = row.get('KDJ(J)', 100.0) 
    daily = row['当日跌幅']
    mdd   = row['最大回撤']
    boll  = row['布林带位置']
    macd  = row['MACD信号']
    consec = row['近10日连跌']
    
    # RSI 极值确认 (EXTREME_RSI_THRESHOLD_P1=29.0)
    if rsi14 <= EXTREME_RSI_THRESHOLD_P1:
        
        is_rsi6_extreme = (rsi6 <= SHORT_TERM_RSI_EXTREME) # 18.0
        is_kdj_extreme = (kdj_j <= KDJ_J_EXTREME_THRESHOLD) # 5.0
        is_daily_panic = (daily <= -MIN_DAILY_DROP_PERCENT)
        
        # Priority 1: 顶级共振/确认 (要求更严格，最高分 4.5)
        if is_rsi6_extreme:
            signals.append(f"💥【顶级共振】RSI共振({rsi14:.1f}/{rsi6:.1f})")
        elif is_kdj_extreme: 
             signals.append(f"💥【顶级共振】RSI+KDJ极值({rsi14:.1f}/{kdj_j:.1f})")
        elif is_daily_panic:
            signals.append(f"💥【顶级共振】RSI极值+恐慌({rsi14:.1f})")
        
        # Priority 2: 仅 RSI(14) 极值 (降级为加强级，对应 4.0分)
        else:
            signals.append(f"🌟【加强级】RSI极值({rsi14:.1f})")

    # 震荡策略信号 (MIN_MONTH_DRAWDOWN=0.06, HIGH_ELASTICITY_MIN_DRAWDOWN=0.10)
    if mdd >= MIN_MONTH_DRAWDOWN:
        if consec >= 5:
            signals.append("✨【震荡-连跌】连跌5日+高回撤") # 3.5分
        if boll in ["**下轨下方**", "下轨附近"]:
            signals.append("🎯【震荡-高吸】触及BOLL下轨") # 4.0分
        if mdd >= HIGH_ELASTICITY_MIN_DRAWDOWN:
            signals.append("🔥【高弹性】回撤达标") # 4.0分
    
    # MACD 信号
    if macd == '弱势金叉':
        signals.append("🛡️【防御-反弹】MACD弱金叉") # 3.0分
    
    if not signals:
        return '等待信号'
    return ' | '.join(signals)

# 退出/止损信号 (保持不变)
def generate_exit_signal(row):
    if row['最大回撤'] > 0.10:
        return f"🛑 止损：近1月回撤超10% ({row['最大回撤']:.2%})"
    if row['RSI(14)'] > 70:
        return "🚫 止盈：RSI(14)过买"
    if row['MACD信号'] == '死叉':
        return "🚫 止盈/止损：MACD死叉"
    return "持有"

# 单股票分析（核心修改：回撤过滤收紧）
def analyze_single_stock(filepath, names_dict):
    code = os.path.basename(filepath)[:6]
    df, msg = load_and_preprocess_data(filepath, code)
    if df is None:
        return None
    
    # ------------------ 价格过滤 ------------------
    latest_price = df['value'].iloc[-1]
    if not (MIN_STOCK_PRICE <= latest_price <= MAX_STOCK_PRICE):
        return None
    # ----------------------------------------------------
    
    mdd_month = calculate_recent_month_drawdown(df)
    # ------------------ 回撤过滤 (已收紧) ------------------
    if mdd_month < MIN_MONTH_DRAWDOWN: # MIN_MONTH_DRAWDOWN = 0.06
        return None
    # ----------------------------------------------------
    
    tech = calculate_technical_indicators(df)
    consec = calculate_consecutive_drops(df['value'].tail(10))
    
    row = {
        '股票代码': code,
        '股票名称': names_dict.get(code, '未知'),
        '最大回撤': round(mdd_month, 4),
        '近10日连跌': consec,
        **tech,
        '行动提示': '',
        '退出提示': ''
    }
    row['行动提示'] = generate_v5_action_signal(row)
    row['退出提示'] = generate_exit_signal(row)
    return row

# 主分析函数（并行） (保持不变)
def analyze_all_stocks():
    files = glob.glob(os.path.join(STOCK_DATA_DIR, '*.csv'))
    names_dict = load_stock_names()
    
    results = []
    max_workers = os.cpu_count() or 4
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(analyze_single_stock, f, names_dict) for f in files]
        for future in futures:
            res = future.result()
            if res:
                results.append(res)
    
    logging.info(f"共分析完成 {len(files)} 只股票，进入候选池 {len(results)} 只")
    return results, names_dict

# 生成Markdown + CSV报告 (已修改：信号打分和入选标准)
def generate_reports(results, timestamp_str, timestamp_file):
    if not results:
        content = f"# 股票 V5.0 策略选股报告 ({timestamp_str} 上海)\n\n暂无满足条件的股票（近1月回撤≥{MIN_MONTH_DRAWDOWN:.0%}）"
        return content, None
    
    df = pd.DataFrame(results)
    
    # 信号打分
    df['signal_score'] = 0.0
    # 顶级共振信号 4.5 分
    df.loc[df['行动提示'].str.contains('💥【顶级共振】'), 'signal_score'] = 4.5
    # 加强级RSI极值、震荡-高吸、高弹性 4.0 分
    df.loc[df['行动提示'].str.contains('🌟【加强级】RSI极值'), 'signal_score'] = 4.0
    df.loc[df['行动提示'].str.contains('高弹性'), 'signal_score'] = df['signal_score'].clip(upper=4.0)
    df.loc[df['行动提示'].str.contains('震荡-高吸'), 'signal_score'] = df['signal_score'].clip(upper=4.0)
    # 其他信号 3.5 分或 3.0 分
    df.loc[df['行动提示'].str.contains('震荡-连跌'), 'signal_score'] = df['signal_score'].clip(upper=3.5)
    df.loc[df['行动提示'].str.contains('防御-反弹'), 'signal_score'] = df['signal_score'].clip(upper=3.0)
    
    # 趋势健康度 (TREND_HEALTH_THRESHOLD = 0.90)
    df['trend_ok'] = (df['MA50/MA250'] >= TREND_HEALTH_THRESHOLD) & (df['MA50/MA250趋势'] != '向下')
    
    # 排序因子
    df['is_top_tier_signal'] = df['行动提示'].str.contains('💥【顶级共振】').astype(int)
    
    df['final_score'] = (
        df['is_top_tier_signal'] * 1000000 + 
        df['trend_ok'].astype(int) * 1000 + 
        df['最大回撤'] * 10000
    )
    
    # 止损标记
    df['止损否决'] = (df['退出提示'].str.contains('🛑 止损')) | (df['退出提示'].str.contains('🚫 止盈/止损：MACD死叉'))
    
    # 注意：这里使用 MIN_BUY_SIGNAL_SCORE = 4.0
    df_buy = df[(df['trend_ok']) & (df['signal_score'] >= MIN_BUY_SIGNAL_SCORE) & (~df['止损否决'])].copy()
    
    df_buy = df_buy.sort_values('final_score', ascending=False).reset_index(drop=True)
    df_buy['排名'] = range(1, len(df_buy)+1)
    
    # Markdown报告
    lines = [f"# 股票 V5.0 策略选股报告 - **严格版** ({timestamp_str} 上海)\n"]
    lines.append(f"**筛选标准:** 近1月回撤≥{MIN_MONTH_DRAWDOWN:.0%}, 趋势健康度(MA50/MA250)≥{TREND_HEALTH_THRESHOLD:.2f}, 最低信号分≥{MIN_BUY_SIGNAL_SCORE:.1f}\n")
    lines.append(f"**价格筛选范围: [{MIN_STOCK_PRICE}元, {MAX_STOCK_PRICE}元]**\n")
    lines.append(f"共扫描 {len(glob.glob(os.path.join(STOCK_DATA_DIR, '*.csv')))} 只股票，**{len(df_buy)} 只进入最高优先级可试仓池**\n")
    
    if len(df_buy) > 0:
        lines.append("\n## 🥇 最高优先级可试仓标的（顶级信号优先，趋势健康+强信号+未止损）\n")
        lines.append("| 排名 | 代码 | 名称 | 最新价格 | 近1月回撤 | 当日跌幅 | RSI(14) | KDJ(J) | V5.0信号 | 退出提示 | MA健康度 | 试水买价(-3%) |")
        lines.append("|------|--------|------------|----------|-----------|----------|---------|----------|----------------------------|------------|----------|----------------|")
        for _, r in df_buy.iterrows():
            trial_price = round(r['最新价格'] * 0.97, 3)
            # RSI高亮阈值 STRONG_RSI_THRESHOLD_P2 = 32.0
            rsi_display = f"**{r['RSI(14)']:.1f}**" if r['RSI(14)'] <= STRONG_RSI_THRESHOLD_P2 else f"{r['RSI(14)']:.1f}"
            # KDJ高亮阈值 KDJ_J_EXTREME_THRESHOLD = 5.0
            kdj_display = f"**{r['KDJ(J)']:.1f}**" if r['KDJ(J)'] <= KDJ_J_EXTREME_THRESHOLD else f"{r['KDJ(J)']:.1f}"
            
            lines.append(f"| {r['排名']} | `{r['股票代码']}` | {r['股票名称']} | **{r['最新价格']:.2f}** | **{r['最大回撤']:.2%}** | {r['当日跌幅']:+.2%} | {rsi_display} | {kdj_display} | {r['行动提示']} | {r['退出提示']} | {r['MA50/MA250']:.3f}({r['MA50/MA250趋势']}) | `{trial_price}` |")
    
    md_content = "\n".join(lines)
    
    return md_content, df_buy

# 主函数 (保持不变)
def main():
    setup_logging()
    tz = pytz.timezone('Asia/Shanghai')
    now = datetime.now(tz)
    timestamp_str = now.strftime('%Y-%m-%d %H:%M:%S')
    timestamp_file = now.strftime('%Y%m%d_%H%M%S')
    dir_name = now.strftime('%Y%m')
    os.makedirs(dir_name, exist_ok=True)
    
    results, _ = analyze_all_stocks()
    md_content, df_buy = generate_reports(results, timestamp_str, timestamp_file)
    
    # 保存Markdown
    md_path = os.path.join(dir_name, f"{REPORT_BASE_NAME}_{timestamp_file}.md")
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(md_content)
    
    # 保存CSV（仅最高优先级标的）
    if df_buy is not None and len(df_buy) > 0:
        csv_path = os.path.join(dir_name, f"{REPORT_BASE_NAME}_{timestamp_file}.csv")
        df_buy[['排名', '股票代码', '股票名称', '最新价格', '最大回撤', '当日跌幅', 'RSI(14)', 'KDJ(J)',
                '行动提示', '退出提示', 'MA50/MA250', 'MA50/MA250趋势']].to_csv(csv_path, index=False, encoding='utf-8-sig')
    
    logging.info(f"报告已生成：{md_path}" + (f" 和 {csv_path}" if df_buy is not None and len(df_buy)>0 else ""))

if __name__ == '__main__':
    main()
