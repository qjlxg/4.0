import os
import pandas as pd
import glob
from datetime import datetime
from multiprocessing import Pool, cpu_count

def process_full_history(args):
    code, signal_price, signal_type, signal_date_str, stock_names_dict = args
    code_str = str(code).zfill(6)
    file_path = f"stock_data/{code_str}.csv"
    
    if not os.path.exists(file_path): return None

    try:
        df = pd.read_csv(file_path)
        if df.empty: return None
        
        # 匹配信号日期
        formatted_sig_date = f"{signal_date_str[:4]}-{signal_date_str[4:6]}-{signal_date_str[6:]}"
        sig_index_list = df[df['日期'] == formatted_sig_date].index
        
        if sig_index_list.empty:
            later_df = df[df['日期'] > formatted_sig_date]
            if later_df.empty: return None
            idx = later_df.index[0] - 1
        else:
            idx = sig_index_list[0]

        after_signal = df.iloc[idx + 1:].copy()
        if after_signal.empty: return None

        daily_returns = []
        trend_symbols = []
        first_win_day = None
        first_loss_day = None

        for i, (_, row) in enumerate(after_signal.iterrows(), 1):
            change = ((row['收盘'] - signal_price) / signal_price) * 100
            daily_returns.append(change)
            trend_symbols.append("+" if change > 0 else "-")
            
            if change > 0 and first_win_day is None: first_win_day = i
            if change < 0 and first_loss_day is None: first_loss_day = i

        current_ret = daily_returns[-1]
        hold_days = len(after_signal)

        # --- 风险预警逻辑 ---
        warnings = []
        # 1. T+3 尚未回正预警
        if hold_days >= 3 and first_win_day is None:
            warnings.append("长期未回正")
        # 2. 深度套牢预警 (亏损超过 10%)
        if current_ret < -10:
            warnings.append("深度套牢")
        # 3. 冲高回落预警 (曾获利5%以上但当前亏损)
        max_ever = max(daily_returns)
        if max_ever > 5 and current_ret < 0:
            warnings.append("冲高回落")

        return {
            "代码": code_str, "名称": stock_names_dict.get(code_str, "未知"),
            "信号日期": signal_date_str, "策略类型": signal_type, "信号价": round(signal_price, 2),
            "当前总涨跌": current_ret, "平均表现": sum(daily_returns) / hold_days,
            "首次盈利天数": first_win_day if first_win_day is not None else "从未",
            "首次亏损天数": first_loss_day if first_loss_day is not None else "从未",
            "持仓天数": hold_days, "走势流": "".join(trend_symbols),
            "风险预警": " | ".join(warnings) if warnings else "正常",
            "是否盈利": 1 if current_ret > 0 else 0,
            "是否有风险": 1 if warnings else 0
        }
    except Exception:
        return None

def main():
    stock_names_dict = {}
    if os.path.exists('stock_names.csv'):
        names_df = pd.read_csv('stock_names.csv')
        stock_names_dict = dict(zip(names_df['code'].astype(str).str.zfill(6), names_df['name']))

    search_pattern = os.path.join("combined_results", "**", "combined_results_5strategy_V4_0_NakedK_Volume_System*.csv")
    signal_files = glob.glob(search_pattern, recursive=True)
    tasks = []
    for f in signal_files:
        path_parts = os.path.normpath(f).split(os.sep)
        sig_date = next((p for p in path_parts if p.isdigit() and len(p) == 8), "未知")
        try:
            df_sig = pd.read_csv(f)
            for _, row in df_sig.iterrows():
                tasks.append((str(row['code']).zfill(6), float(row['Close']), row['Strategy_Type'], sig_date, stock_names_dict))
        except: continue

    with Pool(cpu_count()) as pool:
        results = pool.map(process_full_history, tasks)
    
    final_list = [r for r in results if r is not None]
    if not final_list: return

    df_main = pd.DataFrame(final_list)

    # --- 生成汇总报表 (含风险统计) ---
    summary = df_main.groupby('策略类型').agg(
        信号总数=('代码', 'count'),
        平均收益=('当前总涨跌', 'mean'),
        盈利个数=('是否盈利', 'sum'),
        预警个数=('是否有风险', 'sum')
    ).reset_index()
    
    summary['胜率'] = (summary['盈利个数'] / summary['信号总数'] * 100).map(lambda x: f"{x:.2f}%")
    summary['风险率'] = (summary['预警个数'] / summary['信号总数'] * 100).map(lambda x: f"{x:.2f}%")
    summary['平均收益'] = summary['平均收益'].map(lambda x: f"{x:.2f}%")
    summary = summary.sort_values(by='胜率', ascending=False)
    
    # --- 格式化主报表 ---
    df_main['当前总涨跌'] = df_main['当前总涨跌'].map(lambda x: f"{x:.2f}%")
    df_main['平均表现'] = df_main['平均表现'].map(lambda x: f"{x:.2f}%")

    # 保存文件
    df_main.drop(columns=['是否盈利', '是否有风险']).to_csv("strategy_optimized_tracker.csv", index=False, encoding='utf-8-sig')
    summary.to_csv("strategy_summary_stats.csv", index=False, encoding='utf-8-sig')
    print("分析完成：包含风险预警标注。")

if __name__ == "__main__":
    main()
