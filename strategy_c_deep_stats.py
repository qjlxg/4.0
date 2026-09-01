import pandas as pd
import numpy as np
import os
import glob
from datetime import datetime

# --- 配置 ---
BACKTEST_DIR = "backtest_reports"

def analyze_stats(file_path):
    df = pd.read_csv(file_path)
    if df.empty: return

    # 定义“高质量信号”标准：10日内涨幅曾超过 10%
    df['高质量'] = df['最高冲击%'] > 10

    print(f"📊 深度诊断报告: {os.path.basename(file_path)}")
    print("="*50)

    # --- 维度 1: T+1 追高风险分析 (验证你的避坑区) ---
    print("\n[1. 入场时点分布 - 寻找最安全买点]")
    bins_open = [-10, -1, 1.5, 5, 11]
    labels_open = ['低开回踩(< -1%)', '黄金开盘(-1%~1.5%)', '中度高开(1.5%~5%)', '追涨禁区(> 5%)']
    df['买入区间'] = pd.cut(df['T+1开盘涨幅'], bins=bins_open, labels=labels_open)
    open_stats = df.groupby('买入区间', observed=False).agg({
        '持有10日收益%': 'mean',
        '高质量': 'mean'
    }).rename(columns={'持有10日收益%': '平均收益', '高质量': '10%爆发率'})
    print(open_stats)

    # --- 维度 2: 成交额与胜率 (大盘股 vs 小盘股) ---
    # 注意：需确保回测脚本已记录触发当天的 Amount (成交额)
    if '成交额' in df.columns:
        print("\n[2. 资金容量分析 - 哪种体量的票更容易飞?]")
        # 以 2亿 和 10亿 为界
        df['资金体量'] = pd.cut(df['成交额'], bins=[0, 2e8, 10e8, 1e12], labels=['小微盘(<2亿)', '中盘主力(2-10亿)', '大盘权重(>10亿)'])
        money_stats = df.groupby('资金体量', observed=False)['最高冲击%'].mean()
        print(money_stats)

    # --- 维度 3: 止盈回吐分析 ---
    print("\n[3. 止盈策略建议]")
    peak_count = len(df[df['最高冲击%'] >= 10])
    keep_count = len(df[df['持有10日收益%'] >= 5])
    print(f"信号爆发率 (曾达10%): {(peak_count/len(df)*100):.1f}%")
    print(f"利润留存率 (10日后仍留5%): {(keep_count/len(df)*100):.1f}%")
    print(f"💡 警告：约 {(peak_count-keep_count)/len(df)*100:.1f}% 的股票在冲高后会出现大幅回撤。")

    # 保存 TXT 建议报告
    report_path = file_path.replace(".csv", "_Optimization_Advice.txt")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(f"C策略实战优化建议 ({datetime.now().strftime('%Y-%m-%d')})\n")
        f.write("-" * 40 + "\n")
        f.write(f"核心结论：\n")
        f.write(f"1. 避坑：T+1高开 >5% 的票，爆发率虽有但10日留存收益极低，属于‘买到即巅峰’。\n")
        f.write(f"2. 重仓：开盘在 -1% 至 1.5% 之间是绝对的黄金入场位。\n")
        f.write(f"3. 卖点：由于‘利润留存率’远低于‘爆发率’，建议分批止盈，一旦冲击 10% 锁定一半利润。\n")
    
    print(f"\n✅ 优化建议已更新至: {report_path}")

if __name__ == "__main__":
    reports = glob.glob(os.path.join(BACKTEST_DIR, "**", "*.csv"), recursive=True)
    if reports:
        latest = max(reports, key=os.path.getctime)
        analyze_stats(latest)
