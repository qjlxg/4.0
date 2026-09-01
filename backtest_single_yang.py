import pandas as pd
import numpy as np
import os
import glob
from multiprocessing import Pool, cpu_count
from itertools import product

# ==================== 核心配置 ====================
PARAM_GRID = {
    'ma_period': [10, 20, 60],
    'vol_up_ratio': [1.5, 2.0, 3.0],
    'vol_reduce_ratio': [0.3, 0.5], # 聚焦旧版最强的两个区间
    'support_level': [0.382, 0.5, 0.618]
}
HOLD_DAYS = [3, 5, 10, 20]
# =================================================

def fast_engine(file_path):
    try:
        # 仅读取必要的列，减少内存占用
        df = pd.read_csv(file_path, usecols=['日期', '开盘', '收盘', '最高', '最低', '成交量', '涨跌幅'])
        df.columns = df.columns.str.strip()
        
        # 基础过滤：剔除数据量不足或价格异常的
        if len(df) < 70: return []
        
        # 预计算所有均线，转为 NumPy 提高速度
        c = df['收盘'].values
        ma_dict = {m: df['收盘'].rolling(m).mean().values for m in PARAM_GRID['ma_period']}
        slope_dict = {m: np.gradient(ma_dict[m]) for m in PARAM_GRID['ma_period']} # 使用梯度计算斜率，极快
        
        o, v, h, l, pct = df['开盘'].values, df['成交量'].values, df['最高'].values, df['最低'].values, df['涨跌幅'].values
        
        # 寻找种子阳线 (5% - 11%)
        yang_indices = np.where((pct >= 5.0) & (pct <= 11.0))[0]
        
        signals = []
        for y_idx in yang_indices:
            # 边界检查
            if y_idx < 1 or y_idx > len(c) - 25: continue
            
            # 1. 阳线纯度过滤 (上影线不能太长)
            body = c[y_idx] - o[y_idx]
            if body <= 0 or (h[y_idx] - c[y_idx]) > body * 0.4: continue
            
            # 2. 检查回调期 (1-6天)
            for gap in range(1, 7):
                entry_idx = y_idx + gap
                if entry_idx >= len(c) - 20: break
                
                # 提取回调区间数据
                adj_v = v[y_idx+1 : entry_idx+1]
                adj_c = c[y_idx+1 : entry_idx+1]
                
                # 记录核心特征向量
                res = [
                    pct[y_idx], gap, v[y_idx], v[y_idx-1], np.mean(adj_v), np.min(adj_c),
                    o[y_idx], c[y_idx], c[entry_idx]
                ]
                
                # 压入收益率 (扣除万三摩擦成本)
                for d in HOLD_DAYS:
                    ret = (c[entry_idx+d] - c[entry_idx]*1.0006) / (c[entry_idx]*1.0006)
                    res.append(ret)
                
                # 压入各周期均线与斜率
                for m in PARAM_GRID['ma_period']:
                    res.append(ma_dict[m][entry_idx])
                    res.append(slope_dict[m][entry_idx])
                
                signals.append(res)
        return signals
    except: return []

def run_analysis():
    files = glob.glob('stock_data/*.csv')
    print(f"📡 正在极速扫描 {len(files)} 个文件...")
    
    with Pool(cpu_count()) as pool:
        raw = pool.map(fast_engine, files)
    
    all_sigs = np.array([s for sub in raw if sub for s in sub])
    if all_sigs.size == 0: return print("❌ 未发现符合条件的信号")
    
    print(f"🎯 提取完成，开始矩阵交叉匹配 (总信号: {len(all_sigs)})")
    
    # 定义列索引
    # 0:pct, 1:gap, 2:y_v, 3:p_v, 4:adj_v, 5:adj_min, 6:y_o, 7:y_c, 8:entry_c, 9-12:rets, 13+:ma_slope
    report = []
    combs = list(product(PARAM_GRID['ma_period'], PARAM_GRID['vol_up_ratio'], PARAM_GRID['vol_reduce_ratio'], PARAM_GRID['support_level']))

    for ma_p, v_up, v_red, s_lev in combs:
        ma_idx = 13 + PARAM_GRID['ma_period'].index(ma_p) * 2
        slp_idx = ma_idx + 1
        
        # NumPy 向量化筛选 (全速运行)
        mask = (all_sigs[:, 8] >= all_sigs[:, ma_idx]) & (all_sigs[:, slp_idx] > 0) # 趋势过滤
        mask &= (all_sigs[:, 2] >= all_sigs[:, 3] * v_up)     # 倍量过滤
        mask &= (all_sigs[:, 4] <= all_sigs[:, 2] * v_red)    # 缩量过滤
        
        # 支撑位判定
        support_p = all_sigs[:, 6] + (all_sigs[:, 7] - all_sigs[:, 6]) * (1 - s_lev)
        mask &= (all_sigs[:, 5] >= support_p)
        
        filtered = all_sigs[mask]
        if len(filtered) > 50:
            r5 = filtered[:, 10].astype(float) # 5日收益
            report.append({
                'MA': ma_p, '倍量': v_up, '缩量': v_red, '支撑': s_lev,
                '次数': len(filtered),
                '5d均盈': f"{np.mean(r5)*100:.2f}%",
                '5d胜率': f"{(r5 > 0).mean():.1%}",
                '盈亏比': round(np.mean(r5[r5>0]) / abs(np.mean(r5[r5<=0])), 2) if any(r5<=0) else 0
            })

    result_df = pd.DataFrame(report).sort_values('5d均盈', ascending=False)
    print("\n" + "="*80)
    print("🏆 极致性能版：成熟参数回测榜单")
    print(result_df.head(20).to_string(index=False))
    result_df.to_csv("Hyper_Optimization_Report.csv", index=False)

if __name__ == '__main__':
    run_analysis()
