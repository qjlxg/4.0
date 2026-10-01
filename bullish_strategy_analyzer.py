import os
import pandas as pd
import glob
from datetime import datetime
import multiprocessing
from concurrent.futures import ProcessPoolExecutor

def process_single_stock(file_path, signal_dict):
    stock_code = os.path.basename(file_path).replace('.csv', '').zfill(6)
    if stock_code not in signal_dict:
        return None
    
    try:
        df = pd.read_csv(file_path)
        if df.empty: return None
        df['日期'] = pd.to_datetime(df['日期']).dt.strftime('%Y-%m-%d')
        df = df.sort_values('日期')
        
        results = []
        for _, sig in signal_dict[stock_code].iterrows():
            sig_date = sig['Latest_Date']
            sig_price = sig['Latest_Close']
            
            # 提取信号日之后的所有交易记录
            track_df = df[df['日期'] > sig_date].copy()
            if track_df.empty: continue
            
            # 关键表现数据
            latest_price = track_df.iloc[-1]['收盘']
            max_price = track_df['收盘'].max()
            
            # 计算收益
            curr_return = ((latest_price - sig_price) / sig_price) * 100
            max_return = ((max_price - sig_price) / sig_price) * 100
            
            results.append({
                '代码': stock_code,
                '信号日期': sig_date,
                '信号价格': sig_price,
                '当前价格': latest_price,
                '当前收益%': round(curr_return, 2),
                '最高收益%': round(max_return, 2),
                '是否获利': 1 if curr_return > 0 else 0, # 用于计算胜率
                '持有天数': len(track_df),
                '触发条件': sig['Matched_Conditions']
            })
        return results
    except:
        return None

def main():
    multiprocessing.set_start_method('spawn', force=True)
    print(f"[{datetime.now()}] 启动策略效能分析...")

    # 1. 动态获取信号文件
    curr_month = datetime.now().strftime("%Y-%m")
    sig_files = glob.glob(f"scan_results/{curr_month}/**/bullish_scan_OR_logic_COMPREHENSIVE*.csv", recursive=True)
    if not sig_files:
        print("未找到当月信号文件。")
        return

    # 2. 预处理信号数据
    all_sig = pd.concat([pd.read_csv(f, dtype={'Code': str}) for f in sig_files]).drop_duplicates()
    all_sig['Code'] = all_sig['Code'].str.zfill(6)
    all_sig['Latest_Date'] = pd.to_datetime(all_sig['Latest_Date']).dt.strftime('%Y-%m-%d')
    sig_dict = {code: group for code, group in all_sig.groupby('Code')}

    # 3. 筛选相关的股票数据文件进行并行处理
    all_files = glob.glob("stock_data/*.csv")
    relevant_files = [f for f in all_files if os.path.basename(f).replace('.csv', '').zfill(6) in sig_dict]
    
    analysis_list = []
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(process_single_stock, f, sig_dict) for f in relevant_files]
        for f in futures:
            res = f.result()
            if res: analysis_list.extend(res)

    if analysis_list:
        df_detail = pd.DataFrame(analysis_list)
        
        # 4. 生成策略汇总表 (Strategy Summary)
        # 统计：信号次数、平均收益、最高收益、胜率
        summary = df_detail.groupby('触发条件').agg({
            '代码': 'count',
            '当前收益%': 'mean',
            '最高收益%': 'mean',
            '是否获利': 'mean' # 0-1 之间的平均值即为胜率
        }).rename(columns={
            '代码': '出现次数',
            '当前收益%': '平均收益%',
            '最高收益%': '历史最高涨幅平均%',
            '是否获利': '胜率'
        })
        
        # 格式化胜率显示
        summary['胜率'] = (summary['胜率'] * 100).round(2).astype(str) + '%'
        summary = summary.sort_values('平均收益%', ascending=False)

        # 5. 保存结果
        timestamp = datetime.now().strftime('%Y%m%d')
        summary_file = f"strategy_performance_summary_{timestamp}.csv"
        detail_file = f"signal_final_status_{timestamp}.csv"
        
        summary.to_csv(summary_file, encoding='utf-8-sig')
        df_detail.to_csv(detail_file, index=False, encoding='utf-8-sig')
        
        print(f"分析完成！\n汇总表已生成: {summary_file} (查看哪种条件最赚钱)\n详情表已生成: {detail_file} (查看每只股票最终盈亏)")
    else:
        print("未匹配到任何有效行情数据。")

if __name__ == "__main__":
    main()
