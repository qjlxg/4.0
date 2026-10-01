import pandas as pd
import glob
import os

def solve_strong():
    # 载入名称
    names = pd.read_csv('stock_names.csv', dtype={'code': str})
    name_map = dict(zip(names['code'].str.zfill(6), names['name']))
    files = glob.glob('stock_data/*.csv')
    results = []

    print(f"--- 正在执行：单阳涨停 + 1-5天缩量调整 + 5/10日线支撑 ---")

    for f in files:
        try:
            df = pd.read_csv(f)
            df.columns = df.columns.str.strip()
            df = df.sort_values('日期').reset_index(drop=True)
            
            # 计算均线
            df['MA5'] = df['收盘'].rolling(window=5).mean()
            df['MA10'] = df['收盘'].rolling(window=10).mean()
            
            last_idx = len(df) - 1
            curr_row = df.iloc[last_idx]
            code = str(curr_row['股票代码']).split('.')[0].zfill(6)
            name = name_map.get(code, "未知")

            # 基础过滤 (5-20元)
            if not (5.0 <= curr_row['收盘'] <= 20.0): continue

            # --- 核心逻辑：1-5天洗盘 ---
            for gap in [1, 2, 3, 4, 5]:
                yang_idx = last_idx - gap
                if yang_idx < 0: continue
                
                row_yang = df.iloc[yang_idx]
                
                # 1. 涨停判定
                if row_yang['涨跌幅'] >= 9.8:
                    adjust_period = df.iloc[yang_idx + 1:]
                    
                    # 2. 严格缩量
                    cond_vol = (adjust_period['成交量'] < row_yang['成交量']).all()
                    
                    # 3. 价格不破单阳最低
                    cond_price = (adjust_period['收盘'] >= row_yang['最低']).all()
                    
                    # 4. 均线支撑 (今天必须踩在5日或10日线上)
                    # 只要大于其中一个，就认为支撑有效
                    cond_ma = (curr_row['收盘'] >= curr_row['MA5']) or (curr_row['收盘'] >= curr_row['MA10'])

                    if cond_vol and cond_price and cond_ma:
                        results.append({
                            '代码': code, '名称': name, '最后日期': curr_row['日期'],
                            '涨停日': row_yang['日期'], '调整天数': gap, '现价': curr_row['收盘']
                        })
                        break
        except:
            continue

    res_df = pd.DataFrame(results)
    if not res_df.empty:
        print(f"\n✅ 匹配成功 ({len(res_df)}只)：")
        print(res_df.sort_values(by='调整天数'))
        res_df.to_csv("Strong_Single_Yang.csv", index=False, encoding='utf_8_sig')
    else:
        print("\n❌ 今日无符合条件的强势股。")

if __name__ == '__main__':
    solve_strong()
