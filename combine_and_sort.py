import pandas as pd
import glob
import os
from datetime import datetime
import pytz

# --- 新增常量：名称映射文件路径 ---
NAME_MAP_FILE = 'stock_names.csv' 

# --- 新增函数：加载名称映射 ---
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
                        
                        # print(f"成功使用编码'{enc}'和分隔符'{repr(delim)}'加载名称映射。")
                        name_map = df_names.set_index('code')['name'].to_dict()
                        found_map = True
                        break 
                except Exception:
                    continue
        if not name_map:
            print("警告：无法正确解析名称映射文件，将跳过名称映射。")
    else:
        print(f"警告：名称映射文件 '{NAME_MAP_FILE}' 未找到，将跳过名称映射。")
    return name_map

def combine_and_sort_csvs():
    """
    读取 results/ 下的所有 CSV 文件，与已存在的合并文件追加、去重、排序，
    并保存到 combined_results/YYYYMMDD/combined_buy_signals.csv。
    """
    # 设定时区为上海，并获取当前日期作为目录名 (YYYYMMDD)
    shanghai_tz = pytz.timezone('Asia/Shanghai')
    now_shanghai = datetime.now(shanghai_tz)
    date_str = now_shanghai.strftime('%Y%m%d')
    
    # 构造输入目录路径
    input_dir_pattern = 'results/*.csv'
    
    # 构造输出目录路径和文件名
    output_dir = f'combined_results/{date_str}'
    output_file = os.path.join(output_dir, 'combined_buy_signals.csv')
    
    # 定义需要的列名 (保持一致)
    # 注意: 即使原文件没有 '股票名称'，我们也要保留这个列以便后续填充
    required_columns = ['股票代码', '股票名称', '买入信号', '评分', '图表路径']
    
    # 明确指定 '股票代码' 列应作为字符串处理，以保留前导零
    dtype_spec = {'股票代码': str}

    # --- 0. 加载名称映射 ---
    name_map = load_name_map()

    # --- 1. 读取所有新的 CSV 文件 ( results/*.csv ) ---
    all_files = glob.glob(input_dir_pattern)
    list_new_dfs = []
    
    if not all_files:
        print(f"未在 '{input_dir_pattern}' 找到任何 CSV 文件。脚本终止。")
        return

    print(f"找到 {len(all_files)} 个新文件，开始读取...")
    
    for file in all_files:
        try:
            # 读取时，将 '股票代码' 列指定为字符串类型
            df = pd.read_csv(file, dtype=dtype_spec)
            
            # 检查并创建缺失的列，然后只保留需要的列
            for col in required_columns:
                if col not in df.columns:
                    df[col] = '' # 填充空字符串
            
            df = df[required_columns]
            list_new_dfs.append(df)
        except Exception as e:
            print(f"读取文件 {file} 时出错: {e}. 跳过。")

    # 将所有新读取的CSV合并成一个DataFrame
    new_combined_df = pd.concat(list_new_dfs, ignore_index=True)
    
    # 【修复】修正新数据的股票代码，确保它们是6位数，不足的用前导零填充
    print("【修复】正在为新数据中的股票代码添加前导零，确保为6位...")
    if '股票代码' in new_combined_df.columns:
        new_combined_df['股票代码'] = new_combined_df['股票代码'].astype(str).str.zfill(6)


    # --- 2. 读取已存在的合并文件 (实现追加) ---
    existing_df = pd.DataFrame(columns=required_columns)
    if os.path.exists(output_file):
        try:
            print(f"发现已存在的合并文件 {output_file}，将进行追加。")
            # 读取旧文件时，也要保证 '股票代码' 是字符串
            existing_df = pd.read_csv(output_file, dtype=dtype_spec)
            
            # 检查并创建缺失的列
            for col in required_columns:
                if col not in existing_df.columns:
                    existing_df[col] = '' 
                    
            existing_df = existing_df[required_columns]
            
            # 【修复】修正历史数据的股票代码，确保它们是6位数，不足的用前导零填充
            print("【修复】正在为历史数据中的股票代码添加前导零，确保为6位...")
            if '股票代码' in existing_df.columns:
                existing_df['股票代码'] = existing_df['股票代码'].astype(str).str.zfill(6)
                
        except Exception as e:
            print(f"读取已存在的合并文件 {output_file} 时出错: {e}. 将忽略旧数据。")

    # --- 3. 合并新数据和旧数据 ---
    final_df = pd.concat([existing_df, new_combined_df], ignore_index=True)
    initial_rows = len(final_df)
    
    # 确保 '股票代码' 列是6位数，不足的用前导零填充 (最终确认)
    if '股票代码' in final_df.columns:
        final_df['股票代码'] = final_df['股票代码'].astype(str).str.zfill(6)
        
    # --- 3.5. NEW: 应用名称映射 ---
    if name_map:
        print("正在应用名称映射...")
        # 使用 map 方法更新 '股票名称' 列。
        # .fillna(final_df['股票名称']) 确保如果映射中找不到代码，则保留原始名称。
        final_df['股票名称'] = final_df['股票代码'].map(name_map).fillna(final_df['股票名称'])
        print("名称映射完成。")


    # --- 4. 清理、去重、排序 ---
    
    # 确保 '评分' 列是数值类型
    final_df['评分'] = pd.to_numeric(final_df['评分'], errors='coerce')
    
    # 核心步骤：先按 '评分' 降序排序。这将确保在下一步去重时，评分最高的记录被保留。
    final_df.sort_values(by='评分', ascending=False, na_position='last', inplace=True)
    
    # 去重: 假设 '股票代码' 是唯一的识别字段
    final_df.drop_duplicates(subset=['股票代码'], keep='first', inplace=True)
    
    final_rows = len(final_df)
    
    # --- 5. 写入文件 ---
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 写入最终的CSV文件
    # 严格遵循用户要求：不包含索引 (index=False)
    final_df.to_csv(output_file, index=False, encoding='utf-8')

    # --- 6. 结果反馈 ---
    print(f"\n操作完成！")
    print(f"总共处理记录数（合并前）：{initial_rows}")
    print(f"去重后保留的记录数：{final_rows}")
    print(f"成功保存到文件：{output_file}")
    
# 允许直接运行脚本
if __name__ == '__main__':
    combine_and_sort_csvs()
