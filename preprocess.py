import os
import pandas as pd
import talib
from tqdm import tqdm

def preprocess_data(data_dir, output_dir):
    files = [file for file in os.listdir(data_dir) if file.endswith('.csv')]
    
    # 使用 tqdm 显示进度条
    for file in tqdm(files, desc="Processing files"):
        stock_name = file.split('.')[0]+file.split('.')[1]
        df = pd.read_csv(os.path.join(data_dir, file), index_col=0, parse_dates=True)

        # 计算技术指标
        df['MA5'] = talib.MA(df['close'], timeperiod=5)
        df['MA10'] = talib.MA(df['close'], timeperiod=10)
        df['RSI'] = talib.RSI(df['close'], timeperiod=14)
        df['Williams %R'] = talib.WILLR(df['high'], df['low'], df['close'], timeperiod=14)

        # 保存处理后的数据
        df.to_csv(os.path.join(output_dir, f"{stock_name}.csv"))

# 使用示例
# preprocess_data('data_more', 'data_more_processed')
preprocess_data('E:/ML/datasets/StockChina/daily/Version 2/daily', 'E:/ML/datasets/StockChina/daily/Version 2/data_processed')
