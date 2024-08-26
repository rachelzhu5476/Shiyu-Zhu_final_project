import os
import pandas as pd
import re

# 定义CSV文件的目录路径
csv_dir = r'/Users/rachelzhu/Desktop/MIMo/mimoEnv/models/standup/model_ex1'  # 请根据实际路径调整

# 获取目录下所有CSV文件的文件名
csv_files = [file for file in os.listdir(csv_dir) if file.endswith('.csv')]

# 使用正则表达式提取文件名中的数字部分进行排序
def extract_number(filename):
    match = re.search(r'\d+', filename)
    if match:
        return int(match.group())
    else:
        raise ValueError(f"文件名 {filename} 中未找到数字部分。")

# 按文件名中的数字部分排序
csv_files_sorted = sorted(csv_files, key=extract_number)

# 初始化一个空的列表，用于存放每个CSV文件的数据
csv_data = []

# 依次读取每个CSV文件并将其内容添加到列表中
for file in csv_files_sorted:
    file_path = os.path.join(csv_dir, file)
    data = pd.read_csv(file_path)
    csv_data.append(data)

# 将所有数据合并成一个DataFrame
merged_data = pd.concat(csv_data, ignore_index=True)

# 定义保存合并后文件的路径
output_file_path = os.path.join(csv_dir, 'merged_training_data.csv')

# 将合并后的数据保存为一个新的CSV文件
merged_data.to_csv(output_file_path, index=False, encoding='utf-8-sig')

print(f"CSV文件已成功合并并保存在：{output_file_path}")