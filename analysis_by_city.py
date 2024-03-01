#!/usr/bin/env python
# coding: utf-8

# In[23]:


import os
import pandas as pd

# Specify the folder path containing the input CSV files
folder_path = 'C:/Users/zhaoj/Desktop/umclst/result/SingleRegression/detrend/jjj/'

# Get a list of all CSV files in the folder
csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]

# Loop through each CSV file
for file in csv_files:
    # Read the CSV file into a DataFrame
    file_path = os.path.join(folder_path, file)
    df = pd.read_csv(file_path)
    
    # Calculate the median value of 'diff' for each unique 'change' type
    median_diff = df.groupby('change')['diff'].median().reset_index()
    
    # Construct the output file name
    output_file = os.path.splitext(file)[0] + '_out.csv'
    folder_pathout = 'C:/Users/zhaoj/Desktop/umclst/result/SingleRegression/Figure3/step1/jjj/'
    
    # Write the results to a new CSV file
    median_diff.to_csv(folder_pathout+output_file, index=False)

print("Processing complete.")


# In[1]:


import os
import pandas as pd

# Specify the folder path containing the input CSV files
folder_path = 'C:/Users/zhaoj/Desktop/umclst/result/SingleRegression/detrend/yrd/'

# Get a list of all CSV files in the folder
csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]

# Loop through each CSV file
for file in csv_files:
    # Read the CSV file into a DataFrame
    file_path = os.path.join(folder_path, file)
    df = pd.read_csv(file_path)
    
    # Calculate the median value of 'diff' for each unique 'change' type
    median_diff = df.groupby('change')['diff'].median().reset_index()
    
    # Construct the output file name
    output_file = os.path.splitext(file)[0] + '_out.csv'
    folder_pathout = 'C:/Users/zhaoj/Desktop/umclst/result/SingleRegression/Figure3/step1/yrd/'
    
    # Write the results to a new CSV file
    median_diff.to_csv(folder_pathout+output_file, index=False)

print("Processing complete.")


# In[25]:


import os
import pandas as pd

# Specify the folder path containing the input CSV files
folder_path = 'C:/Users/zhaoj/Desktop/umclst/result/SingleRegression/detrend/prd/'

# Get a list of all CSV files in the folder
csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]

# Loop through each CSV file
for file in csv_files:
    # Read the CSV file into a DataFrame
    file_path = os.path.join(folder_path, file)
    df = pd.read_csv(file_path)
    
    # Calculate the median value of 'diff' for each unique 'change' type
    median_diff = df.groupby('change')['diff'].median().reset_index()
    
    # Construct the output file name
    output_file = os.path.splitext(file)[0] + '_out.csv'
    folder_pathout = 'C:/Users/zhaoj/Desktop/umclst/result/SingleRegression/Figure3/step1/prd/'
    
    # Write the results to a new CSV file
    median_diff.to_csv(folder_pathout+output_file, index=False)

print("Processing complete.")


# In[27]:


import os
import pandas as pd

# 定义文件夹路径
change_df_folder = r"C:\Users\zhaoj\Desktop\umclst\result\SingleRegression\Figure3\step1\jjj"
lcz_df_folder = r"C:\Users\zhaoj\Desktop\umclst\reference2020\jjj"
output_folder = r"C:\Users\zhaoj\Desktop\umclst\result\SingleRegression\Figure3\step2\jjj"

# 获取change_df文件夹下的所有文件名
change_df_files = os.listdir(change_df_folder)

# 遍历change_df文件夹下的每个文件
for change_df_file in change_df_files:
    # 提取文件名中的部分信息
    file_name = change_df_file.split("_")[0]  # 提取"Beijingdata.tif"
    
    # 构建lcz_df文件的路径
    lcz_df_file = file_name + "_median_values.csv"
    lcz_df_path = os.path.join(lcz_df_folder, lcz_df_file)
    
    # 读取change_df文件
    change_df_path = os.path.join(change_df_folder, change_df_file)
    change_df = pd.read_csv(change_df_path)
    
    # 读取lcz_df文件
    lcz_df = pd.read_csv(lcz_df_path)
    
    # 进行计算，参考之前的代码
    # 创建一个空的列表，用于存储计算得到的差异值
    differences = []

    # 遍历change_df中的每一行
    for index, row in change_df.iterrows():
        # 获取当前行的'change'值
        change_value = row['change']

        # 拆解'change'值，得到lcz1和lcz2
        lcz1 = change_value // 100
        lcz2 = change_value % 100

        # 在lcz_df中检索'group'值等于lcz1和lcz2的行，并获取它们的'median_value'值
        median_value_lcz1 = lcz_df[lcz_df['group'] == lcz1]['median_value'].values
        median_value_lcz2 = lcz_df[lcz_df['group'] == lcz2]['median_value'].values

        # 如果能找到匹配的行，则计算差异值并添加到列表中
        if len(median_value_lcz1) > 0 and len(median_value_lcz2) > 0:
            difference = median_value_lcz2[0] - median_value_lcz1[0]
            differences.append(difference)
        else:
            differences.append(None)  # 如果没有匹配的行，则将差异值设置为None
    # 这里写上根据change_df和lcz_df进行的计算和处理逻辑
    # 将计算得到的差异值添加到change_df中的新列'difference'
    change_df['difference'] = differences
    change_df['vari'] = change_df['diff'] - change_df['difference']
    # 添加一列 'diff_difference'
    change_df['diff_difference'] = change_df['diff'] - change_df['difference']
    
    # 构建输出文件路径和文件名
    output_file = file_name + '_method_diff.csv'
    output_path = os.path.join(output_folder, output_file)
    
    # 将计算结果保存到CSV文件
    change_df.to_csv(output_path, index=False)


# In[2]:


import os
import pandas as pd

# 定义文件夹路径
change_df_folder = r"C:\Users\zhaoj\Desktop\umclst\result\SingleRegression\Figure3\step1\yrd"
lcz_df_folder = r"C:\Users\zhaoj\Desktop\umclst\reference2020\yrd"
output_folder = r"C:\Users\zhaoj\Desktop\umclst\result\SingleRegression\Figure3\step2\yrd"

# 获取change_df文件夹下的所有文件名
change_df_files = os.listdir(change_df_folder)

# 遍历change_df文件夹下的每个文件
for change_df_file in change_df_files:
    # 提取文件名中的部分信息
    file_name = change_df_file.split("_")[0]  # 提取"Beijingdata.tif"
    
    # 构建lcz_df文件的路径
    lcz_df_file = file_name + "_median_values.csv"
    lcz_df_path = os.path.join(lcz_df_folder, lcz_df_file)
    
    # 读取change_df文件
    change_df_path = os.path.join(change_df_folder, change_df_file)
    change_df = pd.read_csv(change_df_path)
    
    # 读取lcz_df文件
    lcz_df = pd.read_csv(lcz_df_path)
    
    # 进行计算，参考之前的代码
    # 创建一个空的列表，用于存储计算得到的差异值
    differences = []

    # 遍历change_df中的每一行
    for index, row in change_df.iterrows():
        # 获取当前行的'change'值
        change_value = row['change']

        # 拆解'change'值，得到lcz1和lcz2
        lcz1 = change_value // 100
        lcz2 = change_value % 100

        # 在lcz_df中检索'group'值等于lcz1和lcz2的行，并获取它们的'median_value'值
        median_value_lcz1 = lcz_df[lcz_df['group'] == lcz1]['median_value'].values
        median_value_lcz2 = lcz_df[lcz_df['group'] == lcz2]['median_value'].values

        # 如果能找到匹配的行，则计算差异值并添加到列表中
        if len(median_value_lcz1) > 0 and len(median_value_lcz2) > 0:
            difference = median_value_lcz2[0] - median_value_lcz1[0]
            differences.append(difference)
        else:
            differences.append(None)  # 如果没有匹配的行，则将差异值设置为None
    # 这里写上根据change_df和lcz_df进行的计算和处理逻辑
    # 将计算得到的差异值添加到change_df中的新列'difference'
    change_df['difference'] = differences
    change_df['vari'] = change_df['diff'] - change_df['difference']
    # 添加一列 'diff_difference'
    change_df['diff_difference'] = change_df['diff'] - change_df['difference']
    
    # 构建输出文件路径和文件名
    output_file = file_name + '_method_diff.csv'
    output_path = os.path.join(output_folder, output_file)
    
    # 将计算结果保存到CSV文件
    change_df.to_csv(output_path, index=False)


# In[29]:


import os
import pandas as pd

# 定义文件夹路径
change_df_folder = r"C:\Users\zhaoj\Desktop\umclst\result\SingleRegression\Figure3\step1\prd"
lcz_df_folder = r"C:\Users\zhaoj\Desktop\umclst\reference2020\prd"
output_folder = r"C:\Users\zhaoj\Desktop\umclst\result\SingleRegression\Figure3\step2\prd"

# 获取change_df文件夹下的所有文件名
change_df_files = os.listdir(change_df_folder)

# 遍历change_df文件夹下的每个文件
for change_df_file in change_df_files:
    # 提取文件名中的部分信息
    file_name = change_df_file.split("_")[0]  # 提取"Beijingdata.tif"
    
    # 构建lcz_df文件的路径
    lcz_df_file = file_name + "_median_values.csv"
    lcz_df_path = os.path.join(lcz_df_folder, lcz_df_file)
    
    # 读取change_df文件
    change_df_path = os.path.join(change_df_folder, change_df_file)
    change_df = pd.read_csv(change_df_path)
    
    # 读取lcz_df文件
    lcz_df = pd.read_csv(lcz_df_path)
    
    # 进行计算，参考之前的代码
    # 创建一个空的列表，用于存储计算得到的差异值
    differences = []

    # 遍历change_df中的每一行
    for index, row in change_df.iterrows():
        # 获取当前行的'change'值
        change_value = row['change']

        # 拆解'change'值，得到lcz1和lcz2
        lcz1 = change_value // 100
        lcz2 = change_value % 100

        # 在lcz_df中检索'group'值等于lcz1和lcz2的行，并获取它们的'median_value'值
        median_value_lcz1 = lcz_df[lcz_df['group'] == lcz1]['median_value'].values
        median_value_lcz2 = lcz_df[lcz_df['group'] == lcz2]['median_value'].values

        # 如果能找到匹配的行，则计算差异值并添加到列表中
        if len(median_value_lcz1) > 0 and len(median_value_lcz2) > 0:
            difference = median_value_lcz2[0] - median_value_lcz1[0]
            differences.append(difference)
        else:
            differences.append(None)  # 如果没有匹配的行，则将差异值设置为None
    # 这里写上根据change_df和lcz_df进行的计算和处理逻辑
    # 将计算得到的差异值添加到change_df中的新列'difference'
    change_df['difference'] = differences
    change_df['vari'] = change_df['diff'] - change_df['difference']
    # 添加一列 'diff_difference'
    change_df['diff_difference'] = change_df['diff'] - change_df['difference']
    
    # 构建输出文件路径和文件名
    output_file = file_name + '_method_diff.csv'
    output_path = os.path.join(output_folder, output_file)
    
    # 将计算结果保存到CSV文件
    change_df.to_csv(output_path, index=False)


# In[18]:


import os
import pandas as pd
import matplotlib.pyplot as plt

# 指定文件夹路径
folder_path = r"C:\Users\zhaoj\Desktop\umclst\reference2020\method_diff\jjj"

# 定义要排除的 change 值列表
exclude_changes = [101, 202, 303, 404, 505, 606, 707, 808, 909, 1010, 1111, 1212, 1313, 1414, 1515, 1616, 1717, 1818]

# 遍历文件夹中的每个 CSV 文件
for filename in os.listdir(folder_path):
    if filename.endswith(".csv"):
        # 读取 CSV 文件
        file_path = os.path.join(folder_path, filename)
        df = pd.read_csv(file_path)
        
        # 根据要排除的 change 值进行过滤
        filtered_df = df[~df['change'].isin(exclude_changes)]
        
        # 提取 'diff' 和 'difference' 列数据
        diff_values = filtered_df['diff']
        difference_values = filtered_df['difference']
        
        # 绘制散点图
        plt.scatter(diff_values, difference_values)
        
        # 添加 x=0 和 y=0 的虚线
        plt.axhline(0, color='gray', linestyle='--')
        plt.axvline(0, color='gray', linestyle='--')
        
        # 设置图表标题和坐标轴标签
        plt.title(filename)
        plt.xlabel('diff')
        plt.ylabel('difference')
        
        # 显示图表
        plt.show()


# In[19]:


import os
import pandas as pd
import matplotlib.pyplot as plt

# 指定文件夹路径
folder_path = r"C:\Users\zhaoj\Desktop\umclst\reference2020\method_diff\yrd"

# 定义要排除的 change 值列表
exclude_changes = [101, 202, 303, 404, 505, 606, 707, 808, 909, 1010, 1111, 1212, 1313, 1414, 1515, 1616, 1717, 1818]

# 遍历文件夹中的每个 CSV 文件
for filename in os.listdir(folder_path):
    if filename.endswith(".csv"):
        # 读取 CSV 文件
        file_path = os.path.join(folder_path, filename)
        df = pd.read_csv(file_path)
        
        # 根据要排除的 change 值进行过滤
        filtered_df = df[~df['change'].isin(exclude_changes)]
        ·
        # 提取 'diff' 和 'difference' 列数据
        diff_values = filtered_df['diff']
        difference_values = filtered_df['difference']
        
        # 绘制散点图
        plt.scatter(diff_values, difference_values)
        
        # 添加 x=0 和 y=0 的虚线
        plt.axhline(0, color='gray', linestyle='--')
        plt.axvline(0, color='gray', linestyle='--')
        
        # 设置图表标题和坐标轴标签
        plt.title(filename)
        plt.xlabel('diff')
        plt.ylabel('difference')
        
        # 显示图表
        plt.show()


# In[22]:


import os
import pandas as pd
import matplotlib.pyplot as plt

# 指定文件夹路径
folder_path = r"C:\Users\zhaoj\Desktop\umclst\reference2020\method_diff\jjj"

# 设置图片大小
fig, ax = plt.subplots(figsize=(6, 6))

# 定义要排除的 change 值列表
exclude_changes = [101, 202, 303, 404, 505, 606, 707, 808, 909, 1010, 1111, 1212, 1313, 1414, 1515, 1616, 1717, 1818]

# 遍历文件夹中的每个 CSV 文件
for filename in os.listdir(folder_path):
    if filename.endswith(".csv"):
        # 读取 CSV 文件
        file_path = os.path.join(folder_path, filename)
        df = pd.read_csv(file_path)
        
        # 根据要排除的 change 值进行过滤
        filtered_df = df[~df['change'].isin(exclude_changes)]
        
        # 提取 'diff' 和 'difference' 列数据
        diff_values = filtered_df['diff']
        difference_values = filtered_df['difference']
        change_values = filtered_df['change']
        
        # 绘制散点图
        plt.scatter(diff_values, difference_values)
        
        # 添加 x=0 和 y=0 的虚线
        plt.axhline(0, color='gray', linestyle='--')
        plt.axvline(0, color='gray', linestyle='--')
        
        # 在每个数据点旁边添加 'change' 标签
        for i in range(len(diff_values)):
            plt.annotate(change_values.iloc[i], (diff_values.iloc[i], difference_values.iloc[i]), textcoords="offset points", xytext=(5,5), ha='center')
        
        # 设置图表标题和坐标轴标签
        plt.title(filename)
        plt.xlabel('diff')
        plt.ylabel('difference')
        
        # 显示图表
        plt.show()


# In[23]:


# Beijing Example
import geopandas as gpd
import rasterio
from rasterio.mask import mask
import numpy as np

# 读取四个时期的城市边界数据
boundary_files = [r"C:\Users\zhaoj\Desktop\umclst\reference2020\example_beijing\Beijing1990.shp", r"C:\Users\zhaoj\Desktop\umclst\reference2000\example_beijing\Beijing2000.shp", r"C:\Users\zhaoj\Desktop\umclst\reference2020\example_beijing\Beijing2010.shp", r"C:\Users\zhaoj\Desktop\umclst\reference2020\example_beijing\Beijing2020.shp"]
boundaries = [gpd.read_file(file) for file in boundary_files]

# 读取北京市的LST栅格数据
lst_file = r"C:\Users\zhaoj\Desktop\umclst\dataimage\jjj\Beijingdata.tif"
with rasterio.open(lst_file) as src:
    lst_data = src.read(36)

# 定义一个函数，用于裁剪LST栅格数据并计算平均值
def calculate_mean_lst(boundary):
    # 裁剪LST栅格数据
    masked_data, _ = mask(dataset=src, shapes=boundary.geometry, crop=True)
    # 计算平均值
    mean_lst = np.mean(masked_data)
    return mean_lst

# 计算每个时期的平均LST值
mean_lst_values = []
for boundary in boundaries:
    mean_lst = calculate_mean_lst(boundary)
    mean_lst_values.append(mean_lst)

# 打印结果
for i, mean_lst in enumerate(mean_lst_values):
    print("Mean LST for boundary", i+1, ":", mean_lst)

