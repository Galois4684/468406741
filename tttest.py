import numpy as np
import pandas as pd


# 读取数据，将日期和时间列合并，其他列删除，合并后的列转换为时间格式，设为索引
# data = pd.read_csv('data.csv')
# data['Date'] = data['Date'] + ' '
# data['Date'] = data['Date'] + data['Time']
# data.drop(['Num', 'LEVEL', 'TEMPERATURE', 'CONDUCTIVITY', 'Time'], axis=1, inplace=True)
path = r"C:\Users\26271\Desktop\Problem_C_Data_Wordle1.xlsx"
data = pd.read_excel(path)
data.drop(['Contest number','Word','Number in hard mode','1 try','2 tries', '3 tries', '4 tries', '5 tries', '6 tries', '7 or more tries (X)'],axis=1,inplace=True)
data['Date']=pd.to_datetime(data['Date'])
series = data.set_index(['Date'], drop=True)
raw_value = series.values
# 数据增强： 插值

# Calculate the number of elements to insert between each pair of elements
num_insertions = 80

# Calculate the number of total elements after insertion
new_len = (raw_value.shape[0] - 1) * (num_insertions + 1) + 1

# Create a new array of the desired size
new_arr = np.empty((new_len, raw_value.shape[1]))

# Fill in the new array with the original values and the interpolated values

for i in range(raw_value.shape[0] - 1):
    # Calculate the interpolation points
    x = np.linspace(0, 1, num_insertions + 2)[1:-1]

    # Calculate the values to be inserted
    y = raw_value[i] + x * (raw_value[i + 1] - raw_value[i])

    # Calculate the indices of the new array to insert the values
    start_idx = i * (num_insertions + 1)
    end_idx = (i + 1) * (num_insertions + 1) + 1

    # Insert the values into the new array
    new_arr[start_idx:end_idx, :] = np.vstack([raw_value[i], y.reshape(-1, 1), raw_value[i + 1]])
num = list(new_arr)

for i in range(len(num)):
    num[i] = float(num[i])



import matplotlib.pyplot as plt
plt.plot(num)
plt.show()