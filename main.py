import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.layers import LSTM
from keras.layers import Dense
from keras.models import Sequential
from sklearn.metrics import mean_squared_error
import numpy as np
import tensorflow as tf
import math
#忽略
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# 数据的差分转换
def difference(data_set, interval=1):
    diff = list()
    for i in range(interval, len(data_set)):
        value = data_set[i] - data_set[i - interval]
        diff.append(value)
    return pd.Series(diff)


# 对预测的数据进行逆差分转换
def invert_difference(history, yhat, interval=1):
    return yhat + history[-interval]


# 将数据转换为监督学习集，移位后产生的NaN值补0
def timeseries_to_supervised(data, lag=1):
    df = pd.DataFrame(data)
    columns = [df.shift(i) for i in range(1, lag + 1)]
    columns.append(df)
    df = pd.concat(columns, axis=1)
    df.fillna(0, inplace=True)
    return df


# 将数据缩放到[-1,1]之间
def scale(train, test):
    # 创建一个缩放器，将数据集中的数据缩放到[-1,1]的取值范围中
    scaler = MinMaxScaler(feature_range=(-1, 1))
    # 使用数据来训练缩放器
    scaler = scaler.fit(train)
    # 使用缩放器来将训练集和测试集进行缩放
    train_scaled = scaler.transform(train)
    test_scaled = scaler.transform(test)
    return scaler, train_scaled, test_scaled


# 将预测值进行逆缩放，使用之前训练好的缩放器，x为一维数组，y为实数
def invert_scale(scaler, X, y):
    # 将X,y转换为一个list列表
    new_row = [x for x in X] + [y]
    # 将列表转换为数组
    array = np.array(new_row, dtype=object)
    # 将数组重构成一个形状为[1,2]的二维数组->[[10,12]]
    array = array.reshape(1, len(array))
    # 逆缩放输入的形状为[1,2]，输出形状也是如此
    invert = scaler.inverse_transform(array)
    # 只需要返回y值即可
    return invert[0, -1]


# 构建一个LSTM模型
def fit_lstm(train, batch_size, nb_epoch, neurons):
    # 将数据对中的x和y分开
    X, y = train[:, 0:-1], train[:, -1]
    # 将2D数据拼接成3D数据，形状为[N*1*1]
    X = X.reshape(X.shape[0], 1, X.shape[1])

    model = Sequential()
    model.add(LSTM(neurons, batch_input_shape=(batch_size, X.shape[1], X.shape[2]), stateful=True))
    model.add(Dense(1))

    model.compile(loss='mean_squared_error', optimizer='adam')
    for i in range(nb_epoch):
        # shuffle是不混淆数据顺序
        his = model.fit(X, y, batch_size=batch_size, verbose=1, shuffle=False)
        # 每训练完一次就重置一次网络状态，网络状态与网络权重不同
        model.reset_states()
    return model


# 开始单步预测
def forecast_lstm(model, batch_size, X):
    # 将形状为[1:]的，包含一个元素的一维数组X，转换形状为[1,1,1]的3D张量
    X = X.reshape(1, 1, len(X))
    # 输出形状为1行一列的二维数组yhat
    yhat = model.predict(X.astype(np.float), batch_size=batch_size)
    # 将yhat中的结果返回
    return yhat[0, 0]

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

# 将原数据转换为二维数组形式，例如：
# [[4.6838],[4.6882],[4.7048]]
raw_value = series.values

# 数据增强： 插值

# Calculate the number of elements to insert between each pair of elements
num_insertions = 10

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
raw_value = new_arr

# for i in range(len(raw_value) - 1):
#     ave_delta = (raw_value[i+1] - raw_value[i]) / 20
#     for j in range(1, 21):
#         list(raw_value).insert(i+j, list(raw_value[i] + j*ave_delta))

# 将数据进行差分转换，例如[[4.6838],[4.6882],[4.7048]]转换为[[4.6882-4.6838],[4.7048-4.6882]]
diff_value = difference(raw_value, 1)
#
# 将序列形式的数据转换为监督学习集形式，例如[[10],[11],[12],[13]]
# 在此将其转换为监督学习集形式：[[0,10],[10,11],[11,12],[12,13]]，
# 即前一个数作为输入，后一个数作为对应的输出
supervised = timeseries_to_supervised(diff_value, 1)
supervised_value = supervised.values

# 将数据集分割为训练集和测试集
testNum = 1000
train, test = supervised_value[:-testNum], supervised_value[-testNum:]
# print(test)
# 将训练集和测试集都缩放到[-1,1]之间
scaler, train_scaled, test_scaled = scale(train, test)
print(train_scaled)
print(train_scaled[0])
print(type(train_scaled[0]))
print(train_scaled.shape)
print(len(train_scaled))
print(test_scaled[len(train_scaled), 0:-1])

# 构建一个LSTM模型并训练，样本数为1，训练次数为5，LSTM层神经元个数为4
lstm_model = fit_lstm(train_scaled, 1, 1, 4)
# 遍历测试集，对数据进行单步预测
predictions = list()
len_test_scaled = len(test_scaled)
for i in range(len_test_scaled+1000):
    # 将测试集拆分为X和y
    X, y = test[i, 0:-1], test[i, -1]
    print(X, y)
    # 将训练好的模型、测试数据传入预测函数中
    yhat = forecast_lstm(lstm_model, 1, X)
    if i == len_test_scaled-1:
        np.append(test, np.array([np.array(yhat), np.array(yhat)]), axis = 1)
    # 将预测值进行逆缩放
    yhat = invert_scale(scaler, X, yhat)
    # 对预测的y值进行逆差分
    yhat = invert_difference(raw_value, yhat, len(test_scaled) + 1 - i)
    # 存储正在预测的y值
    predictions.append(yhat)

# 计算方差
mse = mean_squared_error(raw_value[-testNum:], predictions)
print("Test MSE:", mse)
plt.plot(raw_value[-testNum:])
plt.plot(predictions)
plt.legend(['true', 'pred'])
plt.show()