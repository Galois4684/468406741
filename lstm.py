import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import LSTM, Dense


def Get_matrix(path):
    # Load the Excel file into a Pandas dataframe
    df = pd.read_excel(path)

    # Alternatively, you can convert the dataframe to a matrix using the `as_matrix()` method (but note that this method is deprecated and will be removed in a future version of Pandas)
    mat = df.values

    return mat

# 读取数据
path = r'C:\Users\26271\Desktop\lstm.xlsx'
data = Get_matrix(path).reshape(-1, 1)

# 读取数据
df = pd.read_csv('data.csv')

# 准备数据集
data = []
labels = []
look_back = 3
for i in range(len(df)-look_back-6):
    data.append(df.iloc[i:(i+look_back), :].values)
    labels.append(df.iloc[i+look_back+6, :].values)
data = np.array(data)
labels = np.array(labels)

# 划分数据集
k = 5
num_val_samples = len(data) // k
num_epochs = 10
all_scores = []

for i in range(k):
    print('processing fold #', i)
    val_data = data[i * num_val_samples: (i + 1) * num_val_samples]
    val_labels = labels[i * num_val_samples: (i + 1) * num_val_samples]

    partial_train_data = np.concatenate(
        [data[:i * num_val_samples], data[(i + 1) * num_val_samples:]],
        axis=0)
    partial_train_labels = np.concatenate(
        [labels[:i * num_val_samples], labels[(i + 1) * num_val_samples:]],
        axis=0)

    # 定义模型
    model = Sequential()
    model.add(LSTM(64, input_shape=(look_back, 6)))
    model.add(Dense(6))
    model.compile(optimizer='adam', loss='mse')

    # 训练模型
    model.fit(partial_train_data, partial_train_labels, epochs=num_epochs, batch_size=10)

    # 测试模型
    val_mse = model.evaluate(val_data, val_labels)
    all_scores.append(val_mse)

print('all_scores:', all_scores)
print('mean score:', np.mean(all_scores))

# 预测未来60天数据
X_future = np.array(df.tail(look_back).values.reshape(1, look_back, 6))
prediction = np.empty((60, 6))
for i in range(60):
    y_future = model.predict(X_future)
    prediction[i] = y_future
    X_future = np.concatenate([X_future[:, 1:, :], y_future.reshape(1, 1, 6)], axis=1)

# 输出最后一个预测数据
print(prediction[-1])

# 绘制预测数据
import matplotlib.pyplot as plt
plt.figure(figsize=(12, 6))
plt.plot(df.index, df['A'], label='actual')
plt.plot(df.index[-1]+np.arange(60)+1, prediction[:, 0], label='predicted')
plt.legend()
plt.show()
