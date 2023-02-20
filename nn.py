import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

def Get_matrix(path):
    # Load the Excel file into a Pandas dataframe
    df = pd.read_excel(path)

    # Alternatively, you can convert the dataframe to a matrix using the `as_matrix()` method (but note that this method is deprecated and will be removed in a future version of Pandas)
    mat = df.values

    return mat

path_t = r'C:\Users\26271\Desktop\T_Matrix.xlsx'
path_f = r'C:\Users\26271\Desktop\F_Matrix.xlsx'

t = Get_matrix(path_t)
f = Get_matrix(path_f)

# 构建数据集和标签
data = f
label = t

# 划分训练集和测试集
kf = KFold(n_splits=5, shuffle=True, random_state=42)
train_index, test_index = next(kf.split(data))

train_data, train_label = data[train_index], label[train_index]
test_data, test_label = data[test_index], label[test_index]

# 构建神经网络
from keras.models import Sequential
from keras.layers import Dense

model = Sequential()
model.add(Dense(64, input_dim=3, activation='relu'))
model.add(Dense(6, activation='linear'))

model.compile(loss='mean_squared_error', optimizer='adam')

# test_data = np.array(test_data, dtype=np.float32)
# 训练神经网络
# print(test_data)
model.fit(train_data, train_label, epochs=50, batch_size=1, validation_data=(test_data, test_label))
prediction = model.predict(test_data)
print(np.sum(np.absolute(test_label - prediction), axis = 0)/np.sum(test_label)*100)



# # 使用神经网络进行回归预测
# test = Get_matrix(r'C:\Users\26271\Desktop\yanshi.xlsx')
# test = np.array(test, dtype=np.float32)
# prediction = model.predict(test)
# print(prediction)