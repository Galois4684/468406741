import numpy as np
import pandas as pd

# 取消科学计数
np.set_printoptions(suppress=True)

def Get_matrix(path):
    # Load the Excel file into a Pandas dataframe
    df = pd.read_excel(path)

    # Alternatively, you can convert the dataframe to a matrix using the `as_matrix()` method (but note that this method is deprecated and will be removed in a future version of Pandas)
    mat = df.values

    return mat

def Save_excel(array_1,path):
    data = pd.DataFrame(array_1)

    writer = pd.ExcelWriter(path)
    # header参数表示列的名称，index表示行的标签
    data.to_excel(writer, 'sheet_1', float_format='%.2f', header=False, index=False)
    writer.save()
    writer.close()

epoch = 20000

path_t = r'C:\Users\26271\Desktop\T_Matrix.xlsx'
path_f = r'C:\Users\26271\Desktop\F_Matrix.xlsx'
path_a = r'C:\Users\26271\Desktop\A_Matrix.xlsx'

t = Get_matrix(path_t).T
f = Get_matrix(path_f).T
print(t.shape,f.shape )
# print(t)
# print(f)
# print(f[0][0])
# print(type(f[0][0]))

p = np.random.normal(30,30, size=(359, 3))
print(p)
# print(np.linalg.det(f @ p))
print( (f @ p)@np.linalg.inv(f @ p))
# test = np.array([[0.319144], [0.629250], [0.232061]])
# print(a@test)

sum = np.zeros((6, 3))
a = t @ p @ np.linalg.inv(f @ p)
test = np.array([[1.0],[ 0.10034541883246695 ],[0.0]])
print(a@test)
# print(a@f-t)
# for i in range(epoch):
#     e = np.random.normal(1, 0.2, size=(6,359))
#     a = (t - e)@p@np.linalg.inv(f @ p)
#     # a = t @ p @ (f @ p) ** (-1)
#     # print(a)
#     sum = sum + a
# sum = sum / epoch
# print(sum)
# print()
# print(a)
# print()
# print(sum - a)
# print()
# Save_excel(sum,path_a)
# delta = np.absolute(((sum - a) / sum))*100
# print(delta)
# print(sum.shape)
# print(np.array([[0.319144], [0.629250], [0.232061]]).shape)
# p_e = sum@(np.array([[0.319144], [0.629250], [0.232061]]))
# print(p_e)
# print(p_e.shape)
# print(np.sum(p_e))
# # tries = np.array([1, 2, 3, 4, 5, 6])
# # print(hard)