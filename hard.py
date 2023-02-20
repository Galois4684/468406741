import pandas as pd
import matplotlib.pyplot as plt

path = r"C:\Users\26271\Desktop\hard.xlsx"
data = pd.read_excel(path)
# day = list(range(7))
# for i in range(10):
#     plt.subplot(2,5,i+1)
#     plt.plot(day, data[7*i:7*i+7])
#     plt.plot(day, data[7*i:7*i+7], '.r')
# plt.show()
day = list(range(56))
plt.plot(day, data[0:56])
plt.show()
