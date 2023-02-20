import numpy as np
from scipy.integrate import odeint
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import pandas as pd

# 定义SIR模型
def sir_model(y, t, N, beta, gamma):
    S, I, R = y
    dSdt = -beta * S * I / N
    dIdt = beta * S * I / N - gamma * (I - 13800)
    dRdt = gamma * (I - 13800)
    return dSdt, dIdt, dRdt

# 定义损失函数
def loss(params, y0, dI, N, t):
    beta, gamma = params
    res = odeint(sir_model, y0, t, args=(N, beta, gamma))
    I = res[:, 1]
    dI_pred = np.diff(I)
    return np.sum((dI[:-1] - dI_pred)**2)

# 读取数据
path = r'C:\Users\26271\Desktop\donglixue.xlsx'
df = pd.read_excel(path)
my_list = df.values.tolist()
beta_sum = 0
gamma_sum = 0
for i in range(20):
    # 群体总人数
    N = 4000000

    # 初始感染者和康复者人数
    I0, R0 = 80000, 1500*i

    # 初始易感者人数
    S0 = N - I0 - R0

    # 初始状态
    y0 = S0, I0, R0

    # 计算每天新增感染者人数
    nums = my_list
    if i == 0:
        for j in range(len(nums)):
            nums[j] = nums[j][0]
    dI = np.diff(nums)

    # 优化参数
    t = np.arange(len(dI))
    res = minimize(loss, x0=[0.1, 0.1], args=(y0, dI, N, t))
    beta, gamma = res.x
    beta_sum = beta_sum + beta
    gamma_sum = gamma_sum + gamma
    # 求解微分方程
    t = np.linspace(0, len(nums)+60-1, len(nums)+60)
    res = odeint(sir_model, y0, t, args=(N, beta, gamma))
    S, I, R = res.T
    print(I[-1:])
print(beta_sum/20, gamma_sum/20)
# # 绘制结果
# fig, ax = plt.subplots(figsize=(8, 6))
# ax.plot(t, S, 'b', label='Potential Gamers')
# ax.plot(t, I, 'r', label='Active Gamers')
# ax.plot(t, R, 'g', label='Non-game Gamers')
# ax.plot(range(1,len(dI)+1), dI, 'k', label='Daily new infections (data)')
# ax.legend(loc='best')
# ax.set_xlabel('Time (days)')
# ax.set_ylabel('Number of individuals')
# plt.show()
