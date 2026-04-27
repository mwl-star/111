import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.arima_process import ArmaProcess

# 设置随机种子以确保结果可重复
np.random.seed(0)

# 样本数量
N = 100  

# 一阶自回归参数
phi_1 = 0.5

# 生成AR(1)模型的时间序列数据
ar_params = np.array([1, -phi_1])  # 注意这里是1和-phi_1，表示y_t = phi_1*y_{t-1} + epsilon_t
ma_params = np.array([1])
arma_process = ArmaProcess(ar=ar_params, ma=ma_params)
y = arma_process.generate_sample(nsample=N)
print(y)
# 将时间序列数据转换为Pandas数据框
df = pd.DataFrame(y, columns=['value'])

# 绘制生成的时间序列数据
plt.figure(figsize=(10, 6))
plt.plot(df['value'], label='Generated AR(1) Time Series')
plt.title('Generated AR(1) Time Series')
plt.xlabel('Time')
plt.ylabel('Value')
plt.legend()
plt.show()

# 拟合一阶自回归模型
model = AutoReg(df['value'], lags=1)
model_fit = model.fit()

# 打印模型参数
print('AR(1) Model Parameters:')
print(model_fit.params)

# 提取并打印ϕ1的值
phi_1_estimated = model_fit.params[1]
print(f'Estimated ϕ1 (phi_1) = {phi_1_estimated}')

# 进行预测
predictions = model_fit.predict(start=len(df), end=len(df)+3)
print(predictions)

# # 绘制预测结果
# plt.figure(figsize=(10, 6))
# plt.plot(df['value'], label='Original Time Series')
# plt.plot(np.arange(len(df), len(df)+11), predictions, label='Predicted Values', linestyle='--')
# plt.title('AR(1) Model Prediction')
# plt.xlabel('Time')
# plt.ylabel('Value')
# plt.legend()
# plt.show()
