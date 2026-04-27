import csv
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import matplotlib
from random import sample, randint, random
from statsmodels.tsa.holtwinters import Holt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

################# Begin Import the train data1(training data japan1.csv) ##################
with open(r'E:\L\Tohoku\tohoku\tohoku_Consumption\training data japan1.csv', 'r', newline='') as csvfile:
    reader = csv.reader(csvfile)
    # Skip the title row（if have）
    next(reader)
    # Define the index of columns
    columns_of_interest = [2]  # the index begin from 0
    # Define a list
    str_array = []
    for row in reader:
        selected_data = [row[col] for col in columns_of_interest]  # choose the columns
        # Put the item into the list
        str_array.append(selected_data[0])
# Change the string item of list into float
train_japan1 = [float(s) for s in str_array]
# print(train_japan1)
#####################################END#########################################

################# Begin Import the train data2(actual data)(training data japan2.csv) ##############
with open(r'E:\L\Tohoku\tohoku\tohoku_Consumption\training data japan2.csv', 'r', newline='') as csvfile2:
    reader2 = csv.reader(csvfile2)
    # Skip the title row（if have）
    next(reader2)
    # Define the index of columns
    columns_of_interest2 = [2]  # the index begin from 0
    # Define a list
    str_array2 = []
    for row2 in reader2:
        selected_data2 = [row2[col] for col in columns_of_interest2]  # choose the columns
        # Put the item into the list
        str_array2.append(selected_data2[0])
# Change the string item of list into float
train_japan2 = [float(s) for s in str_array2]
# print(train_japan2)
#####################################END#########################################

################# Begin Import the train data3(actual data)(training data japan2.csv) ##############
with open(r'E:\L\Tohoku\tohoku\tohoku_Consumption\training data japan3.csv', 'r', newline='') as csvfile2:
    reader2 = csv.reader(csvfile2)
    # Skip the title row（if have）
    next(reader2)
    # Define the index of columns
    columns_of_interest2 = [2]  # the index begin from 0
    # Define a list
    str_array2 = []
    for row2 in reader2:
        selected_data2 = [row2[col] for col in columns_of_interest2]  # choose the columns
        # Put the item into the list
        str_array2.append(selected_data2[0])
# Change the string item of list into float
train_japan3 = [float(s) for s in str_array2]
# print(train_japan2)
#####################################END#########################################

x_max = 1 # The max dimension
x_min = 0 # The min dimension
N=30 # Number of population
D=2 # Dimension

# Generating the population
# x = np.random.rand(N, D) * (x_max - x_min) + x_min
# print(x)

############################# Begin LSTM model ##############################
n_preds=24
def LSTM_model(train_japan2,train_japan3, alpha):
    # 假设我们有一些时间序列数据
    # X_train 是输入数据，形状为 (样本数, 时间步长, 特征数)
    # y_train 是目标数据，形状为 (样本数, 输出维度)
    # 这里我们只是创建一些随机数据作为示例
    # X_train = np.random.random((100, 10, 1))  # 100个样本，每个样本10个时间步长，每个时间步长1个特征
    # y_train = np.random.random((100, 1))      # 100个样本，每个样本1个输出值
    X_train = train_japan3 
    y_train = train_japan2

    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_train= X_train.reshape(-1, 1, 1)
    y_train= y_train.reshape(-1, 1, 1)

    # 定义LSTM模型
    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=(1, 1)))  # 输入层，50个LSTM单元
    model.add(Dense(1))  # 输出层，1个输出单元

    # 编译模型
    # learning_rate=0.001
    learning_rate=alpha
    optimizer = tf.keras.optimizers.Adam(learning_rate)
    model.compile(optimizer=optimizer, loss='mse')  # 使用Adam优化器和均方误差损失函数

    # 训练模型
    model.fit(X_train, y_train, epochs=1, batch_size=32, verbose=1)

    # 使用模型进行预测
    # X_test = np.random.random((100, 10, 1))  # 假设我们有一个测试样本

    X_test = train_japan3 

    X_test = np.array(X_test)
    X_test = X_test.reshape(-1, 1, 1)


    y_pred = model.predict(X_test)
    # print(y_pred)

    array_forecast = y_pred
    # print(array_forecast)
        # Define the dataset as python lists 
    actual   = train_japan2
    forecast = array_forecast
    # Consider a list APE to store the 
    # APE value for each of the records in dataset 
    APE = [] 
     # Iterate over the list values 
    for day in range(24): 

        # Calculate percentage error 
        per_err = (actual[day] - forecast[day]) / actual[day] 

        # Take absolute value of 
        # the percentage error (APE) 
        per_err = abs(per_err) 

        # Append it to the APE list 
        APE.append(per_err) 

    # Calculate the MAPE 
    MAPE = sum(APE)/len(APE)
    # Print the MAPE value and percentage 
    # print(f''' 
    # MAPE   : { round(MAPE, 2) } 
    # MAPE % : { round(MAPE*100, 2) } % 
    # ''')
    return MAPE , forecast
############################# End LSTM model ##############################

w=1.5
x = np.random.rand(N, D) * (x_max - x_min) + x_min
################### PSO algorithm ###############################
def psoAlgorithm(w, c1 , c2, x, N, D, n_preds):


    # 设置字体和设置负号
    # matplotlib.rc("font", family="KaiTi")
    # matplotlib.rcParams["axes.unicode_minus"] = False
    # 初始化种群，群体规模，每个粒子的速度和规模
    # N = 100 # 种群数目
    # D = 3 # 维度
    T = 1 # 最大迭代次数
    c1 = c2 = 1.5 # 个体学习因子与群体学习因子
    w_max = 0.8 # 权重系数最大值
    w_min = 0.4 # 权重系数最小值
    x_max = 4 # 每个维度最大取值范围，如果每个维度不一样，那么可以写一个数组，下面代码依次需要改变
    x_min = -4 # 同上
    v_max = 1 # 每个维度粒子的最大速度
    v_min = -1 # 每个维度粒子的最小速度


    # 定义适应度函数
    def func(x):
        # print(x)
        alpha=x[0]
        m1=LSTM_model(train_japan2,train_japan3, alpha)
        return m1[0]


    # 初始化种群个体
    # x = np.random.rand(N, D) * (x_max - x_min) + x_min # 初始化每个粒子的位置
    v = np.random.rand(N, D) * (v_max - v_min) + v_min # 初始化每个粒子的速度

    # 初始化个体最优位置和最优值
    p = x # 用来存储每一个粒子的历史最优位置
    p_best = np.ones((N, 1))  # 每行存储的是最优值
    for i in range(N): # 初始化每个粒子的最优值，此时就是把位置带进去，把适应度值计算出来
        p_best[i] = func(x[i, :]) 

    # 初始化全局最优位置和全局最优值
    g_best = 100 #设置真的全局最优值
    gb = np.ones(T) # 用于记录每一次迭代的全局最优值
    x_best = np.ones(D) # 用于存储最优粒子的取值

    # 按照公式依次迭代直到满足精度或者迭代次数
    for i in range(T):
        for j in range(N):
            # 个更新个体最优值和全局最优值
            if p_best[j] > func(x[j,:]):
                p_best[j] = func(x[j,:])
                p[j,:] = x[j,:].copy()
            # p_best[j] = func(x[j,:]) if func(x[j,:]) < p_best[j] else p_best[j]
            # 更新全局最优值
            if g_best > p_best[j]:
                g_best = p_best[j]
                x_best = x[j,:].copy()   # 一定要加copy，否则后面x[j,:]更新也会将x_best更新
            # 计算动态惯性权重
            # w = w_max - (w_max - w_min) * i / T
            # 更新位置和速度
            v[j, :] = w * v[j, :] + c1 * np.random.rand(1) * (p[j, :] - x[j, :]) + c2 * np.random.rand(1) * (x_best - x[j, :])
            x[j, :] = x[j, :] + v[j, :]
            # 边界条件处理
            for ii in range(D):
                if (v[j, ii] > v_max) or (v[j, ii] < v_min):
                    v[j, ii] = v_min + np.random.rand(1) * (v_max - v_min)
                if (x[j, ii] > x_max) or (x[j, ii] < x_min):
                    x[j, ii] = x_min + np.random.rand(1) * (x_max - x_min)
        # 记录历代全局最优值
        gb[i] = g_best
    # print("最优值为", gb[T - 1],"最优位置为",x_best)
    # plt.plot(range(T),gb)
    # plt.xlabel("迭代次数")
    # plt.ylabel("适应度值")
    # plt.title("适应度进化曲线")
    plt.show()
    return gb[T - 1],x_best

w=1
c1=1.5
c2=1.5
trainvalue=psoAlgorithm(w, c1 , c2, x, N, D, n_preds)
#训练集数据储存
trainresults=trainvalue[0]
filename = "TrainResults_PSO_LSTM.csv"
# 打开文件，并使用csv.writer写入数据
with open(filename, mode='w', newline='') as file:
    writer = csv.writer(file)
    # 检查trainresults是否为可迭代对象（如数组或列表）
    if isinstance(trainresults, (list, np.ndarray)):
        writer.writerow([float(item) for item in trainresults])
    else:
        # 如果trainresults是单个值
        writer.writerow([float(trainresults)])

############################################# Test Data Set#########################################################

################# Begin Import the test data1(test data japan1.csv) ##################
with open(r'E:\L\Tohoku\tohoku\tohoku_Consumption\test data japan1.csv', 'r', newline='') as csvfile:
    reader = csv.reader(csvfile)
    # Skip the title row（if have）
    next(reader)
    # Define the index of columns
    columns_of_interest = [2]  # the index begin from 0
    # Define a list
    str_array = []
    for row in reader:
        selected_data = [row[col] for col in columns_of_interest]  # choose the columns
        # Put the item into the list
        str_array.append(selected_data[0])
# Change the string item of list into float
test_japan1 = [float(s) for s in str_array]
# print(train_japan1)
#####################################END#########################################

################# Begin Import the test data2(actual data)(test data japan2.csv) ##############
with open(r'E:\L\Tohoku\tohoku\tohoku_Consumption\test data japan2.csv', 'r', newline='') as csvfile2:
    reader2 = csv.reader(csvfile2)
    # Skip the title row（if have）
    next(reader2)
    # Define the index of columns
    columns_of_interest2 = [2]  # the index begin from 0
    # Define a list
    str_array2 = []
    for row2 in reader2:
        selected_data2 = [row2[col] for col in columns_of_interest2]  # choose the columns
        # Put the item into the list
        str_array2.append(selected_data2[0])
# Change the string item of list into float
test_japan2 = [float(s) for s in str_array2]
# print(train_japan2)
#####################################END#########################################

################# Begin Import the test data3(actual data)(test data japan3.csv) ##############
with open(r'E:\L\Tohoku\tohoku\tohoku_Consumption\test data japan3.csv', 'r', newline='') as csvfile2:
    reader2 = csv.reader(csvfile2)
    # Skip the title row（if have）
    next(reader2)
    # Define the index of columns
    columns_of_interest2 = [2]  # the index begin from 0
    # Define a list
    str_array2 = []
    for row2 in reader2:
        selected_data2 = [row2[col] for col in columns_of_interest2]  # choose the columns
        # Put the item into the list
        str_array2.append(selected_data2[0])
# Change the string item of list into float
test_japan3 = [float(s) for s in str_array2]
# print(train_japan2)
#####################################END#########################################


x_max = 1 # The max dimension
x_min = 0 # The min dimension
N=30 # Number of population
D=1 # Dimension

# Generating the population
# x = np.random.rand(N, D) * (x_max - x_min) + x_min
# print(x)
  ############################# Begin LSTM model ##############################
n_preds=24
def test_LSTM_model(test_japan2,test_japan3, alpha):
    # 假设我们有一些时间序列数据
    # X_train 是输入数据，形状为 (样本数, 时间步长, 特征数)
    # y_train 是目标数据，形状为 (样本数, 输出维度)
    # 这里我们只是创建一些随机数据作为示例
    # X_train = np.random.random((100, 10, 1))  # 100个样本，每个样本10个时间步长，每个时间步长1个特征
    # y_train = np.random.random((100, 1))      # 100个样本，每个样本1个输出值
    X_train = test_japan3 
    y_train = test_japan2

    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_train= X_train.reshape(-1, 1, 1)
    y_train= y_train.reshape(-1, 1, 1)

    # 定义LSTM模型
    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=(1, 1)))  # 输入层，50个LSTM单元
    model.add(Dense(1))  # 输出层，1个输出单元

    # 编译模型
    # learning_rate=0.001
    learning_rate=alpha
    optimizer = tf.keras.optimizers.Adam(learning_rate)
    model.compile(optimizer=optimizer, loss='mse')  # 使用Adam优化器和均方误差损失函数

    # 训练模型
    model.fit(X_train, y_train, epochs=1, batch_size=32, verbose=1)

    # 使用模型进行预测
    # X_test = np.random.random((100, 10, 1))  # 假设我们有一个测试样本

    X_test = test_japan3 

    X_test = np.array(X_test)
    X_test = X_test.reshape(-1, 1, 1)


    y_pred = model.predict(X_test)
    # print(y_pred)

    array_forecast = y_pred
    # print(array_forecast)
        # Define the dataset as python lists 
    actual   = test_japan2
    forecast = array_forecast
    # Consider a list APE to store the 
    # APE value for each of the records in dataset 
    APE = [] 
     # Iterate over the list values 
    for day in range(24): 

        # Calculate percentage error 
        per_err = (actual[day] - forecast[day]) / actual[day] 

        # Take absolute value of 
        # the percentage error (APE) 
        per_err = abs(per_err) 

        # Append it to the APE list 
        APE.append(per_err) 

    # Calculate the MAPE 
    MAPE = sum(APE)/len(APE)
    # Print the MAPE value and percentage 
    # print(f''' 
    # MAPE   : { round(MAPE, 2) } 
    # MAPE % : { round(MAPE*100, 2) } % 
    # ''')
    return MAPE , forecast
############################# End LSTM model ##############################

Plotvalue = trainvalue[1]
alpha = Plotvalue[0]
data1 = test_LSTM_model(test_japan2,test_japan3, alpha)

########################################  测试集上的数据储存 ######################################## 
testresults = data1[0]
filename = "TestResults_PSO_LSTM.csv"
# 打开文件，并使用csv.writer写入数据
with open(filename, mode='w', newline='') as file:
    writer = csv.writer(file)
    # # for row in data2:
    # #     writer.writerow(float(item) for item in row)
    # writer.writerow([float(item) for item in testresults])

    # 检查trainresults是否为可迭代对象（如数组或列表）
    if isinstance(testresults, (list, np.ndarray)):
        writer.writerow([float(item) for item in testresults])
    else:
        # 如果trainresults是单个值
        writer.writerow([float(testresults)])
########################################  测试集上的预测值数据 ######################################## 

data2 = data1[1]
filename = "PSO_LSTM.csv"
# 打开文件，并使用csv.writer写入数据
with open(filename, mode='w', newline='') as file:
    writer = csv.writer(file)
    # for row in data2:
    #     writer.writerow(float(item) for item in row)
    writer.writerow([float(item) for item in data2])

print(data2)
print(testresults)