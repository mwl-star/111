#=========导入相关库===============
import numpy as np
from numpy.random import random as rand
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
with open(r'E:\L\Kyushu\Kyushu\Kyushu_Generation\training data japan1.csv', 'r', newline='') as csvfile:
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
with open(r'E:\L\Kyushu\Kyushu\Kyushu_Generation\training data japan2.csv', 'r', newline='') as csvfile2:
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

################# Begin Import the train data3(actual data)(training data japan3.csv) ##############
with open(r'E:\L\Kyushu\Kyushu\Kyushu_Generation\training data japan3.csv', 'r', newline='') as csvfile2:
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


#========参数设置==============
# objfun:目标函数 
# N_pop: 种群规模，通常为10到40
# N_gen: 迭代数
# A: 响度（恒定或降低） 
# r: 脉冲率（恒定或减小） 
# 此频率范围决定范围
# 如有必要，应更改这些值 
# Qmin: 频率最小值
# Qmax: 频率最大值
# d: 维度
# lower: 下界
# upper: 上界
A=0.5
r=0.5 
x = np.random.rand(N, D) * (x_max - x_min) + x_min
def baAlgorithm(A, r , x, N, D, n_preds):
    def bat_algorithm(func, A,r,N_pop=len(x), N_gen=1,
        Qmin=0, Qmax=2, d=2, lower=-2, upper=2):
    
        N_iter = 0 # Total number of function evaluations
    
        #=====速度上下限================
        Lower_bound = lower * np.ones((1,d))
        Upper_bound = upper * np.ones((1,d))
    
        Q = np.zeros((N_pop, 1)) # 频率
        v = np.zeros((N_pop, d)) # 速度
        S = np.zeros((N_pop, d))
    
        #=====初始化种群、初始解=======
        # Sol = np.random.uniform(Lower_bound, Upper_bound, (N_pop, d))
        # Fitness = objfun(Sol)
        # Sol = np.zeros((N_pop, d))
        Sol = x
        Fitness = np.zeros((N_pop, 1))
        for i in range(N_pop):
            # Sol[i] = np.random.uniform(Lower_bound, Upper_bound, (1, d))
            Fitness[i] = func(Sol[i])
    
        #====找出初始最优解===========
        fmin = min(Fitness)
        Index = list(Fitness).index(fmin)
        best = Sol[Index]
    
        #======开始迭代=======
        for t in range(N_gen):
    
            #====对所有蝙蝠/解决方案进行循环 ======
            for i in range(N_pop):
                # Q[i] = Qmin + (Qmin - Qmax) * np.random.rand
                Q[i] = np.random.uniform(Qmin, Qmax)
                v[i] = v[i] + (Sol[i] - best) * Q[i]
                S[i] = Sol[i] + v[i]
    
                #===应用简单的界限/限制====
                Sol[i] = simplebounds(Sol[i], Lower_bound, Upper_bound)
                # Pulse rate
                if rand() > r:
                    # The factor 0.001 limits the step sizes of random walks
                    S[i] = best + 0.001*np.random.randn(1, d)
    
                #====评估新的解决方案 ===========
                # print(i)
                Fnew = func(S[i])
                #====如果解决方案有所改进，或者声音不太大，请更新====
                if (Fnew <= Fitness[i]) and (rand() < A):
                    Sol[i] = S[i]
                    Fitness[i] = Fnew
    
                #====更新当前的最佳解决方案======
                if Fnew <= fmin:
                    best = S[i]
                    fmin = Fnew
    
            N_iter = N_iter + N_pop
    
        print('Number of evaluations: ', N_iter)
        print("Best = ", best, '\n fmin = ', fmin)
    
        return fmin, best
    
    
    def simplebounds(s, Lower_bound, Upper_bound):
    
        Index = s > Lower_bound
        s = Index * s + ~Index * Lower_bound
        Index = s < Upper_bound
        s = Index * s + ~Index * Upper_bound
    
        return s
    
    
    #====目标函数=============
    # def test_function(u):
    #     a = u ** 2
    #     return a.sum(axis=0)
    
    def func(x):
        # print(x)
        alpha=x[0]
        m1=LSTM_model(train_japan2,train_japan3, alpha)
        return m1[0]

    return bat_algorithm(func, A, r,N_pop=len(x), N_gen=1,  
        Qmin=0, Qmax=2, d=2, lower=-2, upper=2)
 

trainvalue=baAlgorithm(A, r , x, N, D, n_preds)
#训练集数据储存
trainresults=trainvalue[0]
filename = "TrainResults_BA_LSTM.csv"
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
with open(r'E:\L\Kyushu\Kyushu\Kyushu_Generation\test data japan1.csv', 'r', newline='') as csvfile:
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
with open(r'E:\L\Kyushu\Kyushu\Kyushu_Generation\test data japan2.csv', 'r', newline='') as csvfile2:
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
with open(r'E:\L\Kyushu\Kyushu\Kyushu_Generation\test data japan3.csv', 'r', newline='') as csvfile2:
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
filename = "TestResults_BA_LSTM.csv"
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
filename = "BA_LSTM.csv"
# 打开文件，并使用csv.writer写入数据
with open(filename, mode='w', newline='') as file:
    writer = csv.writer(file)
    # for row in data2:
    #     writer.writerow(float(item) for item in row)
    writer.writerow([float(item) for item in data2])

print(data2)
print(testresults)

