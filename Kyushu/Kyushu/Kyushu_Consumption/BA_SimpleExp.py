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
from statsmodels.tsa.holtwinters import SimpleExpSmoothing

################# Begin Import the train data1(training data japan1.csv) ##################
with open(r'E:\L\Kyushu\Kyushu\Kyushu_Consumption\training data japan1.csv', 'r', newline='') as csvfile:
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
with open(r'E:\L\Kyushu\Kyushu\Kyushu_Consumption\training data japan2.csv', 'r', newline='') as csvfile2:
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



x_max = 1 # The max dimension
x_min = 0 # The min dimension
N=30 # Number of population
D=1 # Dimension

# Generating the population
# x = np.random.rand(N, D) * (x_max - x_min) + x_min
# print(x)

############################# Begin SimpleExpSmoothing model ##############################
series = train_japan1 #Define the seasonal data list
n_preds=24
# Define the SimpleExpSmoothing model
def SimpleExp_model(series, alpha, n_preds):
    # 读取时间序列数据
    data = pd.read_csv(r'E:\L\Kyushu\Kyushu\Kyushu_Consumption\training data japan3.csv', parse_dates=['date'], index_col='date')

    # 设置平滑系数
    # alpha = 0.2

    # 初始化预测值为第一个观察值
    data['smoothed'] = data['value'].iloc[0]

    # 计算指数平滑
    for i in range(1, len(data)):
        data['smoothed'].iloc[i] = alpha * data['value'].iloc[i] + (1 - alpha) * data['smoothed'].iloc[i - 1]

    # 打印结果
    # print(data)
    array_forecast = data['smoothed'].tolist()
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
############################# End SimpleExpSmoothing model ##############################

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
    def bat_algorithm(func, A,r,N_pop=len(x), N_gen=500,
        Qmin=0, Qmax=2, d=1, lower=-2, upper=2):
    
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
            Fitness[i] = func(series,Sol[i],n_preds)
    
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
                Fnew = func(series,S[i],n_preds)
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
    
    def func(series,x,n_preds):
        # print(x)
        # print(xs)
        alpha=x[0]
        m1=SimpleExp_model(series, alpha, n_preds)
        return m1[0]
    
    return bat_algorithm(func, A, r,N_pop=len(x), N_gen=500,  
        Qmin=0, Qmax=2, d=1, lower=-2, upper=2)
 


trainvalue=baAlgorithm(A, r , x, N, D, n_preds)

#训练集数据储存
trainresults=trainvalue[0]
filename = "TrainResults_BA_SimpleExp.csv"
# 打开文件，并使用csv.writer写入数据
with open(filename, mode='w', newline='') as file:
    writer = csv.writer(file)
#     # for row in data2:
#     #     writer.writerow(float(item) for item in row)
#     writer.writerow([float(item) for item in trainresults])

# # 打开文件以写入模式
# with open('output.csv', 'w', newline='') as file:
#     writer = csv.writer(file)
    
    # 检查trainresults是否为可迭代对象（如数组或列表）
    if isinstance(trainresults, (list, np.ndarray)):
        writer.writerow([float(item) for item in trainresults])
    else:
        # 如果trainresults是单个值
        writer.writerow([float(trainresults)])

############################################# Test Data Set#########################################################

################# Begin Import the test data1(test data japan1.csv) ##################
with open(r'E:\L\Kyushu\Kyushu\Kyushu_Consumption\test data japan1.csv', 'r', newline='') as csvfile:
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
with open(r'E:\L\Kyushu\Kyushu\Kyushu_Consumption\test data japan2.csv', 'r', newline='') as csvfile2:
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
with open(r'E:\L\Kyushu\Kyushu\Kyushu_Consumption\test data japan3.csv', 'r', newline='') as csvfile2:
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


############################# Begin SimpleExpSmoothing model ##############################
series = test_japan1 #Define the seasonal data list
n_preds=24
# Define the SimpleExpSmoothing model
def test_SimpleExp_model(series, alpha, n_preds):
    # 读取时间序列数据
    data = pd.read_csv(r'E:\L\Kyushu\Kyushu\Kyushu_Consumption\test data japan3.csv', parse_dates=['date'], index_col='date')

    # 设置平滑系数
    # alpha = 0.2

    # 初始化预测值为第一个观察值
    data['smoothed'] = data['value'].iloc[0]

    # 计算指数平滑
    for i in range(1, len(data)):
        data['smoothed'].iloc[i] = alpha * data['value'].iloc[i] + (1 - alpha) * data['smoothed'].iloc[i - 1]

    # 打印结果
    # print(data)
    array_forecast = data['smoothed'].tolist()
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
############################# End SimpleExpSmoothing model ##############################

alpha = trainvalue[1]
data1 = test_SimpleExp_model(series, alpha, n_preds)

########################################  测试集上的数据储存 ######################################## 
testresults = data1[0]
filename = "TestResults_BA_SimpleExp.csv"
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
filename = "BA_SimpleExp.csv"
# 打开文件，并使用csv.writer写入数据
with open(filename, mode='w', newline='') as file:
    writer = csv.writer(file)
    # for row in data2:
    #     writer.writerow(float(item) for item in row)
    writer.writerow([float(item) for item in data2])

print(data2)
print(testresults)