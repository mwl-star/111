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
with open(r'E:\L\Hokkaido\Hokaiido\Hokaiido_Generation\training data japan1.csv', 'r', encoding='utf-8', newline='') as csvfile:
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
train_japan1 = []
for s in str_array:
    try:
        train_japan1.append(float(s))
    except ValueError:
        continue  # 跳过无法转换为数字的值
# print(train_japan1)
#####################################END#########################################

################# Begin Import the train data2(actual data)(training data japan2.csv) ##############
with open(r'E:\L\Hokkaido\Hokaiido\Hokaiido_Generation\training data japan2.csv', 'r', encoding='utf-8', newline='') as csvfile2:
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
train_japan2 = []
for s in str_array2:
    try:
        train_japan2.append(float(s))
    except ValueError:
        continue  # 跳过无法转换为数字的值
# print(train_japan2)
#####################################END#########################################



x_max = 1 # The max dimension
x_min = 0 # The min dimension
N=30 # Number of population
D=1 # Dimension

# Generating the populationq
# x = np.random.rand(N, D) * (x_max - x_min) + x_min
# print(x)

############################# Begin SimpleExpSmoothing model ##############################
series = train_japan1 #Define the seasonal data list
n_preds=24
# Define the SimpleExpSmoothing model
def SimpleExp_model(series, alpha, n_preds):
    # 读取时间序列数据
    data = pd.read_csv(r'E:\L\Hokkaido\Hokaiido\Hokaiido_Generation\training data japan3.csv', parse_dates=['date'], index_col='date')

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

x = np.random.rand(N, D) * (x_max - x_min) + x_min
################### Begin PSO algorithm ###############################
def psoAlgorithm(w, c1 , c2, x, N, D, n_preds):


    # 设置字体和设置负号
    # matplotlib.rc("font", family="KaiTi")
    # matplotlib.rcParams["axes.unicode_minus"] = False
    # 初始化种群，群体规模，每个粒子的速度和规模
    # N = 100 # 种群数目
    # D = 3 # 维度
    T = 500 # 最大迭代次数
    c1 = c2 = 1.5 # 个体学习因子与群体学习因子
    w_max = 0.8 # 权重系数最大值
    w_min = 0.4 # 权重系数最小值
    x_max = 4 # 每个维度最大取值范围，如果每个维度不一样，那么可以写一个数组，下面代码依次需要改变
    x_min = -4 # 同上
    v_max = 1 # 每个维度粒子的最大速度
    v_min = -1 # 每个维度粒子的最小速度


    # 定义适应度函数
    def func(series, x, n_preds):
        # print(x)
        alpha=x[0]
        m1=SimpleExp_model(series, alpha, n_preds)
        return m1[0]


    # 初始化种群个体
    # x = np.random.rand(N, D) * (x_max - x_min) + x_min # 初始化每个粒子的位置
    v = np.random.rand(N, D) * (v_max - v_min) + v_min # 初始化每个粒子的速度

    # 初始化个体最优位置和最优值
    p = x # 用来存储每一个粒子的历史最优位置
    p_best = np.ones((N, 1))  # 每行存储的是最优值
    for i in range(N): # 初始化每个粒子的最优值，此时就是把位置带进去，把适应度值计算出来
        p_best[i] = func(series, x[i, :], n_preds) 

    # 初始化全局最优位置和全局最优值
    g_best = 100 #设置真的全局最优值
    gb = np.ones(T) # 用于记录每一次迭代的全局最优值
    x_best = np.ones(D) # 用于存储最优粒子的取值

    # 按照公式依次迭代直到满足精度或者迭代次数
    for i in range(T):
        for j in range(N):
            # 个更新个体最优值和全局最优值
            if p_best[j] > func(series, x[j,:],n_preds):
                p_best[j] = func(series, x[j,:],n_preds)
                p[j,:] = x[j,:].copy()
            # p_best[j] = func(x[j,:]) if func(x[j,:]) < p_best[j] else p_best[j]
            # 更新全局最优值
            if g_best > p_best[j]:
                g_best = p_best[j]
                x_best = x[j,:].copy()   # 一定要加copy，否则后面x[j,:]更新也会将x_best更新
            # 计算动态惯性权重
            w = w_max - (w_max - w_min) * i / T
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
    # plt.show()
    return gb[T - 1],x_best

w=1
c1=1.5
c2=1.5
trainvalue=psoAlgorithm(w, c1 , c2, x, N, D, n_preds)
################### END PSO algorithm ###############################

#训练集数据储存
trainresults=trainvalue[0]
filename = "TrainResults_PSO_SimpleExp.csv"
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
with open(r'E:\L\Hokkaido\Hokaiido\Hokaiido_Generation\test data japan1.csv', 'r', encoding='utf-8', newline='') as csvfile:
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
test_japan1 = []
for s in str_array:
    try:
        test_japan1.append(float(s))
    except ValueError:
        continue  # 跳过无法转换为数字的值
# print(train_japan1)
#####################################END#########################################

################# Begin Import the test data2(actual data)(test data japan2.csv) ##############
with open(r'E:\L\Hokkaido\Hokaiido\Hokaiido_Generation\test data japan2.csv', 'r', encoding='utf-8', newline='') as csvfile2:
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
test_japan2 = []
for s in str_array2:
    try:
        test_japan2.append(float(s))
    except ValueError:
        continue  # 跳过无法转换为数字的值
# print(train_japan2)
#####################################END#########################################

################# Begin Import the test data3(actual data)(test data japan3.csv) ##############
with open(r'E:\L\Hokkaido\Hokaiido\Hokaiido_Generation\test data japan3.csv', 'r', encoding='utf-8', newline='') as csvfile2:
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
test_japan3 = []
for s in str_array2:
    try:
        test_japan3.append(float(s))
    except ValueError:
        continue  # 跳过无法转换为数字的值
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
    data = pd.read_csv(r'E:\L\Hokkaido\Hokaiido\Hokaiido_Generation\test data japan3.csv', parse_dates=['date'], index_col='date')

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
filename = "TestResults_PSO_SimpleExp.csv"
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
filename = "PSO_SimpleExp.csv"
# 打开文件，并使用csv.writer写入数据
with open(filename, mode='w', newline='') as file:
    writer = csv.writer(file)
    # for row in data2:
    #     writer.writerow(float(item) for item in row)
    writer.writerow([float(item) for item in data2])

print(data2)
print(testresults)