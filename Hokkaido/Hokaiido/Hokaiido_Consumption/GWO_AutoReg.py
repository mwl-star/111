import csv
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import matplotlib
from random import sample, randint, random
from statsmodels.tsa.holtwinters import Holt

################# Begin Import the train data1(training data japan1.csv) ##################
with open(r'E:\lunwen2\fuxian2\Hokkaido\Hokaiido\Hokaiido_Consumption\training data japan1.csv', 'r',encoding='utf-8', newline='') as csvfile:
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
with open(r'E:\lunwen2\fuxian2\Hokkaido\Hokaiido\Hokaiido_Consumption\training data japan2.csv', 'r',encoding='utf-8', newline='') as csvfile2:
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

############################# Begin AutoReg model #############################
series = train_japan1 #Define the seasonal data list
n_preds=24
def autoreg_model(series, alpha, n_preds):
    # 创建DataFrame
    df = pd.DataFrame(series)

    # 手动设置自回归参数ϕ1
    phi_1 = alpha  # 您可以根据需要手动设置该值

    # 手动进行预测
    def manual_ar1_prediction(data, phi_1, steps):
        predictions = []
        last_value = data[-1]
        for _ in range(steps):
            next_value = phi_1 * last_value
            predictions.append(next_value)
            last_value = next_value
        return predictions

    # 预测未来24个时间点
    future_steps = 24
    # manual_predictions = manual_ar1_prediction(df['value'].values, phi_1, future_steps)
    manual_predictions = manual_ar1_prediction(series, phi_1, future_steps)
    array_forecast = manual_predictions
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
    return MAPE,forecast
############################# END AutoReg model ##############################


x = np.random.rand(N, D) * (x_max - x_min) + x_min
################### Begin GWO algorithm ###############################
def gwoAlgorithm(x, N, D, n_preds):
    # 参数设置
    pop_size = 30
    # dim = 5
    # lower_bound = -10
    # upper_bound = 10
    max_iter = 500
    # 定义目标函数（可以根据实际问题更改）
    # def objective_function(x):
    #     return np.sum(x**2)

    def func(series, x, n_preds):
        # print(x)
        alpha=x[0]
        m1 = autoreg_model(series, alpha, n_preds)
        return m1[0]

    # 初始化灰狼种群
    # def initialize_population(pop_size, D, lower_bound, upper_bound):
    #     return lower_bound + (upper_bound - lower_bound) * np.random.rand(pop_size, D)

    # 更新 alpha, beta 和 delta 灰狼
    def update_alpha_beta_delta(population, fitness):
        alpha1, beta1, delta = np.argsort(fitness)[:3]
        return population[alpha1], population[beta1], population[delta]

    # 灰狼优化算法
    def gwo(func, pop_size, D, max_iter,x):
        # 初始化种群
        #population = initialize_population(pop_size, D, lower_bound, upper_bound)
        population = x
        #fitness = np.apply_along_axis(func(series,slen,x,n_preds), 1, population)
        fitness = np.apply_along_axis(func, 1, population, series, n_preds)
        
        # 初始化 alpha, beta 和 delta 灰狼
        alpha1, beta1, delta = update_alpha_beta_delta(population, fitness)
        
        # 主循环
        for iteration in range(max_iter):
            a = 2 - 2 * iteration / max_iter  # a 从 2 线性递减到 0
            
            for i in range(pop_size):
                for j in range(D):
                    r1, r2 = np.random.rand(), np.random.rand()
                    A1, C1 = 2 * a * r1 - a, 2 * r2
                    D_alpha = abs(C1 * alpha1[j] - population[i, j])
                    X1 = alpha1[j] - A1 * D_alpha
                    
                    r1, r2 = np.random.rand(), np.random.rand()
                    A2, C2 = 2 * a * r1 - a, 2 * r2
                    D_beta = abs(C2 * beta1[j] - population[i, j])
                    X2 = beta1[j] - A2 * D_beta
                    
                    r1, r2 = np.random.rand(), np.random.rand()
                    A3, C3 = 2 * a * r1 - a, 2 * r2
                    D_delta = abs(C3 * delta[j] - population[i, j])
                    X3 = delta[j] - A3 * D_delta
                    
                    population[i, j] = (X1 + X2 + X3) / 3
            
            # 计算新种群的适应度
            fitness = np.apply_along_axis(func, 1, population, series, n_preds)
            
            # 更新 alpha, beta 和 delta 灰狼
            alpha1, beta1, delta = update_alpha_beta_delta(population, fitness)
            
            # 打印当前最佳结果
            #print(f"Iteration {iteration + 1}, Best fitness: {fitness.min()}")
        return  fitness.min(),alpha1
    return gwo(func, pop_size, D, max_iter,x)



# 运行灰狼优化算法
# best_position, best_fitness = gwo(objective_function, pop_size, D, lower_bound, upper_bound, max_iter)
#best_position, best_fitness = gwoAlgorithm(x, N, D, slen, n_preds)

value = gwoAlgorithm(x, N, D, n_preds)
best_position = value[1]
best_fitness = value[0]

#训练集数据储存
trainresults=best_fitness
filename = "TrainResults_GWO_AutoReg.csv"
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
with open(r'E:\lunwen2\fuxian2\Hokkaido\Hokaiido\Hokaiido_Consumption\test data japan1.csv', 'r', encoding='utf-8', newline='') as csvfile:
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
with open(r'E:\lunwen2\fuxian2\Hokkaido\Hokaiido\Hokaiido_Consumption\test data japan2.csv', 'r',encoding='utf-8', newline='') as csvfile2:
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

############################# Begin AutoReg model #############################
series = test_japan1 #Define the seasonal data list
n_preds=24
def test_autoreg_model(series, alpha, n_preds):
    # 创建DataFrame
    df = pd.DataFrame(series)

    # 手动设置自回归参数ϕ1
    phi_1 = alpha  # 您可以根据需要手动设置该值

    # 手动进行预测
    def manual_ar1_prediction(data, phi_1, steps):
        predictions = []
        last_value = data[-1]
        for _ in range(steps):
            next_value = phi_1 * last_value
            predictions.append(next_value)
            last_value = next_value
        return predictions

    # 预测未来24个时间点
    future_steps = 24
    # manual_predictions = manual_ar1_prediction(df['value'].values, phi_1, future_steps)
    manual_predictions = manual_ar1_prediction(series, phi_1, future_steps)
    array_forecast = manual_predictions
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

    return MAPE,forecast
############################# END AutoReg model ##############################

alpha = best_position
data1 = test_autoreg_model(series, alpha, n_preds)

########################################  测试集上的数据储存 ######################################## 
testresults = data1[0]
filename = "TestResults_GWO_AutoReg.csv"
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
filename = "GWO_AutoReg.csv"
# 打开文件，并使用csv.writer写入数据
with open(filename, mode='w', newline='') as file:
    writer = csv.writer(file)
    # for row in data2:
    #     writer.writerow(float(item) for item in row)
    writer.writerow([float(item) for item in data2])

print(data2)
print(testresults)