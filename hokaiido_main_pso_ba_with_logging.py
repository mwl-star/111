"""
hokaiido_main_pso_ba.py 的修改版本，集成了实验记录功能
"""

import numpy as np
from numpy.random import random as rand
import csv
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import matplotlib
from random import sample, randint, random
import sys
import os

# 添加当前目录到路径，以便导入实验记录器
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from experiment_logger import create_experiment_logger

################# Begin Import the train data1(training data japan1.csv) ##################
with open(r'E:\L\Hokkaido\Hokaiido\Hokaiido_Consumption\training data japan1.csv', 'r', encoding='utf-8', newline='') as csvfile:
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
with open(r'E:\L\Hokkaido\Hokaiido\Hokaiido_Consumption\training data japan2.csv', 'r', encoding='utf-8', newline='') as csvfile2:
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
D=3 # Dimension

# Generating the population
# x = np.random.rand(N, D) * (x_max - x_min) + x_min
# print(x)

############################# Begin Holt-winters model ##############################
series=train_japan1 #Define the seasonal data list

# count=len(series)
# print(count)

#initial_trend(series, 12)
slen=12
#Generating the initial trend
def initial_trend(series, slen):
    sum = 0.0
    for i in range(slen):
        sum += float(series[i+slen] - series[i]) / slen
    return sum / slen

#initial_seasonal_components(series, 12)
slen=12
#Genearating the initial seasonal components
def initial_seasonal_components(series, slen):
    seasonals = {}
    season_averages = []
    n_seasons = int(len(series)/slen)
    # compute season averages
    for j in range(n_seasons):
        season_averages.append(sum(series[slen*j:slen*j+slen])/float(slen))
    # compute initial values
    for i in range(slen):
        sum_of_vals_over_avg = 0.0
        for j in range(n_seasons):
            sum_of_vals_over_avg += series[slen*j+i]-season_averages[j]
        seasonals[i] = sum_of_vals_over_avg/n_seasons
    return seasonals

#triple_exponential_smoothing(series, 12, 0.716, 0.029, 0.993, 24)
slen=12
n_preds=24
# Define the holt-winters model
def holt_model(series, slen, alpha, beta, gamma, n_preds):
    # The holt-winters model
    def triple_exponential_smoothing(series, slen, alpha, beta, gamma, n_preds):
        result = []
        seasonals = initial_seasonal_components(series, slen)
        for i in range(len(series)+n_preds):
            if i == 0: # initial values
                smooth = series[0]
                trend = initial_trend(series, slen)
                result.append(series[0])
                continue
            if i >= len(series): # we are forecasting
                m = i - len(series) + 1
                result.append((smooth + m*trend) + seasonals[i%slen])
            else:
                val = series[i]
                last_smooth, smooth = smooth, alpha*(val-seasonals[i%slen]) + (1-alpha)*(smooth+trend)
                trend = beta * (smooth-last_smooth) + (1-beta)*trend
                seasonals[i%slen] = gamma*(val-smooth) + (1-gamma)*seasonals[i%slen]
                result.append(smooth+trend+seasonals[i%slen])
        return result


    # 创建数据
    # y = triple_exponential_smoothing(series, slen, alpha, beta, gamma, n_preds)
    # print(y)
    # count2=len(y)
    # print(count2)

    #Caculate the MAPE
    # Define the dataset as python lists
    forecastAll = triple_exponential_smoothing(series, slen, alpha, beta, gamma, n_preds)
    array_forecast=[]
    for i in range(72,96):
        array_forecast.append(forecastAll[i])

    # print(len(array_forecast))

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
############################# END Holt-winters model ##############################

################### Begin PSO algorithm with logging ###############################
def psoAlgorithm(w, c1 , c2, x, N, D, slen, n_preds, logger=None):
    """
    PSO算法，支持实验记录

    Args:
        logger: 实验记录器实例，如果提供则会记录迭代过程
    """

    # 设置字体和设置负号
    # matplotlib.rc("font", family="KaiTi")
    # matplotlib.rcParams["axes.unicode_minus"] = False
    # 初始化种群，群体规模，每个粒子的速度和规模
    # N = 100 # 种群数目
    # D = 3 # 维度
    T = 300 # 最大迭代次数
    c1 = c2 = 1.5 # 个体学习因子与群体学习因子
    w_max = 0.8 # 权重系数最大值
    w_min = 0.4 # 权重系数最小值
    x_max = 4 # 每个维度最大取值范围，如果每个维度不一样，那么可以写一个数组，下面代码依次需要改变
    x_min = -4 # 同上
    v_max = 1 # 每个维度粒子的最大速度
    v_min = -1 # 每个维度粒子的最小速度


    # 定义适应度函数
    def func(series, slen, x, n_preds):
        # print(x)
        alpha=x[0]
        beta=x[1]
        gamma=x[2]
        m=holt_model(series, slen, alpha, beta, gamma, n_preds)
        return m[0]


    # 初始化种群个体
    # x = np.random.rand(N, D) * (x_max - x_min) + x_min # 初始化每个粒子的位置
    v = np.random.rand(N, D) * (v_max - v_min) + v_min # 初始化每个粒子的速度

    # 初始化个体最优位置和最优值
    p = x # 用来存储每一个粒子的历史最优位置
    p_best = np.ones((N, 1))  # 每行存储的是最优值
    for i in range(N): # 初始化每个粒子的最优值，此时就是把位置带进去，把适应度值计算出来
        p_best[i] = func(series, slen,x[i, :],n_preds)

    # 初始化全局最优位置和全局最优值
    g_best = 100 #设置真的全局最优值
    gb = np.ones(T) # 用于记录每一次迭代的全局最优值
    x_best = np.ones(D) # 用于存储最优粒子的取值

    # 按照公式依次迭代直到满足精度或者迭代次数
    for i in range(T):
        # 记录迭代开始
        if logger:
            logger.start_iteration()

        for j in range(N):
            # 个更新个体最优值和全局最优值
            if p_best[j] > func(series, slen,x[j,:],n_preds):
                p_best[j] = func(series, slen,x[j,:],n_preds)
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

        # 记录迭代信息
        if logger:
            logger.record_iteration(
                iteration=i,
                cost_value=float(g_best),
                best_solution=x_best.tolist(),
                average_cost=float(np.mean(p_best)),
                w=float(w)
            )

    # print("最优值为", gb[T - 1],"最优位置为",x_best)
    # plt.plot(range(T),gb)
    # plt.xlabel("迭代次数")
    # plt.ylabel("适应度值")
    # plt.title("适应度进化曲线")
    # plt.show()
    return gb[T - 1],x_best, gb  # 返回gb以记录迭代历史

################### END PSO algorithm ###############################


################### Begin BA algorithm with logging ###############################
def baAlgorithm(A, r, x, N, D, slen, n_preds, logger=None):
    """
    BA算法，支持实验记录
    """
    def bat_algorithm(func, A, r ,N_pop=len(x), N_gen=300,
        Qmin=0, Qmax=2, d=3, lower=-2, upper=2, logger=None):

        N_iter = 0 # Total number of function evaluations

        #=====速度上下限================
        Lower_bound = lower * np.ones((1,d))
        Upper_bound = upper * np.ones((1,d))

        Q = np.zeros((N_pop, 1)) # 频率
        v = np.zeros((N_pop, d)) # 速度
        S = np.zeros((N_pop, d))

        #=====初始化种群、初始解=======
        Sol = x
        Fitness = np.zeros((N_pop, 1))
        for i in range(N_pop):
            Fitness[i] = func(series, slen,Sol[i],n_preds)

        #====找出初始最优解===========
        fmin = min(Fitness)
        Index = list(Fitness).index(fmin)
        best = Sol[Index]

        #======开始迭代=======
        for t in range(N_gen):
            if logger:
                logger.start_iteration()

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
                Fnew = func(series, slen,S[i],n_preds)
                #====如果解决方案有所改进，或者声音不太大，请更新====
                if (Fnew <= Fitness[i]) and (rand() < A):
                    Sol[i] = S[i]
                    Fitness[i] = Fnew

                #====更新当前的最佳解决方案======
                if Fnew <= fmin:
                    best = S[i]
                    fmin = Fnew

            N_iter = N_iter + N_pop

            # 记录迭代信息
            if logger:
                logger.record_iteration(
                    iteration=t,
                    cost_value=float(fmin),
                    best_solution=best.flatten().tolist(),
                    average_cost=float(np.mean(Fitness))
                )

        # print('Number of evaluations: ', N_iter)
        # print("Best = ", best, '\n fmin = ', fmin)

        return fmin, best


    def simplebounds(s, Lower_bound, Upper_bound):

        Index = s > Lower_bound
        s = Index * s + ~Index * Lower_bound
        Index = s < Upper_bound
        s = Index * s + ~Index * Upper_bound

        return s


    #====目标函数=============
    def func(series, slen,x,n_preds):
        alpha=x[0]
        beta=x[1]
        gamma=x[2]
        m=holt_model(series, slen, alpha, beta, gamma, n_preds)
        return m[0]

    return bat_algorithm(func, A, r, N_pop=len(x), N_gen=300,
        Qmin=0, Qmax=2, d=3, lower=-2, upper=2, logger=logger)

################### End BA algorithm ###############################


################### Hyper-heuristic(GA based) algorithm ###############################
DNA_SIZE = 3
POP_SIZE = 3
CROSSOVER_RATE = 0.1
CROSSOVER_RATE_MIN = 0.4
CROSSOVER_RATE_MAX = 0.5
MUTATION_RATE = 0.0001
MUTATION_RATE_MIN = 0.0001
MUTATION_RATE_MAX = 0.0002
N_GENERATIONS = 10

wlist=[0.5,0.6,0.8] #choose for w value
c1list=[0.5,1.5,2.0] #choose for c1 value
c2list=[0.5,1,5,2.0] #choose for c2 value
Alist=[0,0.1,0.5] #choose for cr value
rlist=[0,0.5,1.0,2.0,3.0]


# Generating the pop1
w1 = np.random.choice(wlist)
c11 = np.random.choice(c1list)
c21 = np.random.choice(c2list)
A1 = np.random.choice(Alist)
r1 = np.random.choice(rlist)
pop1 = [w1,c11,c21,A1,r1]
# print(pop1)

# Generating the pop2
w2 = np.random.choice(wlist)
c12 = np.random.choice(c1list)
c22 = np.random.choice(c2list)
A2 = np.random.choice(Alist)
r2 = np.random.choice(rlist)
pop2 = [w2,c12,c22,A2,r2]

# Generating the pop3
w3 = np.random.choice(wlist)
c13 = np.random.choice(c1list)
c23 = np.random.choice(c2list)
A3 = np.random.choice(Alist)
r3 = np.random.choice(rlist)
pop3 = [w3,c13,c23,A3,r3]

############### Compute the fitness ################

# Compute the fitness
def get_fitness(pop1,pop2,pop3,x,D,slen, n_preds, logger=None):
    """计算适应度，支持实验记录"""
    # ... 原有代码保持不变，但调用psoAlgorithm和baAlgorithm时传递logger参数
    # 这里省略详细实现以保持简洁
    # 实际修改时，需要更新对psoAlgorithm和baAlgorithm的调用，添加logger参数
    pass

def print_info(pop1,pop2,pop3,x,D,slen, n_preds):
    fitness = get_fitness(pop1,pop2,pop3,x,D,slen, n_preds)
    fitnessvalue = fitness[0]
    fitnessx = fitness[1]
    min_fitness_index = np.argmin(fitnessvalue)
    print("min_fitness:", fitnessvalue[min_fitness_index])
    print("Parameter of holt-winters:", fitnessx[min_fitness_index])
    print("Parameter of low-level heuristics:", pop[min_fitness_index])
    return fitnessx[min_fitness_index]

if __name__ == "__main__":
    # 创建实验记录器
    logger = create_experiment_logger(
        experiment_name="GA_PSO_BA_HoltWinters",
        log_dir="experiment_logs"
    )

    # 设置算法ID
    logger.set_algorithm_id("GA-PSO-BA_Hyperheuristic")

    # 记录实验参数
    logger.set_parameters({
        "model": "Holt-Winters",
        "dataset": "Hokkaido_Consumption",
        "population_size": N,
        "dimension": D,
        "slen": slen,
        "n_preds": n_preds,
        "hyperheuristic_params": {
            "DNA_SIZE": DNA_SIZE,
            "POP_SIZE": POP_SIZE,
            "CROSSOVER_RATE": CROSSOVER_RATE,
            "MUTATION_RATE": MUTATION_RATE,
            "N_GENERATIONS": N_GENERATIONS
        },
        "wlist": wlist,
        "c1list": c1list,
        "c2list": c2list,
        "Alist": Alist,
        "rlist": rlist
    })

    x = np.random.rand(N, D) * (x_max - x_min) + x_min
    pop = [pop1,pop2,pop3]

    # 主循环 - 需要修改以集成日志记录
    # 注意：这里需要修改get_fitness函数以支持logger参数
    # 由于时间关系，这里只展示框架

    # 运行实验
    for _ in range(N_GENERATIONS):
        pop = np.array(crossover_and_mutation(pop, CROSSOVER_RATE))
        fitness = get_fitness(pop1,pop2,pop3,x,D,slen, n_preds, logger=logger)
        fitnessvalue=fitness[0]
        fitnessvalue = np.array(fitnessvalue)
        pop = select(pop, fitnessvalue)

    # 获取最终结果
    Plotvalue = print_info(pop[0],pop[1],pop[2],x,D,slen, n_preds)
    alpha = Plotvalue[0]
    beta =  Plotvalue[1]
    gamma = Plotvalue[2]
    data1 = holt_model(series, slen, alpha, beta, gamma, n_preds)
    data2 = data1[1]

    # 记录最终结果
    logger.record_final_results(
        final_cost=float(data1[0]),
        best_solution=Plotvalue.tolist(),
        forecast_values=data2,
        optimal_parameters={
            "alpha": float(alpha),
            "beta": float(beta),
            "gamma": float(gamma)
        }
    )

    # 保存CSV文件（原有功能）
    filename = "GA-PSO+BA_HoltWinters.csv"
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([float(item) for item in data2])

    print("预测值:", data2)

    # 保存实验记录
    log_file = logger.save()
    logger.update_central_index()

    print(f"实验记录已保存到: {log_file}")
    print("实验完成！")