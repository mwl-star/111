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

################# Begin Import the train data3(actual data)(training data japan2.csv) ##############
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
D=1 # Dimension

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



################### Begin PSO algorithm ###############################
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
        m = LSTM_model(train_japan2,train_japan3, alpha)
        return m[0]


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
    # plt.show()
    return gb[T - 1],x_best

# w=1
# c1=1.5
# c2=1.5
# m=psoAlgorithm(w, c1 , c2, x, N, D,slen, n_preds)
# print(m[0])
################### END PSO algorithm ###############################


################### Begin BA algorithm ###############################
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
# A=0.5
# r=0.5 
def baAlgorithm(A, r, x, N, D, n_preds):
    def bat_algorithm(func, A, r ,N_pop=len(x), N_gen=1,
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
    # def test_function(u):
    #     a = u ** 2
    #     return a.sum(axis=0)
    
    def func(x):
        # print(x)
        # print(xs)
        alpha=x[0]
        m2=LSTM_model(train_japan2,train_japan3, alpha)
        return m2[0]
    
    return bat_algorithm(func, A, r, N_pop=len(x), N_gen=1,  
        Qmin=0, Qmax=2, d=1, lower=-2, upper=2)

# print(deAlgorithm(cr, x, N, D, slen, n_preds))

################### End BA algorithm ###############################


################### Hyper-heuristic(GA based) algorithm ###############################
DNA_SIZE = 3
POP_SIZE = 3
CROSSOVER_RATE = 0.8
CROSSOVER_RATE_MIN = 0.4
CROSSOVER_RATE_MAX = 0.99
MUTATION_RATE = 0.005
MUTATION_RATE_MIN = 0.0001
MUTATION_RATE_MAX = 0.1
N_GENERATIONS = 1
# X_BOUND = [-3, 3]
# Y_BOUND = [-3, 3]

# #Define the population range list
# wlist=[0.4,0.5,0.6,0.7,0.8] #choose for w value
# c1list=[0,1.0,2.0,3.0,4.0] #choose for c1 value
# c2list=[0,1.0,2.0,3.0,4.0] #choose for c2 value
# Alist=[0,0.3,0.5,0.8,1.0] #choose for cr value
# rlist=[0,0.3,0.5,0.8,1.0]
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
def get_fitness(pop1,pop2,pop3,x,D,n_preds):
    ################################################## 
    ######### Compute the fitness of group1 ##########
    ##################################################
    # Choose the x1 for DE
    x1=[]
    for i in range(0,5):
        x1.append(x[i])
    x1=np.array(x1)
    # Caculate the psofitness1 by PSO
    w=pop1[0]
    c1=pop1[1]
    c2=pop1[2]
    N1=5
    psofitness1=psoAlgorithm(w, c1 , c2, x1, N1, D,n_preds)
    # print(psofitness1[0])
    # Choose the x2 for BA
    x2=[]
    for i in range(5,10):
        x2.append(x[i])
    #Caculate the defitness1 by BA
    A=pop1[1]
    r=pop1[2]
    N2=5
    bafitness1=baAlgorithm(A, r, x2, N2, D, n_preds)
    # print(defitness1[0])
    # Compeare the fitness1 between BA and DE
    # min_fitness1= min(num1, num2)
    # 初始化最小值和对应的x值
    min_fitness1 = float('inf')
    min_x1 = None
    # 循环遍历x值并比较函数值
    if psofitness1[0] <= bafitness1[0]:
        min_fitness1 = psofitness1[0]
        min_x1 = psofitness1[1]
    else:
        min_fitness1 = bafitness1[0]
        min_x1 = bafitness1[1]
    ##################################################
    ######### Compute the fitness of group2 ##########
    ##################################################
    # Choose the x3 for PSO
    x3=[]
    for i in range(10,15):
        x3.append(x[i])
    x3=np.array(x3)
    # Caculate the psofitness2 by DE
    w=pop2[0]
    c1=pop2[1]
    c2=pop2[2]
    N3=5
    psofitness2=psoAlgorithm(w, c1 , c2, x3, N3, D, n_preds)
    # print(psofitness2[0])
    # Choose the x4 for BA
    x4=[]
    for i in range(15,20):
        x4.append(x[i])
    # Caculate the defitness2 by BA
    A=pop2[1]
    r=pop2[2]
    N4=5
    bafitness2=baAlgorithm(A, r, x4, N4, D, n_preds)
    # print(defitness2[0])
    # Compeare the fitness2 between PSO and DE
    # min_fitness1= min(num1, num2)
    # 初始化最小值和对应的x值
    min_fitness2 = float('inf')
    min_x2 = None
    # 循环遍历x值并比较函数值
    if psofitness2[0] <= bafitness2[0]:
        min_fitness2 = psofitness2[0]
        min_x2 = psofitness2[1]
    else:
        min_fitness2 = bafitness2[0]
        min_x2 = bafitness2[1]
    ##################################################
    ######### Compute the fitness of group3 ##########
    ##################################################
    # Choose the x5 for PSO
    x5=[]
    for i in range(20,25):
        x5.append(x[i])
    x5=np.array(x5)
    # Caculate the psofitness3 by PSO
    w=pop3[0]
    c1=pop3[1]
    c2=pop3[2]
    N5=5
    psofitness3=psoAlgorithm(w, c1 , c2, x5, N5, D, n_preds)
    # print(psofitness3[0])
    # Choose the x6 for BA
    x6=[]
    for i in range(25,30):
        x6.append(x[i])
    # Caculate the defitness3 by BA
    A=pop3[1]
    r=pop3[2]
    N6=5
    bafitness3=baAlgorithm(A, r, x6, N6, D, n_preds)
    # print(defitness3[0])
    # Compeare the fitness3 between PSO and DE
    # min_fitness1= min(num1, num2)
    # 初始化最小值和对应的x值
    min_fitness3 = float('inf')
    min_x3 = None
    # 循环遍历x值并比较函数值
    if psofitness3[0] <= bafitness3[0]:
        min_fitness3 = psofitness3[0]
        min_x3 = psofitness3[1]
    else:
        min_fitness3 = bafitness3[0]
        min_x3 = bafitness3[1]
    # Generate the array for the fitness and min_x
    fitness=[min_fitness1,min_fitness2,min_fitness3 ]
    min_x=[min_x1,min_x2,min_x3]
    return fitness,min_x

# def plot_3d(ax):

# 	X = np.linspace(*X_BOUND, 100)
# 	Y = np.linspace(*Y_BOUND, 100)
# 	X,Y = np.meshgrid(X, Y)
# 	Z = F(X, Y)
# 	ax.plot_surface(X,Y,Z,rstride=1,cstride=1,cmap=cm.coolwarm)
# 	ax.set_zlim(-10,10)
# 	ax.set_xlabel('x')
# 	ax.set_ylabel('y')
# 	ax.set_zlabel('z')
# 	plt.pause(3)
# 	plt.show()

def crossover_and_mutation(pop, CROSSOVER_RATE_MIN, CROSSOVER_RATE_MAX, CROSSOVER_RATE, k, N_GENERATIONS,MUTATION_RATE_MIN,MUTATION_RATE_MAX,MUTATION_RATE):
	new_pop = []
	fitback = get_fitness(pop[0],pop[1],pop[2],x,D,n_preds)
	fitvalue = fitback[0]	
	fitvaluetotal = sum(fitvalue)
	for father in pop:		#遍历种群中的每一个个体，将该个体作为父亲
		child = father		#孩子先得到父亲的全部基因（这里我把一串二进制串的那些0，1称为基因）
		# childid = pop.index(child)
		target_row = np.array(child)
		row_index = -1
		for i, row in enumerate(pop):
			if np.all(row == target_row):
				row_index = i
				break
		fitvalueup=fitvalue[row_index]
		CROSSOVER_RATE =  CROSSOVER_RATE_MIN + 2*(1-k/N_GENERATIONS)*(fitvalueup/fitvaluetotal)*CROSSOVER_RATE_MAX
		if np.random.rand() < CROSSOVER_RATE:			#产生子代时不是必然发生交叉，而是以一定的概率发生交叉
			mother = pop[np.random.randint(POP_SIZE)]	#再种群中选择另一个个体，并将该个体作为母亲
			cross_points = np.random.randint(low=0, high=DNA_SIZE)	#随机产生交叉的点
			child[cross_points:] = mother[cross_points:]		#孩子得到位于交叉点后的母亲的基因
		mutation(child,father,MUTATION_RATE_MIN,MUTATION_RATE_MAX,MUTATION_RATE,k,N_GENERATIONS,fitvalueup,fitvaluetotal)	#每个后代有一定的机率发生变异
		new_pop.append(child)
                            
	return new_pop

def mutation(child, father,MUTATION_RATE_MIN,MUTATION_RATE_MAX,MUTATION_RATE,k,N_GENERATIONS,fitvalueup,fitvaluetotal):
	MUTATION_RATE = MUTATION_RATE_MIN + 2*(1-k/N_GENERATIONS)*(fitvalueup/fitvaluetotal)*MUTATION_RATE_MAX
	if np.random.rand() < MUTATION_RATE: 				#以MUTATION_RATE的概率进行变异
		mutate_point = np.random.randint(0, DNA_SIZE)	#随机产生一个实数，代表要变异基因的位置
		child[mutate_point] = father[mutate_point] 	#将变异点的二进制为反转

def select(pop, fitness):    # nature selection wrt pop's fitness
    # 为了选择概率最小的元素，我们需要将概率数组 p 反转（或取反）
    # 注意：概率必须是正数，并且总和为1，因此我们需要对概率进行归一化
    # total_sum = sum(item[0] if isinstance(item, np.ndarray) else item for item in fitness)
    # p_normalized = (np.array(fitness))/(total_sum)  # 归一化概率

    fitness = [np.array([val]) if not isinstance(val, np.ndarray) else val for val in fitness]
    fitness_array = np.hstack(fitness)
    total_sum = fitness_array.sum()
    p_normalized = fitness_array / total_sum
    p_min = 1 - p_normalized  # 计算每个元素不被选中的概率
    p_min_normalized = p_min / p_min.sum()  # 归一化不被选中的概率
    idx = np.random.choice(np.arange(POP_SIZE), size=POP_SIZE, replace=True,
                           p=p_min_normalized)
    return pop[idx]

def print_info(pop1,pop2,pop3,x,D,n_preds):
    fitness = get_fitness(pop1,pop2,pop3,x,D,n_preds)
    fitnessvalue = fitness[0]
    fitnessx = fitness[1]
    fitnessvalue = [np.array([val]) if not isinstance(val, np.ndarray) else val for val in fitnessvalue]
    fitness_array2 = np.hstack(fitnessvalue)
    min_fitness_index = np.argmin(fitness_array2)
    print("min_fitness:", fitnessvalue[min_fitness_index])
    print("Parameter of holt-winters:", fitnessx[min_fitness_index])
    # x,y = translateDNA(pop)
    print("Parameter of low-level heuristics:", pop[min_fitness_index])
    # print("(x, y):", (x[max_fitness_index], y[max_fitness_index]))
    return fitnessx[min_fitness_index],fitnessvalue[min_fitness_index]


x = np.random.rand(N, D) * (x_max - x_min) + x_min
if __name__ == "__main__":
	# fig = plt.figure()
	# ax = Axes3D(fig)	
	# plt.ion()#将画图模式改为交互模式，程序遇到plt.show不会暂停，而是继续执行
	# plot_3d(ax)

	# pop = np.random.randint(2, size=(POP_SIZE, DNA_SIZE*2)) #matrix (POP_SIZE, DNA_SIZE)
    pop = [pop1,pop2,pop3]
    # pop = np.array(pop)
    k = 0
    for _ in range(N_GENERATIONS):#迭代N代
		# x,y = translateDNA(pop)
		# if 'sca' in locals(): 
		# 	sca.remove()
		# sca = ax.scatter(x, y, F(x,y), c='black', marker='o');plt.show();plt.pause(0.1)
     k += 1 
     pop = np.array(crossover_and_mutation(pop, CROSSOVER_RATE_MIN, CROSSOVER_RATE_MAX, CROSSOVER_RATE,k,N_GENERATIONS,MUTATION_RATE_MIN,MUTATION_RATE_MAX,MUTATION_RATE))
    #  pop = crossover_and_mutation(pop, CROSSOVER_RATE)
		# F_values = F(translateDNA(pop)[0], translateDNA(pop)[1])#x, y --> Z matrix
		# fitness = get_fitness(pop)
     fitness = get_fitness(pop1,pop2,pop3,x,D,n_preds)
     fitnessvalue=fitness[0]
    #  fitnessvalue = np.array(fitnessvalue)
     pop = select(pop, fitnessvalue) #选择生成新的种群
	# 修改NumPy的打印选项以显示小数点后的零
    print_info(pop[0],pop[1],pop[2],x,D,n_preds)
	# plt.ioff()
	# plot_3d(ax)
    #训练集数据储存
    trainvalue = print_info(pop[0],pop[1],pop[2],x,D,n_preds)
    trainresults=trainvalue[1]
    filename = "TrainResults_AutoEvo_LSTM.csv"
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

Plotvalue = trainvalue[0]
alpha = Plotvalue[0]
data1 = test_LSTM_model(test_japan2,test_japan3, alpha)

########################################  测试集上的数据储存 ######################################## 
testresults = data1[0]
filename = "TestResults_AutoEvo_LSTM.csv"
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
filename = "AutoEvo_LSTM.csv"
# 打开文件，并使用csv.writer写入数据
with open(filename, mode='w', newline='') as file:
    writer = csv.writer(file)
    # for row in data2:
    #     writer.writerow(float(item) for item in row)
    writer.writerow([float(item) for item in data2])

print(data2)
print(testresults)