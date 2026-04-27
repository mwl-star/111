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
import matplotlib.pyplot as plt



################# Begin Import the data of Hyper#################################
with open(r'E:\L\Tohoku\tohoku\tohoku_Generation\Plot_AutoReg.csv', 'r', newline='') as csvfile:
    reader = csv.reader(csvfile)
    # Skip the title row（if have）
    next(reader)
    # Define the index of columns
    columns_of_interest = [0]  # the index begin from 0
    # Define a list
    str_array = []
    for row in reader:
        selected_data = [row[col] for col in columns_of_interest]  # choose the columns
        # Put the item into the list
        str_array.append(selected_data[0])
# Change the string item of list into float
Hyper = [float(s) for s in str_array]
#####################################END#########################################

################# Begin Import the data of PSO#################################
with open(r'E:\L\Tohoku\tohoku\tohoku_Generation\Plot_AutoReg.csv', 'r', newline='') as csvfile:
    reader = csv.reader(csvfile)
    # Skip the title row（if have）
    next(reader)
    # Define the index of columns
    columns_of_interest = [1]  # the index begin from 0
    # Define a list
    str_array = []
    for row in reader:
        selected_data = [row[col] for col in columns_of_interest]  # choose the columns
        # Put the item into the list
        str_array.append(selected_data[0])
# Change the string item of list into float
PSO = [float(s) for s in str_array]
#####################################END#########################################

################# Begin Import the data of BA#################################
with open(r'E:\L\Tohoku\tohoku\tohoku_Generation\Plot_AutoReg.csv', 'r', newline='') as csvfile:
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
BA = [float(s) for s in str_array]
#####################################END#########################################

################# Begin Import the data of DE#################################
with open(r'E:\L\Tohoku\tohoku\tohoku_Generation\Plot_AutoReg.csv', 'r', newline='') as csvfile:
    reader = csv.reader(csvfile)
    # Skip the title row（if have）
    next(reader)
    # Define the index of columns
    columns_of_interest = [3]  # the index begin from 0
    # Define a list
    str_array = []
    for row in reader:
        selected_data = [row[col] for col in columns_of_interest]  # choose the columns
        # Put the item into the list
        str_array.append(selected_data[0])
# Change the string item of list into float
DE = [float(s) for s in str_array]
#####################################END#########################################


################# Begin Import the data of GWO#################################
with open(r'E:\L\Tohoku\tohoku\tohoku_Generation\Plot_AutoReg.csv', 'r', newline='') as csvfile:
    reader = csv.reader(csvfile)
    # Skip the title row（if have）
    next(reader)
    # Define the index of columns
    columns_of_interest = [4]  # the index begin from 0
    # Define a list
    str_array = []
    for row in reader:
        selected_data = [row[col] for col in columns_of_interest]  # choose the columns
        # Put the item into the list
        str_array.append(selected_data[0])
# Change the string item of list into float
GWO = [float(s) for s in str_array]
#####################################END#########################################


################# Begin Import the data of WOA#################################
with open(r'E:\L\Tohoku\tohoku\tohoku_Generation\Plot_AutoReg.csv', 'r', newline='') as csvfile:
    reader = csv.reader(csvfile)
    # Skip the title row（if have）
    next(reader)
    # Define the index of columns
    columns_of_interest = [5]  # the index begin from 0
    # Define a list
    str_array = []
    for row in reader:
        selected_data = [row[col] for col in columns_of_interest]  # choose the columns
        # Put the item into the list
        str_array.append(selected_data[0])
# Change the string item of list into float
WOA = [float(s) for s in str_array]
#####################################END#########################################


################# Begin Import the data of Actual#################################
with open(r'E:\L\Tohoku\tohoku\tohoku_Generation\Plot_AutoReg.csv', 'r', newline='') as csvfile:
    reader = csv.reader(csvfile)
    # Skip the title row（if have）
    next(reader)
    # Define the index of columns
    columns_of_interest = [6]  # the index begin from 0
    # Define a list
    str_array = []
    for row in reader:
        selected_data = [row[col] for col in columns_of_interest]  # choose the columns
        # Put the item into the list
        str_array.append(selected_data[0])
# Change the string item of list into float
Actual = [float(s) for s in str_array]
#####################################END#########################################

# 假设你有5个列表
# list1 = AEHyper
list2 = Hyper
list3 = PSO
list4 = BA
list5 = DE
list6 = GWO
list7 = WOA
list8 = Actual

# 创建一个新的图形
plt.figure()

# # 在同一个坐标轴里画出5个列表的值
# # plt.plot(list1, label='AEHyper')
# plt.plot(list2, label='Hyper')
# plt.plot(list3, label='PSO')
# plt.plot(list4, label='BA')
# plt.plot(list5, label='DE')
# plt.plot(list6, label='Actual')
# # 添加图例
# plt.legend()

# # 显示图形
# plt.show()

# 在同一个坐标轴里画出5个列表的值
# plt.plot(list1, label='AEHyper')
plt.plot(list3, label='PSO', marker='d', markerfacecolor='none',linestyle=":",markevery=1,linewidth=1.8)
plt.plot(list4, label='BA', marker='*', linestyle=":",markerfacecolor='none',markevery=2,linewidth=1.8)
plt.plot(list5, label='DE', marker='o', linestyle=":",markerfacecolor='none',markevery=3,linewidth=1.8)
plt.plot(list6, label='GWO', marker='p', linestyle=":",markerfacecolor='none',markevery=3,linewidth=1.8)
plt.plot(list7, label='WOA', marker='h', linestyle=":",markerfacecolor='none',markevery=3,linewidth=1.8)
plt.plot(list2, label='AE-GAPB', marker='v',linestyle="-",markerfacecolor='none',markevery=4,linewidth=1.8)
plt.plot(list8, label='Actual',marker='x',linestyle='-',markevery=5,linewidth=1.4) 
# 添加图例
plt.legend()
plt.savefig("Tohoku_Generation_AutoReg", dpi=750)

# 显示图形
plt.show()