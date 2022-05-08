# -*- coding: utf-8 -*-
"""
Created on Tue Nov  2 20:08:07 2021

@author: WYW
"""
import yaml
import igraph as ig
import matplotlib.pyplot as plt
import random
from numpy import random
import networkx as nx
import copy

import pop_init as init
import revise as rev
import variation as vari
import overlapping as over
import choice as cho


"""
read config
"""
data = []
with open('decd_config.yaml','rb') as f:
    data = list(yaml.safe_load_all(f))      # 读取配置信息
param = data[0]["param"]

"""
 network init
"""
# 网络信息
netPath = param["netPath"]
G = nx.read_edgelist(netPath)
G = G.to_undirected()  # 转换成无向图
n = G.number_of_nodes()  # 获取一个Graph对象中node的数量
# 基于这些连边使用igraph创建一个新网络
Gi = ig.Graph.Read_Edgelist(netPath)
Gi = Gi.subgraph(map(int, G.nodes()))  # G.nodes()获取一个Graph对象中的node数组,
Gi = Gi.as_undirected()

"""
 param set
"""
NP = param["NP"]    # 种群数
Gen = param["Gen"]  # 代数
F = param["F"]      # 缩放因子
CR = param["CR"]    # 交叉控制参数
threshold_value = param["thresholdValue"]   # 阈值
real_membership = param["realMembership"]   # 网络的真实划分，用于NMI计算

"""
DECD算法检测network
"""
D = n  #个体维数
Fit = []   # 记录分数
pop = []   # 种群
domain = [] # 种群搜索范围的边界约束[Lbound,Ubound]
exetime = 1  # 进化代数
best_Q_in_history = []  # 记录每代种群最优个体的Q值
best_X_in_history = []  # 记录每代种群中的最优个体

# 种群初始化
init.pop_init(pop,n,NP,Gi,D,domain,Fit)

# Main loop
while exetime <= Gen:
    # 变异操作
    mutation_pop = vari.variation_operation(pop,NP,D,F,domain)
    # 修正操作
    mutation_pop = rev.revise_operation(mutation_pop, n, NP, Gi, threshold_value)
    # 交叉操作
    crossover_pop = over.overlapping_operation(pop, NP, D, CR, mutation_pop)
    # 修正操作
    crossover_pop = rev.revise_operation(crossover_pop, n, NP, Gi, threshold_value)
    # 选择操作
    cho.choice_operation(pop, crossover_pop, NP, Fit, Gi)

    # 记录每一代最优解，用于绘制收敛曲线
    best_Q_in_history.append(max(Fit))  # 纵坐标是这一代的最大Q值
    best_X_in_history.append(pop[Fit.index(max(Fit))])  # 最大Q值的那一个个体
    exetime += 1
"""
max Q
best membership
NMI
"""
print('The max Q is=', best_Q_in_history[len(best_Q_in_history) - 1])
print('The best membership is=', best_X_in_history[len(best_X_in_history) - 1])
print('The NMI is=', ig.compare_communities(real_membership, best_X_in_history[len(best_X_in_history) - 1], method='nmi', remove_none=False))
   
"""
绘图
"""
exetimes = [ i for i in range(1,NP+1) ]
plt.plot(exetimes,best_Q_in_history,'b-',marker='o')
plt.title("football DECD plot")
plt.ylabel("Q_value")
plt.xlabel("exetime")   
