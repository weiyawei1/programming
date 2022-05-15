# -*- coding: utf-8 -*-
"""
Created on Tue Mar 29 15:19:05 2022

@author: l
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 09:55:19 2022

@author: WYW
"""
"""
    motif_FFM_CD_main_v1
   使用各种优化算法，基于模体的加权网络的社区检测
"""

import numpy as np
import igraph as ig
import random  as rd
from numpy import random
import networkx as nx
import copy
import pandas as pd
import os
import time
from pandas import DataFrame

# 各模块函数
import motif_network_construct as net_stru_func
import algorithm_FCD_function as alg_func
import motif_FFM_CD_function as func

# =============================================================================
# 网络信息
# network
# =============================================================================
path = r"data/经典数据集"
M1_test_1 = path + r'/M1_test_1.txt'
M1_test_2 = path + r'/M1_test_2.txt'
M1_test_3 = path + r'/M1_test_3.txt'
beican_9_network = path + r'/9_beican.txt'
benson_network = path + r"/benson.txt"
zhang_network = path + r'/zhang.txt'
karate_network = path + r'/karate.txt'
dolphins_network = path + r'/dolphins.txt'
football_network = path + r'/football.txt'
polbooks_network = path + r'/polbooks.txt'

# 选择网络
network = zhang_network
G1 = nx.read_edgelist(network)
G1 = G1.to_undirected()

# 获取网络数据中的边列表，并根据其使用igraph创建网络
Gi=ig.Graph.Read_Edgelist(network)
Gi=Gi.subgraph(map(int,G1.nodes()))          
Gi=Gi.as_undirected()

edge_all = Gi.get_edgelist()

# =============================================================================
# 各参数设置
# =============================================================================
n=G1.number_of_nodes()
NP = 100
c = 3  #社区的真实划分数
M = 1  #模体选择【1:M1,2:M2,3:M3,4:M4,51:M5,6:M6,7:M7,8:M8】
Q_flags = [0,1,2,3,4]  # Q_flag: 0:Q 1:Qg 2:Qc_FCD 3:Qc_OCD 4:Qov 
Q_flag=Q_flags[2] # Qc
                                              
# =============================================================================
# 构建基于模体M1的加权网络
# =============================================================================
G = net_stru_func.construct_weighted_network(Gi,n,M)

# 获得无权网络邻接矩阵
G2 = nx.Graph() 
G2.add_nodes_from([i for i in range(n)])
G2.add_edges_from(edge_all)
adj= nx.adjacency_matrix(G2)
adj=adj.todense() 

# 构建基于模体的加权网络邻接矩阵motif_adj
motif_adj = nx.adjacency_matrix(G)
motif_adj = motif_adj.todense() 
# 获得模体邻接矩阵
motif_adj = net_stru_func.get_motifadd_adj(G, edge_all, M)

## 从外部导入隶属度值2
#test_X = np.empty((c,n), dtype = float) 
#cs_nodes_num = []
#with open(r'E:\weiyawei\workspace\motif_FFM_CD\testQ\testX2.txt', mode='r',encoding='UTF-8') as f:
#    cs_nodes_num = f.readlines()
#    for index,nodes_num in enumerate(cs_nodes_num):
##        nums = nodes_num[1:-2].split(', ')
#        nums = nodes_num.split(',')
#        nums = list(map(eval,nums))
##        nums = list(map(round,nums,[4 for i in range(len(nums))]))
#        test_X[index,:] = np.asarray(nums)
 # =============================================================================
# 种群初始化，有偏操作
# =============================================================================
#种群初始化
pop = func.init_pop(n, c, NP)  #初始化种群
fit_values = func.fit_Qs(pop,adj,n,c,NP,Q_flag)   #适应度函数值计算 
membership_c = np.argmax(pop[:,:,fit_values.index(max(fit_values))], axis=0)
print(membership_c)



#有偏操作
bias_pop = func.bias_init_pop(pop, c, n, NP, adj) # 对初始化后的种群进行有偏操作
bias_fit_values = func.fit_Qs(bias_pop,adj,n,c,NP,Q_flag)   #适应度函数值计算 
#选择优秀个体并保留到种群
for index in range(NP):
    if bias_fit_values[index] > fit_values[index]:
        pop[:,:,index] = bias_pop[:,:,index]    #保存优秀个体
        fit_values[index] = bias_fit_values[index] #保存优秀个体的适应度函数值
membership_c = np.argmax(pop[:,:,fit_values.index(max(fit_values))], axis=0)
print(membership_c)
       
#print("pop=",pop[:,:,fit_values.index(max(fit_values))],end='\n')
#(new_pop, new_fit) = alg_func.SOSFCD(pop, fit_values, n, c, NP, adj,Q_flag)
#print("new_pop=",new_pop[:,:,new_fit.index(max(new_fit))])
#membership_c = np.argmax(new_pop[:,:,new_fit.index(max(new_fit))], axis=0)
#print(membership_c)

#X = pop[:,:,fit_values.index(max(fit_values))]
#W = np.sum(adj) # 权值之和
#m = np.sum(adj, axis=0) # adj 各列之和
#fit = func.fit_Q(X,adj,n,c,W,m,Q_flag)
#print("fit={}".format(fit))

#fits = []
#for Q_flag in Q_flags:
#    fit = func.fit_Q(test_X,adj,n,c,W,m,Q_flag)
#    membership_c = np.argmax(test_X, axis=0)
##    print(membership_c)
#    
#    #membership = [0,0,0,0,1,1,1,1,1]
#    #fit=ig.GraphBase.modularity(Gi, membership)
#    print("fit_"+str(Q_flag)+"=",fit)
#    fits.append(round(fit,4))
















