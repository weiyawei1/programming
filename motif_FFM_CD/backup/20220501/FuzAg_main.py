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

# 各模块函数
import motif_network_construct as net_stru_func
import algorithm_FCD_function as alg_func
import motif_FFM_CD_function as func


# =============================================================================
# 网络信息
# network
# =============================================================================
path = r"F:\研究生工作文件夹\data\经典数据集"
beican_9_network = path + r'\9_beican.txt'
karate_network = path + r'\karate.txt'
dolphins_network = path + r'\dolphins.txt'
football_network = path + r'\football.txt'
polbooks_network = path + r'\polbooks.txt'

# 选择网络
network = karate_network
G1 = nx.read_edgelist(network)
G1 = G1.to_undirected()

print("-------newwork is karate_network--------")
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
c = 2    #社区的真实划分数
Gen = n  #进化代数
threshold_value = 0.25  #阈值
M = 1  #模体选择【1:M1,2:M2,3:M3,4:M4,51:M5,6:M6,7:M7,8:M8】 

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
# motif_adj = net_stru_func.get_motifadd_adj(G, edge_all, M)

# # 设置全局变量
pop_best_history = np.zeros((c,n,Gen)) # 用于保存历史最优的个体记录
best_in_history_Qg = [] # 用于保存历史最优Qg值

# 初始化NMi
nmilist=np.zeros((1,Gen),dtype = float, order = 'C')

# =============================================================================
# ##################################Test#######################################
# FuzAg: 基于自隶属度搜索概念的模糊聚类社区检测
# n: 网络节点数
# itermax: 最大迭代次数
# phi: 获得任何社区成员资格的节点阈值
# motif_adj: 加权网络邻接矩阵
# return: U, K 隶属度矩阵，社区划分数
# =============================================================================
RUbest = {}
Qs = []
RUlist = []
for gen in range(0,1):
    print("gen=",gen)
    RU, RK = alg_func.FuzAg(n, 6, threshold_value, adj,16)
    
    # 设置社区 (离散划分)
    membership = [0]*n
    keyList = RU.keys()
    
    max_i_cs = {}
    for i in range(n):
        i_ships = []
        for key in keyList:
            i_ships.append(RU[key][i])
        maxIndexList = []
        i_cs = []
        if i_ships.count(max(i_ships)) > 1:
            for index,i_ship in enumerate(i_ships):
                if i_ship == max(i_ships):
                    maxIndexList.append(index)
                    i_cs.append(index)
            c = rd.choice(maxIndexList)
            membership[i] = c
            max_i_cs[i] = i_cs
        else:
            c = i_ships.index(max(i_ships))
            membership[i] = c
            
    Q = 0.0

    membership_V = copy.deepcopy(membership)
    for key in max_i_cs.keys():
        for c in max_i_cs[key]:
            membership_V[key] = c
            Qnew = ig.GraphBase.modularity(Gi, membership_V) 
            if Qnew > Q:
                Q = Qnew
                membership[i] = c
    print("membership=",membership_V)
    Qs.append(Q)
    RUlist.append(RU)
print ('Karate_Qs  is',Qs) 
print ('Karate_Qbest  is',max(Qs))

# NMI=ig.compare_communities(membership1, membership2, method='nmi', remove_none=False) 



