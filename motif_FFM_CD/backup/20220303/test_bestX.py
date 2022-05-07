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
network = path + r'/zhang.txt'
# 选择网络
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
Gen = 100  #进化代数
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

test_X = np.empty((c,n), dtype = float) 
cs_nodes_num = []
with open(r'E:\weiyawei\workspace\motif_FFM_CD\result\bestX_v1.txt', mode='r',encoding='UTF-8') as f:
    cs_nodes_num = f.readlines()
    for index,nodes_num in enumerate(cs_nodes_num):
        nums = nodes_num[1:-2].split(', ')
        test_X[index,:] = np.asarray(nums)

W = np.sum(adj) # 权值之和
m = np.sum(adj, axis=0) # adj 各列之和
fit = func.fit_Q(test_X,adj,n,c,W,m)
print("fit=",fit)






















