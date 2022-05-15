# -*- coding: utf-8 -*-
"""
Created on Fri Apr 22 11:31:58 2022

@author: l
"""

# -*- coding: utf-8 -*-

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
beican_9_network = path + r'/9_beican.txt'
benson_network = path + r"/benson.txt"
zhang_network = path + r'/zhang.txt'
karate_network = path + r'/karate.txt'
dolphins_network = path + r'/dolphins.txt'
football_network = path + r'/football.txt'
polbooks_network = path + r'/polbooks.txt'

# 选择网络
network = benson_network
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
c = 2  #社区的真实划分数
M = 1  #模体选择【1:M1,2:M2,3:M3,4:M4,51:M5,6:M6,7:M7,8:M8】 
Q_flags = [0,1,2,3,4]  # Q_flag: 0:Q 1:Qg 2:Qc_FCD 3:Qc_OCD 4:Qov 
#Q_flag=Q_flags[2] # Qc
Gen = 1 # 调整次数
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

X = np.empty((c,n), dtype = float) 
cs_nodes_num = []
with open(r'E:\weiyawei\workspace\motif_FFM_CD\result\bestX.txt', mode='r',encoding='UTF-8') as f:
    cs_nodes_num = f.readlines()
    for index,nodes_num in enumerate(cs_nodes_num):
#        nums = nodes_num[1:-2].split(', ')
        nums = nodes_num.split(',')
        nums = list(map(eval,nums))
#        nums = list(map(round,nums,[4 for i in range(len(nums))]))
        X[index,:] = np.asarray(nums)

# =============================================================================
# 融合隶属度和高阶权重信息
# =============================================================================
motif_m_adj = copy.deepcopy(motif_adj)
for edge in edge_all:
#    print(edge)
    clist = np.argmax(X,axis = 0)
    cs = [clist[edge[0]],clist[edge[1]],0]
    # 寻找基于该边的模体的顶点
    node_ij = set(np.nonzero(adj[edge[0],:])[1]) & set(np.nonzero(adj[edge[1],:])[1])
    w_edge = 0 #初始化该条边的权重
    for node in node_ij:
        # M(edge[0],dege[1],node)
        cs[2] = np.argmax(X[:,node])
        # 获得该模体当前所在的社区
        M_c = np.argmax(np.bincount(np.array(cs)))
        # 计算该模体的隶属度之和
        sum_membership = X[M_c][edge[0]] + X[M_c][edge[1]] + X[M_c][node]
        # 计算该模体的权重之和
        sum_w = motif_adj[edge[0],edge[1]] + motif_adj[edge[0],node] + motif_adj[node,edge[1]]
        # 计算各条边的权重
        w_edge += (motif_adj[edge[0],edge[1]]/sum_w)*sum_membership
    w_edge += motif_m_adj[edge[0],edge[1]]
    motif_m_adj[edge[0],edge[1]] = round(w_edge,3)
    motif_m_adj[edge[1],edge[0]] = round(w_edge,3)

# =============================================================================
# 使用融合后的权重信息进行节点隶属度的调整
# =============================================================================
#    nodes = [i for i in range(n)]
#    for i in nodes:
#        ### 计算当前该点node的隶属度
#        # 寻找节点 i 基于边的邻居节点 j_e_nodes
#        j_e_nodes = np.nonzero(adj[i,:])[1]
#        # 获得邻居节点 j 所在的社区       
#        j_nodes_c = np.argmax(X[:,j_e_nodes], axis=0)
#        cs = list(set(j_nodes_c)) # 节点i的邻居社区
#        node_c = dict(zip(j_e_nodes,j_nodes_c))
#        
#        ## 计算隶吸引力attr（隶属度）
#        for c1 in cs:
#            attr_ic,W,w = 0,0,0 # 吸引力，权重总和，i节点对c社区的权重总和
#            for node in j_e_nodes:
#                if node_c[node] == c1:
#                    w += motif_m_adj[i,node]
#                W += motif_m_adj[i,node]
#            attr_ic = w/W  # 吸引力计算公式
#            # 更改隶属度表
##            if attr_ic > 0 and i == 8:
##                
##                print("c={},node={},attr={},node_c={}".format(c1,i,attr_ic,node_c))
#            X[c1,i] = attr_ic
#    
# =============================================================================
# 计算隶属度值
# =============================================================================
#W = np.sum(adj) # 权值之和
#m = np.sum(adj, axis=0) # adj 各列之和
#for Q_flag in Q_flags:
#    fit = func.fit_Q(X,adj,n,c,W,m,Q_flag)
#    membership_c = np.argmax(X, axis=0)
#    #
#    ##membership = [0,0,0,0,1,1,1,1,1]
#    #fit=ig.GraphBase.modularity(Gi, membership_c)
#    print(membership_c)
#    print("Qflag={0},fit={1}\n".format(Q_flag,round(fit,4)))

# =============================================================================
# NMI计算
# =============================================================================
membership_c = np.argmax(X, axis=0)
print(membership_c)
real_mem = [0,0,0,0,0,1,1,1,1,1]
nmi=ig.compare_communities(real_mem, membership_c, method='nmi', remove_none=False)    
print("nmi=",nmi) 

# =============================================================================
# 融合权重 
# =============================================================================
#for edge in edge_all:
#    print("【{0},{1}】={2}".format(edge[0]+1,edge[1]+1,motif_m_adj[edge[0],edge[1]]))

## =============================================================================
## Test
## =============================================================================
#a=(1.275)/(1.275+3.52+1.47)
#print(round(a,4),round(1-a,4))













