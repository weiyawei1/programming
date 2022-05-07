# -*- coding: utf-8 -*-
"""
Created on Tue Apr 19 23:30:21 2022

@author: WYW
"""
import numpy as np
import igraph as ig
import networkx as nx
import os
import time
import matplotlib.pyplot as plt



# ER = nx.random_graphs.erdos_renyi_graph(1000,0.05)             #生成包含20个节点、以概率0.2连接的ER随机网络
# pos = nx.spring_layout(ER) 
# largest_c = max(nx.connected_components(ER), key=len)


# =============================================================================
# 网络信息
# network
# =============================================================================
## 真实网络
path = r"data/经典数据集"
beican_9_network = path + r'/9_beican.txt'
zhang_network = path + r'/zhang.txt'
karate_network = path + r'/karate.txt'
dolphins_network = path + r'/dolphins.txt'
football_network = path + r'/football.txt'
polbooks_network = path + r'/polbooks.txt'
## 功能网络
func_path = r"data/功能网络"
brain47_network = func_path + r'/brain47.txt'

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
c = 12  #社区的真实划分数
Gen = 1000  #进化代数
threshold_value = 0.25  #阈值
M = 1  #模体选择【1:M1,2:M2,3:M3,4:M4,5:M5,6:M6,7:M7,8:M8】 
Independent_Runs = 40 # 本次实验独立运行次数
 
# =============================================================================
# 构建基于模体M1的加权网络
# =============================================================================
# G = net_stru_func.construct_weighted_network(Gi,n,M)

# 获得无权网络邻接矩阵
G2 = nx.Graph() 
G2.add_nodes_from([i for i in range(n)])
G2.add_edges_from(edge_all)
adj= nx.adjacency_matrix(G2)
adj=adj.todense() 

degree =  nx.degree_histogram(G2) 
print(degree)
degree_index = range(len(degree)) 
degree_distribution = [degree_i / float(sum(degree)) for degree_i in degree]     #将频次转换为概率

plt.loglog(degree_index,degree_distribution,'b-',marker='o')
plt.title("Degree distribution plot")
plt.ylabel("density")
plt.xlabel("degree")
