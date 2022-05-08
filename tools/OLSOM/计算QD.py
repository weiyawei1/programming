# -*- coding: utf-8 -*-
"""
Created on Mon Feb  7 20:46:13 2022

@author: bozite
"""
import numpy as np
import igraph as ig
import pandas as pd
import networkx as nx
from numpy import random, mat,where

def directed_modularity(B, membership):
    # n = G.number_of_nodes()
    # A = np.array(nx.adjacency_matrix(G).todense())
    # k_out = A.sum(axis=0)  # axis=0按列相加
    # k_in = A.sum(axis=1)
    # t = []
    # for i in range(N):
    #     for j in range(N):
    #         if membership[i] == membership[j]:
    #             t.append(1)
    #         else:
    #             t.append(0)
    # print(t)
    t = [1 if membership[qq] == membership[qqq] else 0 for qq in range(N) for qqq in range(N) ]
    b = B.getA()  # 将矩阵类型转化为数组
    B = b.reshape(1, n ** 2)  # 将其变为1行1156列
    B = mat(B)  # 将数组转化为矩阵
    t = np.array(t)
    # print(t.shape)
    k = t.reshape(n ** 2, 1)
    # Q1 = B * k
    # Qd = ((1 / M) * Q1)
    # Qd = (Qd.tolist())[0][0]
    # print(type(Qd))
    return ((1 / M) * (B * k)).tolist()[0][0]

network_path = r"E:\weiyawei\dayToDayWork\对比算法\有向真实网络\test\5000_40_100_20_100"

G = nx.read_edgelist(network_path + "\\" + '5000_40_100_20_100.dat', create_using=nx.DiGraph(), nodetype=int, encoding='utf-8')#networkx网络

n = G.number_of_nodes()
M = nx.DiGraph.number_of_edges(G)
N = nx.DiGraph.number_of_nodes(G)

Gi = ig.Graph.Read_Edgelist(network_path + "\\" + '5000_40_100_20_100.dat', directed=True)#igraph网络
Gi = Gi.subgraph(map(int, G.nodes()))  # G.nodes()获取一个Graph对象中的node数组,
Gi = Gi.as_directed()

# print(Gi)
mini_n = min(G.nodes())
max_n = max(G.nodes())
B = nx.directed_modularity_matrix(G, nodelist=[int(i) for i in range(mini_n, max_n + 1)])#模块度矩阵

##############计算Qd
result_path = r'E:\weiyawei\dayToDayWork\对比算法\有向真实网络\test_result'
fname = '5000_40_100_20_100_result.txt'
mems = []
with open(result_path + "\\" + fname, mode='r',encoding='UTF-8') as f:
    mem = []
    datas = f.read().splitlines()
    for data in datas:
        mem = list(map(int, data.split(',')))
        mems.append(mem)

# mem = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]#mem是社区划分
mods = []
for gen,mem in enumerate(mems):
    mod = directed_modularity(B,mem)#有向模块度计算
    mods.append(mod)
# NMI
realcommunity_path = r'E:\weiyawei\dayToDayWork\对比算法\有向真实网络\test\5000_40_100_20_100'
real_mem= []
with open(realcommunity_path + "\\" + 'community.txt', mode='r',encoding='UTF-8') as f:
    datas = f.read().splitlines()
    for data in datas:
        real = list(map(int, data.split('\t')))[1]
        real_mem.append(real)

nmis = []
for gen,mem in enumerate(mems):
    nmi=ig.compare_communities(real_mem, mem, method='nmi', remove_none=False)
    nmis.append(nmi)

print("mods=",mods)
print("nmis=",nmis)




