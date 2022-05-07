# -*- coding: utf-8 -*-
"""
Created on Thu Jan 20 13:n3:51 2022

@author: WYW
"""
"""
    TEST: 测试
"""
import numpy as np
import pandas as pd 
import random  as rd
import copy
from time import clock
import fit_Qg

# =============================================================================
#     fit_Qg: 计算单个个体的模糊重叠社区划分的模块度函数Qg值
#     X: 种群中的个体
#     c: 社区划分的数目
#     n: 网络节点数目
#     NP： 种群个体数目
# =============================================================================
# def fit_Qg(X,adj,n,c,W,m):
#     #计算单个个体的适应度函数值Qg
#     Qg = 0.0
#     for k in range(c):
#         for i in range(n):
#             for j in range(n):
#                 Qg = Qg + (adj[i,j] - (m[i]*m[j])/W)*X[k,i]*X[k,j]
#     Qg/=W
#     return Qg


n = 100
X = np.zeros((n,n))
adj = np.zeros((n,n),dtype = np.int)
for i in range(n):
    for j in range(n):
        if i > j :
            X[i][j] = i*0.1+j*0.01
            adj[i][j] = i+j
            
for i in range(n):
    for j in range(n):
            if i < j :
              X[i][j] = X[j][i]
              adj[i][j] = adj[j][i]
c = 10
W = np.sum(adj) # 权值之和
m = np.sum(adj, axis=0) # adj 各列之和

start = clock()
# fit = fit_Qg(X, adj, n, c, W ,m)
fit = fit_Qg.fit_Qg(X, adj, n, c, W ,m)
end = clock()
print("spend_time=",end-start)