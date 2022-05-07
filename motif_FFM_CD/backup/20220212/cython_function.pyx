# -*- coding: utf-8 -*-
"""
Created on Tue Jan 25 18:56:18 2022

@author: WYW
"""
# Cython —— Q函数
cimport numpy as np 
cimport cython 


# =============================================================================
#     fit_Qg: 计算单个个体的模糊重叠社区划分的模块度函数Qg值
#     X: 种群中的个体
#     adj: 加权邻接矩阵
#     n: 网络节点数目
#     c: 社区划分的数目
#     W: 加权网络邻接矩阵的总权值之和
#     m: 邻接矩阵各行的权值之和
# =============================================================================
@cython.boundscheck(False) 
@cython.wraparound(False) 
@cython.nonecheck(False)
cpdef double fit_Qg(double[:,:] X, int[:,:] adj, int n, int c, int W, int[:,:] m):
    cdef double Qg = 0.0
    for k in range(c):
        for i in range(n):
            for j in range(n):
                Qg = Qg + (adj[i,j] - (m[0,i]*m[0,j])*1.0/W)*X[k,i]*X[k,j]
    Qg = Qg/W
    return Qg