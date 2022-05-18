# -*- coding: utf-8 -*-
"""
Created on Tue Jan 25 18:56:18 2022

@author: WYW
"""
# Cython —— Q函数
import numpy as np
cimport numpy as np 
cimport cython
from libc.math cimport exp,sqrt
from libc.string cimport memcpy, memset
from libc.stdlib cimport malloc, free

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
@cython.cdivision(True)
cpdef double fit_Qg(double[:,:] X, long[:,:] adj, long n, long c, long W, long[:,:] m):
    cdef double Qg = 0.0
    for k in range(c):
        for i in range(n):
            for j in range(n):
                Qg = Qg + (adj[i,j] - (m[0,i]*m[0,j])*1.0/W)*X[k,i]*X[k,j]
    Qg = Qg*1.0/W
    return Qg

# =============================================================================
#     fit_Q: 计算单个个体的模糊重叠社区划分的模块度函数Q值
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
@cython.cdivision(True)
cpdef double fit_Q(double[:,:] X, long[:,:] adj, long n, long c, long W, long[:,:] m,long [:] mod):
    cdef double Q = 0.0
    for i in range(n):
        for j in range(n):
            if mod[i] == mod[j]:
                Q = Q + (adj[i,j] - (m[0,i]*m[0,j])*1.0/W)
    return Q*1.0/W

# =============================================================================
#     fit_Qc: 计算单个个体的模糊重叠社区划分的模块度函数Qc值
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
@cython.cdivision(True)
cpdef double fit_Qc(double[:,:] X, long[:,:] adj, long n, long c, long W, long[:,:] m):
    cdef double u_ki,u_kj,minu_kij,temp,sij
    cdef double Qc = 0.0
    for i in range(n):
        for j in range(n):
            # 计算sij
            sij= 0.0000
            temp = 0.000
            for k in range(c):
                u_ki = X[k,i]
                u_kj = X[k,j]
                if u_ki<u_kj:
                    temp = sqrt(u_ki)
                else:
                    temp = sqrt(u_kj)
                if temp > sij:
                    sij = temp            
            # 根据sij求Qc
            Qc = Qc + (adj[i,j] - (m[0,i]*m[0,j])*1.0/W)*1.0/W*sij
    return Qc


# =============================================================================
#     fit_Qov: 计算单个个体的模糊重叠社区划分的模块度函数Qg值
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
@cython.cdivision(True)
cpdef double fit_Qov(double[:,:] X, long[:,:] adj, long n, long c, long W, long[:,:] m):
    cdef double Qov = 0.0, wSum, lSum
    cdef double** r
    cdef double** w
    cdef int* pointk
    cdef int index, nk
    
    # 创建二维矩阵
    r = <double**>malloc(n * sizeof(double*))
    for i in range(n):
        r[i] = <double*>malloc(n * sizeof(double))
#        memset(r[i], 0, n * sizeof(double))
        
    w = <double**>malloc(n * sizeof(double*))
    for i in range(n):
        w[i] = <double*>malloc(n * sizeof(double))
#        memset(w[i], 0, n * sizeof(double))
        
    for k in range(c):
        #获得第K个社区的所有节点标号(非零元素的下标)
        nk = 0
        for i in range(n):
            if X[k,i] > 0.0:
                nk = nk + 1       
        pointk = <int*>malloc(nk * sizeof(int))
        index = 0
        for i in range(n):
            if X[k,i] > 0.0:
                pointk[index] = i
                index = index + 1 

        # 对矩阵赋值
        for i in range(nk):
            for j in range(nk):
                r[i][j] = 1.0/((1+exp(-(60*X[k,pointk[i]]-30)))*(1+exp(-(60*X[k,pointk[j]]-30))))  
        for i in range(nk):
            for j in range(nk):
                # 求和
                wSum=0
                for t in range(nk):
                    wSum = wSum + r[i][t]
                lSum=0
                for t in range(nk):
                    lSum = lSum + r[t][j]
                # 计算w
                w[i][j] = wSum*lSum*1.0/(nk*nk)
        # 计算Qov值
        for i in range(nk):
            for j in range(nk):
                Qov = Qov + (r[i][j]*adj[pointk[i],pointk[j]] - w[i][j]*(m[0,pointk[i]]*m[0,pointk[j]])*1.0/W)
                
        free(pointk)
    # 释放内存
    for i in range(n):
        free(r[i])
        free(w[i])
    free(r)
    free(w)
    Qov = Qov*1.0/W
    return Qov      

# =============================================================================
#     getMEM_adj: 获得Xi的模体，边及隶属度融合的权重矩阵
#     X: 种群中的个体
#     adj: 加权邻接矩阵
#     n: 网络节点数目
#     c: 社区划分的数目
#     W: 加权网络邻接矩阵的总权值之和
#     M: 模体阶数，默认为3
# =============================================================================
@cython.boundscheck(False) 
@cython.wraparound(False) 
@cython.nonecheck(False)
@cython.cdivision(True)
cpdef double getEdgeW(double[:,:] X, long[:,:] me_adj, long[:,:] node_set, long[:,:,:] edge_set,long i, long j, long c, long lenSet, long M_n, long M_e):
    cdef double m_W = 0.0, tmp_m, sum_membership, sum_w
    cdef int M_c, c_max
    # 创建一个一维数组, 并以0值初始化
    i_M_c = <int*>malloc(c * sizeof(int*))
    for i_c in range(c):
        i_M_c[i_c]=0
    for m_index in range(lenSet): #第m个模体
        # 模体 M1
        # 获得该模体M当前所在的社区
        M_c = 0 #初始化M_c值
        for i_index in range(M_n): #模体中的第i个节点
            membership = 0.0 # 初始化membership
            c_max=0
            for c_index in range(c):#节点i所在的第c个社区
                tmp_m = X[c_index,node_set[i_index,m_index]]
                if tmp_m > membership:
                    membership = tmp_m
                    c_max=c_index         
            i_M_c[c_max] += 1
        for c_index in range(c): #获得模体M所在社区
            if i_M_c[c_index] > M_c:
                M_c = c_index
            i_M_c[c_index] = 0
        # 计算该模体的隶属度之和
        sum_membership=0.0
        for i_index in range(M_n):
            sum_membership += X[M_c,node_set[i_index,m_index]]
        # 计算该模体的权重之和
        sum_w = 0.0 #初始化sum_w值
        for e_index in range(M_e):
            sum_w += me_adj[edge_set[e_index,0,m_index],edge_set[e_index,1,m_index]]
        # 计算该边的融合权重
        m_W += (me_adj[i,j]/sum_w*sum_membership)
    
    return me_adj[i,j] + m_W    
        
        
        
        
        
        

