# -*- coding: utf-8 -*-
"""
Created on Fri Jan 21 14:04:38 2022

@author: WYW
"""
"""
    各优化算法函数
"""
import numpy as np
import pandas as pd 
import random  as rd
import copy
import math
import operator


from random import shuffle

import motif_FFM_CD_function as func
import FuzAg_function as FuzAg_fuc

# =============================================================================
# SOSFCD: 共生生物搜索算法进行社区检测
# pop: 种群
# fit_values: 适应度函数值列表
# n: 网络节点数
# c: 划分社区数
# NP: 种群中个体数
# adj: (加权)网络邻接矩阵
# return: para_pop, para_fit 返回当前进化后的种群和适应的函数值
# =============================================================================
def SOSFCD(pop, fit_values, n, c, NP, adj):
    # Mutualism【互利共生】
    W = np.sum(adj) # 权值之和
    m = np.sum(adj, axis=0) # adj 各列之和
    mutu_pop = copy.deepcopy(pop)
    mutu_fit = copy.deepcopy(fit_values)
    # better_number = 0
    for i in range(NP):
        # 找到当代种群中的最优个体
        best_fit = max(mutu_fit)
        best_fit_index = mutu_fit.index(best_fit) 
        # Xi != Xj
        ij_list = [i for i in range(NP)]
        ij_list.remove(i)
        j = rd.choice(ij_list)
        # 互利共生算法
        Xbest = mutu_pop[:,:,best_fit_index]
        Xi = mutu_pop[:,:,i]
        Xj = mutu_pop[:,:,j]
        mutual_vector = 0.5 * (Xi + Xj) # 互利共生向量
        BF1=round(1+rd.random())
        BF2=round(1+rd.random())
        # 生成Xinew和Xjnew
        Xinew = Xi + rd.random()*(Xbest - BF1*mutual_vector)
        Xjnew = Xj + rd.random()*(Xbest - BF2*mutual_vector)
        # 边界约束检查与修正
        Xinew = func.bound_check_revise(Xinew,n,c)
        Xjnew = func.bound_check_revise(Xjnew,n,c)
        # 适应度函数值计算
        Xinew_fit = func.fit_Q(Xinew,adj,n,c,W,m)
        Xjnew_fit = func.fit_Q(Xjnew,adj,n,c,W,m)
        # 选择优秀个体并保留到种群
        if Xinew_fit > mutu_fit[i]:
            mutu_pop[:,:,i] = Xinew    # 保存优秀个体
            mutu_fit[i] = Xinew_fit # 保存优秀个体的适应度函数值
            # better_number+=1
        if Xjnew_fit > mutu_fit[j]:
            mutu_pop[:,:,j] = Xjnew    # 保存优秀个体
            mutu_fit[j] = Xjnew_fit # 保存优秀个体的适应度函数值
            # better_number+=1
    # print("mutu_better_number={}".format(better_number))
    # print("mutu_best_Q={}".format(max(mutu_fit)))
    
    # Commensalism【共栖】
    comm_pop = mutu_pop
    comm_fit = mutu_fit
    # better_number = 0
    for i in range(NP):
        # 找到当代种群中的最优个体
        best_fit = max(comm_fit)
        best_fit_index = comm_fit.index(best_fit) 
        # Xi != Xj
        ij_list = [i for i in range(NP)]
        ij_list.remove(i)
        j = rd.choice(ij_list)
        # 共栖算法
        Xbest = comm_pop[:,:,best_fit_index]
        Xi = comm_pop[:,:,i]
        Xj = comm_pop[:,:,j]
        Xinew = Xi + rd.uniform(-1, 1)*(Xbest - Xj)
        # 边界约束检查与修正
        Xinew = func.bound_check_revise(Xinew,n,c)
        # 适应度函数值计算
        Xinew_fit = func.fit_Q(Xinew,adj,n,c,W,m)
        # 选择优秀个体并保留到种群
        if Xinew_fit > comm_fit[i]:
            comm_pop[:,:,i] = Xinew    # 保存优秀个体
            comm_fit[i] = Xinew_fit # 保存优秀个体的适应度函数值
            # better_number+=1
    # print("comm_better_number={}".format(better_number))
    # print("comm_best_Q={}".format(max(comm_fit)))
   
    # Parasitism【寄生】
    para_pop = comm_pop
    para_fit = comm_fit
    # better_number = 0
    for i in range(NP):
        # 找到当代种群中的最优个体
        best_fit = max(para_fit)
        best_fit_index = para_fit.index(best_fit) 
        # Xi != Xj
        ij_list = [i for i in range(NP)]
        ij_list.remove(i)
        j = rd.choice(ij_list)
        # 寄生算法
        para_vector = copy.deepcopy(para_pop[:,:,i])   # 寄生向量
        seeds = [i for i in range(n)]
        rd.shuffle(seeds)
        pick = seeds[:rd.randint(1, n)] # 随机选择一些节点
        # 在约束范围内随机化节点对应的隶属度值
        para_vector[:,pick] = func.init_pop(len(pick),c,1)[:,:,0] 
        # 边界约束检查与修正
        para_vector = func.bound_check_revise(para_vector,n,c)
        # 适应度函数值计算
        para_vector_fit = func.fit_Q(para_vector,adj,n,c,W,m)
        # 选择优秀个体并保留到种群
        if para_vector_fit > para_fit[i]:
            para_pop[:,:,i] = para_vector    # 保存优秀个体
            para_fit[i] = para_vector_fit # 保存优秀个体的适应度函数值
            # better_number+=1
    # print("para_better_number={}".format(better_number))
    # print("para_best_Q={}".format(max(para_fit)))
    # 返回当前进化后的种群和适应的函数值
    return para_pop, para_fit

# =============================================================================
# WOAFCD: 鲸鱼搜索算法进行社区检测
# pop: 种群
# fit_values: 适应度函数值列表
# n: 网络节点数
# c: 划分社区数
# NP: 种群中个体数
# adj: 网络邻接矩阵
# return: para_pop, para_fit 返回当前进化后的种群和适应的函数值
# =============================================================================
def WOAFCD(pop, fit_values, n, c, NP, adj):
    pass    

# =============================================================================
# CRO: 基于双表示化学反应优化的社区检测
# pop: 种群
# fit_values: 适应度函数值列表
# n: 网络节点数
# c: 划分社区数
# NP: 种群中个体数
# adj: 网络邻接矩阵
# return: para_pop, para_fit 返回当前进化后的种群和适应的函数值
# =============================================================================
def CROFCD(pop, fit_values, n, c, NP, adj):
    pass  

# =============================================================================
# FuzAg: 基于自隶属度搜索概念的模糊聚类社区检测
# n: 网络节点数
# itermax: 最大迭代次数
# phi: 获得任何社区成员资格的节点阈值
# motif_adj: 加权网络邻接矩阵
# return: U, K 隶属度矩阵，社区划分数
# =============================================================================
def FuzAg(n,itermax,phi,motif_adj,node):
    ## 参数设定
    rflag=1
    aflag=1
    iterVal=0
    phi =phi  # 设定获得任何社区成员资格的节点阈值
    itermax = itermax  # 最大迭代代数
    K = 1   # 当前社区数，初始值为 1
    
    N = n # 节点数
    W = motif_adj  # 加权网络的邻接矩阵
    U={}    # 各分区的邻接矩阵
    # 初始化各分区邻接矩阵的隶属度值为 0
    U[N+1]=np.zeros(N)
    # 初始化锚点列表
    anchorList=set({})
    # 初始化参数
    FuzAg_fuc.init(phi,N,W,U,anchorList)
    # 随机定义第一个锚点
    # FuzAg_fuc.randomFirstAnchor(N)
    FuzAg_fuc.addAnchor(node)
    listOfOldAnchors = []
    while(1):
    	print("FuzAg_iter:",iterVal)
    	iterVal+=1
    	trflag=0
    	taflag=0
    	for i in range(N):
    		for j in anchorList:
    			U[j][i]=FuzAg_fuc.calculateMembership(i,j)
                
    	for i in range(N):
    		U[N+1][i]=FuzAg_fuc.selfMembership(i)
    	FuzAg_fuc.normalize()    #规范化隶属度矩阵
        # 初始化锚点移除集合
    	anchorsToBeRemoved=[]
    	for i in anchorList:
    		if FuzAg_fuc.identifyFalseAnchor(i) or FuzAg_fuc.identifyRedundantAnchor(i):
    			# removeAnchor(i)
    			anchorsToBeRemoved.append(i)
    			trflag=1
    	for i in anchorsToBeRemoved:
    		# print("Removing Anchor-> ",i)
    		FuzAg_fuc.removeAnchor(i)
    	# print(U)
    	if(trflag==1):
    		# for i in anchorList:
    		# 	tmpList=[]
    		# 	for j in range(N):
    		# 		tmpList.append(FuzAg_fuc.calculateMembership(j,i))
    		# 	U[i]=np.array(tmpList)
    		# # for i in range(N):
    		# # 	tmpList=[]
    		# # 	for j in anchorList:
    		# # 		tmpList.append(calculateMembership(i,j))
    		# # 	U[j]=np.array(tmpList)
    		# for i in range(N):
    		# 	U[N+1][i]=FuzAg_fuc.selfMembership(i)
    		for i in range(N):
    			for j in anchorList:
    				U[j][i]=FuzAg_fuc.calculateMembership(i,j)
    		FuzAg_fuc.normalize()
    		# print(U)
    		rflag=1
    	else:
            rflag=0
            
    	listOfNewAnchors=FuzAg_fuc.newAnchors()
    	if len(listOfNewAnchors) == len(listOfOldAnchors):
            print("新增锚点均不可用->",anchorList)
            # break
    	listOfOldAnchors = listOfNewAnchors
    	print("Adding Anchors->",listOfNewAnchors)
    	if(len(listOfNewAnchors)>0):
            aflag=1
    	else:
            aflag=0
    	for i in listOfNewAnchors:
            FuzAg_fuc.addAnchor(i)
    	if(iterVal>=itermax):
    		break
    	if(aflag==0 and rflag==0):
    		break
    del U[N+1]
    K = len(anchorList)
    return U,K

    

    