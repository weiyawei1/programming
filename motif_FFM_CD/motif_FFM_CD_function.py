# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 11:55:16 2022

@author: WYW
"""
"""
    function: 各类功能函数
"""

import numpy as np
import pandas as pd 
import random  as rd
import networkx as nx
import copy
import math
import time
from matplotlib import pyplot as plt

# 引入外部函数
import find_motifs as fm

# C函数
import cython_function as cfunc

# =============================================================================
#     fit_Qs: 计算种群中每个个体的模糊重叠社区划分的模块度函数Q值
#     Qs: 根据pop计算的模块度值
#     pop: 种群
#     adj: 网络邻接矩阵
#     n: 网络节点数目
#     c: 社区划分的数目
#     NP： 种群个体数目
#    flag: 0:Q 1:Qg 2:Qc_FCD 3:Qc_OCD 4:Qov
# =============================================================================
def fit_Qs(Qs,pop,adj,n,c,NP,flag):
    W = np.sum(adj) # 权值之和
    m = np.sum(adj, axis=0) # adj 各列之和
    for N in range(NP):
        #计算每个个体的适应度函数值Q
        U = pop[:,:,N]
        # print(U)
        Q = fit_Q(U,adj,n,c,W,m,flag)
        Qs.append(Q)

# =============================================================================
#     fit_Q: 计算单个个体的模糊重叠社区划分的模块度函数Q值
#     X: 种群中的个体
#     adj: 加权邻接矩阵
#     n: 网络节点数目
#     c: 社区划分的数目
#     W: 加权网络邻接矩阵的总权值之和
#     m: 邻接矩阵各行的权值之和
#     flag: 0:Q 1:Qg 2:Qc_FCD 3:Qc_OCD 4:Qov
#     return 返回Q值
# =============================================================================
def fit_Q(X,adj,n,c,W,m,flag):
    Q=0
    if flag=="Q":
        ###Q###
        mod = np.argmax(X, axis=0).astype('int32')
        Q = cfunc.fit_Q(X,adj,n,c,W,m,mod)
    elif flag=="Qg":
        ###Qg###
        Q = cfunc.fit_Qg(X,adj,n,c,W,m)
    elif flag=="Qc_FCD":
        ###Qc_FCD###
        Q = cfunc.fit_Qc(X,adj,n,c,W,m)
    elif flag=="Qc_OCD":
        ###Qc_OCD###
        X_V1 = np.empty((c,n), dtype = float)
         # 离散化
        for k in range(c):
            for i in range(n):
                numki = X[k,i]
                if numki > 0.3:
                    X_V1[k,i] = 1
                else:
                    X_V1[k,i] = 0
        Q = cfunc.fit_Qc(X_V1,adj,n,c,W,m)        
    elif flag=="Qov":
        ##Qov###
        Q = cfunc.fit_Qov(X,adj,n,c,W,m)
    else:
        Q=-1
    return Q

# =============================================================================
#     init_pop: 种群初始化
#     n: 网络节点数目
#     c: 社区划分的数目
#     NP： 种群个体数目
#     return: 返回初始化后的种群
# =============================================================================
def init_pop(n,c,NP):
    pop = np.empty((c,n,NP), dtype = float) 
    for N in range(NP):
        for i in range(n):
            membershipList = []
            for k in range(c):
#                rd.seed(N+i+k)
                membershipList.append(rd.random())
            memberships = np.asarray(membershipList)
            memberships = memberships/sum(memberships)  #归一化
            pop[:,i,N]=memberships
    return pop

# =============================================================================
#     bias_init_pop: 种群有偏操作
#     pop：种群
#     n: 网络节点数目
#     c: 社区划分的数目
#     NP： 种群个体数目
#     return: 有偏操作后的种群
# =============================================================================
def bias_init_pop(pop,n,c,NP,adj):
    bias_pop = copy.deepcopy(pop)
    for N in range(NP):
        # 在该个体中，随机选择c个节点,且C个节点隶属于不同社区，将其隶属度赋值给所有相邻节点
        i_rand = rd.randint(0, n-1)  
        for i in range(n):
           if i != i_rand and adj[i_rand,i]>0:
               bias_pop[:,i,N] = bias_pop[:,i_rand,N]
    return bias_pop
       

# =============================================================================
#     bound_check_revise: 边界约束检查与修正
#     X: 个体隶属度矩阵
#     c: 社区划分的数目
#     n: 网络节点数目
#     NP： 种群个体数目
#     return: 约束检测并修正后的个体
# =============================================================================
def bound_check_revise(X,n,c):
    # 将每个元素约束到[0，1],并归一化
    for i in range(n):
        for k in range(c):
            Xki = X[k,i]
            if Xki > 1:
               Xki = 0.9999
            elif Xki < 0:
                X[k,i] = 0.0001
        X[:,i] = X[:,i] / sum(X[:,i])
    return X


# =============================================================================
#     NMM_funcs: 四种基于邻域社区的节点社区修正操作函数【1:"NOMM",2:"NMM",3:"MNMM",4:"NWMM"】
#     pop: 种群
#     n: 网络节点数目
#     c: 社区划分的数目
#     NP： 种群中的个体数
#     adj: 无权网络的邻接矩阵
#     motif_adj: 模体权重矩阵
#     threshold_value: 阈值
#     return: 返回nmm操作后的种群, 适应度值
# =============================================================================
def NMM_funcs(edge_dict, pop, n, c, NP, adj, motif_adj, threshold_value, Q_flag, nmm_flag):
    nmm_pop = copy.deepcopy(pop)
    nmm_fit = []
    me_adj = motif_adj + adj #总权重矩阵=模体权重+边权重
    if nmm_flag=="NOMM":
        ###NOMM###
        fit_Qs(nmm_fit,nmm_pop,adj,n,c,NP,Q_flag)   #适应度函数值计算 
    elif nmm_flag=="NMM":
        ###NMM###
        NMM(pop, n, c, NP, adj, threshold_value, Q_flag, nmm_pop, nmm_fit)
    elif nmm_flag=="MNMM":
        ###MNMM###
        MNMM(pop, n, c, NP, adj, motif_adj, Q_flag, nmm_pop, nmm_fit)
    elif nmm_flag=="NWMM":  
        ###NWMM###
        NWMM(edge_dict, pop, n, c, NP, adj, motif_adj,me_adj, Q_flag, nmm_pop, nmm_fit)
    # 返回种群和对应的模块度值
    return (nmm_pop, nmm_fit)

# =============================================================================
#     NMM: 基于邻居节点的社区修正（仅基于边邻居节点）
#     pop: 种群
#     n: 网络节点数目
#     c: 社区划分的数目
#     NP： 种群中的个体数
#     adj: 无权网络的邻接矩阵
#     threshold_value: 阈值
#     Q_flag: 模块度函数选择标识
#     nmm_pop: NMM种群
#     nmm_fit: NMM种群中个体对应的模块度函数值
# =============================================================================
def NMM(pop, n, c, NP, adj, threshold_value, Q_flag, nmm_pop, nmm_fit):
    for i in range(NP):
        seeds = [i for i in range(n)]
        rd.shuffle(seeds)
        pick = seeds[:rd.randint(1, n)] #随机选择一定数量的节点
        # pick = seeds #选取全部节点
        # 寻找不合理划分的节点和其对应的邻居节点
        unreasonableNodes = []
        NMM_CD_func(unreasonableNodes, pick, nmm_pop[:,:,i], adj, c, n, threshold_value)
        # 获得该节点应划分的社区号
        node_cno_list=[]
        NMM_P_func(node_cno_list, unreasonableNodes, nmm_pop[:,:,i],adj)
        # 修改该节点的隶属度值，对该节点重新划分社区
        unreasonableNodes_revise(node_cno_list,nmm_pop,i)
    # 计算该种群的适应度函数值
    fit_Qs(nmm_fit,nmm_pop,adj,n,c,NP,Q_flag)   #适应度函数值计算           

# =============================================================================
#     MNMM: 基于邻居节点的社区修正（基于边邻居节点和模体邻居节点,使用了模体权重信息）
#     pop: 种群
#     n: 网络节点数目
#     c: 社区划分的数目
#     NP： 种群中的个体数
#     adj: 无权网络的邻接矩阵
#     motif_adj: 模体加权网络的邻接矩阵
#     Q_flag: 模块度函数选择标识
#     nmm_pop: NMM种群
#     nmm_fit: NMM种群中个体对应的模块度函数值
# =============================================================================
def MNMM(pop, n, c, NP, adj, motif_adj, Q_flag, nmm_pop, nmm_fit):
    for i in range(NP):
        seeds = [i for i in range(n)]
        rd.shuffle(seeds)
        pick = seeds[:rd.randint(1, n)] #随机选取一定数量的节点
        # pick = seeds  #选取全部节点
        # 寻找不合理划分的节点和其对应的邻居节点
        unreasonableNodes = []
        MNMM_CD_func(unreasonableNodes,pick,nmm_pop[:,:,i],adj,motif_adj,c,n)
        # 获得该节点应划分的社区号
        node_cnos = []
        MNMM_P_func(node_cnos,unreasonableNodes,nmm_pop[:,:,i],adj,motif_adj)
        # print("len=",len(node_cnos))
        # 修改该节点的隶属度值，对该节点重新划分社区
        unreasonableNodes_revise(node_cnos,nmm_pop,i)
    # 计算该种群的适应度函数值    
    fit_Qs(nmm_fit,nmm_pop,adj,n,c,NP,Q_flag)   #适应度函数值计算 

# =============================================================================
#     NWMM: 融合邻域及权重信息的节点的社区修正（权重：融合了模体权重和隶属度信息）
#     edge_dict: 每条边所对应的模体M的点集和边集
#     pop: 种群
#     n: 网络节点数目
#     c: 社区划分的数目
#     NP： 种群中的个体数
#     adj: 无权网络的邻接矩阵
#     motif_adj: 模体加权网络的邻接矩阵
#     me_adj: #总权重矩阵=模体权重+边权重
#     Q_flag: 模块度函数选择标识
#     nmm_pop: NMM种群
#     nmm_fit: NMM种群中个体对应的模块度函数值
# =============================================================================
def NWMM(edge_dict, pop, n, c, NP, adj, motif_adj, me_adj, Q_flag, nmm_pop, nmm_fit):
    for p in range(NP):      
        # 随机节点选择
        seeds = [i for i in range(n)]
        rd.shuffle(seeds)
        pick = seeds[:rd.randint(1, n)] #随机选取一定数量的节点
        # pick = seeds #选取全部节点
        # 寻找不合理划分的节点和其对应的邻居节点
        unreasonableNodes = []
        NWMM_CD_func(edge_dict, unreasonableNodes,pick,nmm_pop[:,:,p],adj,motif_adj, me_adj, c, n)
        # 获得当前节点对各个社区的隶属程度
        node_cps = {} # 通过 attr(i,ck)调整
        node_cnos = [] # 通过U=U+0.5调整
        NWMM_P_func(edge_dict, node_cnos,node_cps,unreasonableNodes,nmm_pop[:,:,p],adj,motif_adj, me_adj, c)
        # 修改该节点的隶属度值，对该节点重新划分社区
        NWMM_nc_revise(node_cps,nmm_pop,p)
        unreasonableNodes_revise(node_cnos,nmm_pop,p)

    # 计算该种群的适应度函数值    
    fit_Qs(nmm_fit,nmm_pop,adj,n,c,NP,Q_flag)   #适应度函数值计算 

# =============================================================================
#     find_unreasonableNodes: 寻找基于边的不合理划分的节点
#     unreasonableNodes: 划分不合理的节点 []
#     pick: 一定数量的随机节点
#     Xi: 第i个个体的隶属度矩阵
#     adj: 加权网络的临界矩阵
#     c: 社区划分的数目
#     n: 网络节点数目
#     threshold_value: 阈值
# =============================================================================
def NMM_CD_func(unreasonableNodes,pick,Xi,adj,c,n,threshold_value):
    for i in pick:
        # 获得节点 i 所在的社区
        i_node_c = np.argmax(Xi[:,i])
        # 寻找节点 i 基于边的邻居节点 j_nodes
        j_nodes = np.nonzero(adj[i,:])[1]
        # 如果 i 节点无邻居节点，则跳过该节点
        if len(j_nodes) == 0:
            continue
        # 获得基于边的邻居节点 j 所在的社区       
        j_nodes_c = np.argmax(Xi[:,j_nodes], axis=0)
        # i_c != j_c
        cd_i = np.where(j_nodes_c != i_node_c)[0].shape[0] / len(j_nodes)
        # 如果节点 i 划分不合理程度大于阈值，则返回 i
        if cd_i > threshold_value :
            unreasonableNodes.append(i)
            
# =============================================================================
#     NMM_P_func: 寻找节点应划分的社区号
#     node_cno_list: 节点及对应的划分社区[(i,ck)]
#     nodes: 未正确划分的节点
#     Xi: 第i个个体的隶属度矩阵
#     adj: 加权网络的临界矩阵
# =============================================================================
def NMM_P_func(node_cno_list,nodes,Xi,adj):
    for i in nodes:
        # 获得 i 基于边的邻接节点
        j_nodes = np.nonzero(adj[i,:])[1]
        # 获得邻居节点 j 所在的社区       
        j_nodes_c = np.argmax(Xi[:,j_nodes], axis=0)
        # print("j_nodes_c=",j_nodes_c)
        node_cno_list.append((i,rd.choice(j_nodes_c)))  # choice() 依概率选择
        # i_c = np.argmax(np.bincount(j_nodes_c)) # 直接选择概率最大的社区作为i节点划分的社区
        # node_cno_list.append((i,i_c))
            
# =============================================================================
#     MNMM_CD_func: 寻找基于模体权重的不合理划分的节点
#     unreasonableNodes 划分不合理的节点
#     pick: 一定数量的随机节点
#     Xi: 第i个个体的隶属度矩阵
#     adj: 加权网络的临界矩阵
#     motif_adj: 模体邻接矩阵
#     c: 社区划分的数目
#     n: 网络节点数目
# =============================================================================
def MNMM_CD_func(unreasonableNodes,pick,Xi,adj,motif_adj,c,n):
    for i in pick:
        # 获得节点 i 所在的社区
        i_node_c = np.argmax(Xi[:,i])
        # 寻找节点 i 基于边的邻居节点 j_e_nodes
        j_e_nodes = np.nonzero(np.ravel(adj[i,:]))
        # 寻找节点 i 基于模体的邻居节点 j_m_nodes
        j_m_nodes = np.nonzero(np.ravel(motif_adj[i,:]))
        # 获得节点 i 的所有邻居节点(边邻居节点+模体邻居节点) j_nodes
        j_nodes = np.unique(np.ravel(np.concatenate((j_e_nodes,j_m_nodes),axis=1)))
        # 获得邻居节点 j 所在的社区       
        j_nodes_c = np.argmax(Xi[:,j_nodes], axis=0)
        # 如果 i 节点无邻居节点，且该节点无邻居社区，则跳过该节点
        Cnei = len(np.unique(np.concatenate((np.ravel(i_node_c),np.ravel(j_nodes_c)))))
        if len(j_nodes) == 0 or Cnei == 1:
            continue
        # 计算i与其所有邻居节点的权值总和
        wij_sum = 0
        for j in j_nodes:
            wij_sum += motif_adj[i,j]
        # i_c == j_c ,获取与i节点同一社区的邻居节点集,并求与i节点同一社区的邻居节点的权重总和
        wij = 0
        j_node_c_dict = dict(zip(j_nodes,j_nodes_c))
        ijc_nodes = [key for key in j_node_c_dict.keys() if j_node_c_dict[key] == i_node_c]
        for j in ijc_nodes:
            wij += motif_adj[i,j]
        # 如果节点 i 与同一社区的邻居节点之间的权重总和 <= 其与所有邻居节点权重总和的平均值，则返回 i
        if wij - wij_sum/Cnei <= 0:
            unreasonableNodes.append(i)

# =============================================================================
#     MNMM_P_func: 寻找节点应划分的社区号
#     node_cnos: 节点及对应的社区划分号
#     nodes: 未正确划分的节点
#     Xi: 第i个个体的隶属度矩阵
#     adj: 加权网络的临界矩阵
#     motif_adj: 模体邻接矩阵
# =============================================================================
def MNMM_P_func(node_cnos,nodes,Xi,adj,motif_adj):
    for i in nodes:
        # 初始化 i 节点对社区 c 的概率
        c_ps = []
        # 获得节点 i 所在的社区
#        i_node_c = np.argmax(Xi[:,i])
        # 寻找节点 i 基于边的邻居节点 j_e_nodes
        j_e_nodes = np.nonzero(np.ravel(adj[i,:]))
        # 寻找节点 i 基于模体的邻居节点 j_m_nodes
        j_m_nodes = np.nonzero(np.ravel(motif_adj[i,:]))
        # 获得节点 i 的所有邻居节点(边邻居节点+模体邻居节点) j_nodes
        j_nodes = np.unique(np.ravel(np.concatenate((j_e_nodes,j_m_nodes),axis=1)))
        # 获得邻居节点 j 所在的社区       
        j_nodes_c = np.argmax(Xi[:,j_nodes], axis=0)
        j_node_c_dict = dict(zip(j_nodes,j_nodes_c))
        # 计算吸引力(节点i划分到Ck社区的概率)
        attr_sum = np.sum(motif_adj[i,:]) #计算节点i对所有邻居社区的归属程度总和【展示基于模体邻居节点】
        # 如果attr_sum == 0, 依概率选择一个邻居社区作为其划分的社区
        if attr_sum == 0:
            node_cnos.append((i,rd.choice(j_nodes_c))) # choice 依概率选择
            continue   
        # 获得社区及对应的节点
        for ck in set(j_nodes_c):
            ck_jnodes = [key for key in j_node_c_dict.keys() if j_node_c_dict[key] == ck] # 获得社区ck中的个节点
            # 计算节点i对社区c的归属程度
            attr_i_ck = sum([motif_adj[i,j] for j in ck_jnodes])
            c_ps.append((ck,attr_i_ck / attr_sum))
        c = choice_by_probability(c_ps) #依概率选择
        # c = sorted(c_ps, key=lambda x:(x[1]), reverse=True)[0][0]  # 直接选择概率最大的社区作为i节点划分的社区
        node_cnos.append((i,c))  

# =============================================================================
#     NWMM_CD_func: 基于融合模体权重及隶属度信息的权重信息，寻找不合理划分的节点
#     edge_dict: 每条边所对应的模体M的点集和边集
#     unreasonableNodes 划分不合理的节点
#     pick: 一定数量的随机节点
#     Xi: 第i个个体的隶属度矩阵
#     adj: 加权网络的临界矩阵
#     motif_adj: 模体邻接矩阵
#     c: 社区划分的数目
#     n: 网络节点数目
# =============================================================================
def NWMM_CD_func(edge_dict, unreasonableNodes,pick,Xi,adj,motif_adj,me_adj,c,n):
    for i in pick:
        # 获得节点 i 所在的社区
        i_node_c = np.argmax(Xi[:,i])
        # 获得节点 i 的所有邻居节点(边邻居节点+模体邻居节点) j_nodes
        j_nodes = np.ravel(np.nonzero(np.ravel(me_adj[i,:])))        
        # 获得邻居节点 j 所在的社区       
        j_nodes_c = np.argmax(Xi[:,j_nodes], axis=0)
        # 如果 i 节点与其所有邻居节点同属一个社区，则跳过该节点
        Cnei = len(np.unique(np.concatenate((np.ravel(i_node_c),np.ravel(j_nodes_c)))))
        if Cnei == 1:
            continue
        # 计算i与其所有邻居节点的权值总和
        wij_sum = 0
        for j in j_nodes:
            wij_sum = wij_sum + MMW_func(edge_dict, (i,j), Xi, me_adj, c)
        # i_c == j_c ,获取与i节点同一社区的邻居节点集,并求与i节点同一社区的邻居节点的权重总和
        wij = 0
        j_node_c_dict = dict(zip(j_nodes,j_nodes_c))
        
        ijc_nodes = [key for key in j_node_c_dict.keys() if j_node_c_dict[key] == i_node_c]
        for j in ijc_nodes:
            wij = wij + MMW_func(edge_dict, (i,j), Xi, me_adj, c)
        # 如果节点 i 与同一社区的邻居节点之间的权重总和 <= 其与所有邻居节点权重总和的平均值，则返回 i
        if wij - wij_sum/Cnei <= 0:
            unreasonableNodes.append(i)
            
# =============================================================================
#     NWMM_P_func: 寻找节点应划分的社区号
#     edge_dict: 每条边所对应的模体M的点集和边集
#     node_cnos: 节点及对应的社区划分号
#     node_cps: 节点社区归属度字典{i:[(ck,attrk)]}
#     nodes: 未正确划分的节点
#     Xi: 第i个个体的隶属度矩阵
#     adj: 加权网络的临界矩阵
#     motif_adj: 模体邻接矩阵
#     me_adj: #总权重矩阵=模体权重+边权重
#     c: 社区划分的数目
# =============================================================================
def NWMM_P_func(edge_dict, node_cnos,node_cps,nodes,Xi,adj,motif_adj, me_adj, c):
    for i in nodes:
        # 初始化 i 节点对社区 c 的概率
        c_ps = []
        # 获得节点 i 的所有邻居节点(边邻居节点+模体邻居节点) j_nodes
        j_nodes = np.ravel(np.nonzero(np.ravel(me_adj[i,:])))
        # 获得邻居节点 j 真实所在社区
#        j_nodes_c = np.argmax(Xi[:,j_nodes], axis=0)
        j_nodes_c = getRealC(edge_dict, j_nodes, Xi, adj, me_adj, c)
        j_node_c_dict = dict(zip(j_nodes,j_nodes_c))
        # 计算吸引力(节点i划分到Ck社区的概率)
        attr_sum = 0
        for j in j_nodes:
            attr_sum += MMW_func(edge_dict, (i,j), Xi, me_adj, c) #计算节点i对所有邻居社区的归属程度总和【展示基于模体邻居节点】
        # 获得社区及对应的节点
        for ck in set(j_nodes_c):
            ck_jnodes = [key for key in j_node_c_dict.keys() if j_node_c_dict[key] == ck] # 获得社区ck中的个节点
            # 计算节点i对社区c的归属程度
            attr_i_ck = sum([MMW_func(edge_dict, (i,j), Xi, me_adj, c) for j in ck_jnodes])
            c_ps.append((ck,attr_i_ck / attr_sum))    
        c_ps = sorted(copy.deepcopy(c_ps), key=lambda x:(x[1]), reverse=True)
        if c_ps[0][1] != c_ps[-1][1]:
            node_cps[i] = c_ps #通过attr调整
        else:
            i_c = rd.choice(j_nodes_c)
            node_cnos.append((i,i_c)) #通过U=U+0.5调整
#        node_cps[i] = c_ps #通过attr调整
#        i_c = choice_by_probability(c_ps) #依概率选择
#        c_pmax = sorted(copy.deepcopy(c_ps), key=lambda x:(x[1]), reverse=True)[0][0]  # 直接选择概率最大的社区作为i节点划分的社区
#        if i_c == c_pmax:
#            node_cps[i] = c_ps #通过attr调整
#        else:
#            node_cnos.append((i,i_c)) #通过U=U+0.5调整
  
# =============================================================================
#     unreasonableNodes_revise: 修正节点社区编号
#     node_cno_list: 节点和社区编号
#     nmm_pop: nmm种群
#     N: 种群中的第N个个体的序列号
# =============================================================================
def unreasonableNodes_revise(node_cno_list,nmm_pop,N):
    for i_c in node_cno_list:
        i = i_c[0]
        c = i_c[1]
        new_num = nmm_pop[c,i,N] + 0.5
        if new_num > 1.0:
            nmm_pop[c,i,N] = 0.9999
        else:
            nmm_pop[c,i,N] = new_num
        nmm_pop[:,i,N] /= np.sum(nmm_pop[:,i,N]) # 归一化
        
# =============================================================================
#     NWMM_nc_revise: NWMM修正节点社区编号
#     node_cno_list: 节点和社区编号
#     nmm_pop: nmm种群
#     N: 种群中的第N个个体的序列号
# =============================================================================
def NWMM_nc_revise(node_cps,nmm_pop,N):
    for i in node_cps.keys():
        cps = node_cps[i]
        for c_p in cps:
            nmm_pop[c_p[0],i,N] = c_p[1]
 
# =============================================================================
#     choice_by_probability: 依概率选择
#     c_p_list: 节点 i 的候选社区概率列表
#     return : 选择的社区c
# =============================================================================
def choice_by_probability(c_p_list):
    num = 10000
    choice_list = []
    for c_p in c_p_list:
        c = c_p[0]
        p = c_p[1]
        n = int(p*num)
        choice_list += [c]*n
    ic = rd.choice(choice_list) # choice() 依概率选择
    return ic

# =============================================================================
# MMW_func: 融合模体及隶属度信息的边权重
# G: 网络
# edge: 网络的一条边
# Xi: 个体Xi
# return: 返回该条边的融合权重值
# =============================================================================
def MMW_func(edge_dict,edge, Xi, me_adj, c):
    # 寻找基于该边的模体M1的顶点    
    if len(edge_dict[edge]) == 0: # 若该条边未参与模体构建，则返回边权重
        return me_adj[edge[0],edge[1]]
    else:
        # 若该条边参与了模体M1的构建
        node_set, edge_set = edge_dict[edge]
        # edge的融合权重
        edge_W = cfunc.getEdgeW(Xi, me_adj, node_set, edge_set, edge[0], edge[1], c ,node_set.shape[1],node_set.shape[0],edge_set.shape[0])
    return edge_W 

# =============================================================================
#     getRealC: 获得节点所在的真实划分社区
#     edge_dict: 每条边所对应的模体M的点集和边集
#     nodes: 未正确划分的节点
#     Xi: 第i个个体的隶属度矩阵
#     adj: 加权网络的临界矩阵
#     me_adj: #总权重矩阵=模体权重+边权重
#     c: 社区划分的数目
#     return: 各节点所在的真实社区
# =============================================================================
def getRealC(edge_dict, nodes, Xi, adj, me_adj, c):
   cs = []
   for i in nodes:
        # 获得节点 i 所在的社区
        i_node_c = np.argmax(Xi[:,i])
        # 获得节点 i 的所有邻居节点(边邻居节点+模体邻居节点) j_nodes
        j_nodes = np.ravel(np.nonzero(np.ravel(me_adj[i,:])))        # 获得邻居节点 j 所在的社区       
        j_nodes_c = np.argmax(Xi[:,j_nodes], axis=0)
        # 如果 i 节点无邻居节点，且该节点无邻居社区，则跳过该节点
        Cnei = len(np.unique(np.concatenate((np.ravel(i_node_c),np.ravel(j_nodes_c)))))
        if Cnei == 1:
            cs.append(i_node_c)
            continue
        # 计算i与其所有邻居节点的权值总和
        wij_sum = 0
        for j in j_nodes:
            wij_sum = wij_sum + MMW_func(edge_dict, (i,j), Xi, me_adj, c)
        # i_c == j_c ,获取与i节点同一社区的邻居节点集,并求与i节点同一社区的邻居节点的权重总和
        wij = 0
        j_node_c_dict = dict(zip(j_nodes,j_nodes_c))
        
        ijc_nodes = [key for key in j_node_c_dict.keys() if j_node_c_dict[key] == i_node_c]
        for j in ijc_nodes:
            wij = wij + MMW_func(edge_dict, (i,j), Xi, me_adj, c)
        # 如果节点 i 与同一社区的邻居节点之间的权重总和 <= 其与所有邻居节点权重总和的平均值，则返回 i
        if wij - wij_sum/Cnei <= 0:
            c_ps = []
            # 计算吸引力(节点i划分到Ck社区的概率)
            attr_sum = 0
            for j in j_nodes:
                attr_sum += MMW_func(edge_dict, (i,j), Xi, me_adj, c) #计算节点i对所有邻居社区的归属程度总和【展示基于模体邻居节点】
            # 如果attr_sum == 0, 依概率选择一个邻居社区作为其划分的社区
            if attr_sum == 0:
                ic = rd.choice(j_nodes_c.tolist()) # choice() 依概率选择
                cs.append(ic)
                continue
            # 选择候选社区中概率最大的社区作为其当前要划分的社区
            # 获得邻居社区及对应的节点
            for ck in set(j_nodes_c):
                ck_jnodes = [key for key in j_node_c_dict.keys() if j_node_c_dict[key] == ck] # 获得社区ck中的个节点
                # 计算节点 i 对社区c的归属程度
                attr_i_ck = sum([MMW_func(edge_dict, (i,j), Xi, me_adj, c) for j in ck_jnodes])
                c_ps.append((ck,attr_i_ck / attr_sum))               
            c_no = sorted(copy.deepcopy(c_ps), key=lambda x:(x[1]), reverse=True)[0][0]  # 直接选择概率最大的社区作为i节点划分的社区
            cs.append(c_no)
        else:
            cs.append(i_node_c)
   return cs

 # =============================================================================
 #     pop_inherit: 继承以前的种群
 #     n: 节点数
 #     c:  社区数
 #     NP:  种群个体数量
 #     Path: 种群文件路径
 #     return: 返回该种群
 # =============================================================================
def pop_inherit(n, c, NP, Path):
    pop = np.empty((c,n,NP), dtype = float) 
    X = np.empty((c,n), dtype = float) 
    cs_nodes_num = []
    with open(Path, mode='r',encoding='UTF-8') as f:
        cs_nodes_num = f.readlines()
        for index,nodes_num in enumerate(cs_nodes_num):
            nums = nodes_num[1:-2].split(', ')
            nums = list(map(eval,nums))
#            nums = list(map(round,nums,[4 for i in range(len(nums))]))
            X[index,:] = np.asarray(nums)
    for i in range(NP):
        pop[:,:,i] = X
    return pop

# =============================================================================
#    convergence_process_show: 种群进化过程(社区划分收敛过程)展示
#    nmm_best_Q_dict: 种群进化每代中的最优个体的模块度值
#    Gen: 种群进化代数
#    Q_flag: Q值
#    nmmlist: nmm操作列表
# =============================================================================
def convergence_process_show(nmm_best_Q_dict,Gen,Q_flag,nmmlist):
    plt.figure(figsize=(15,7),dpi=800)
    mainColor = (42/256, 87/256, 145/256, 1) # R,G,B,透明度
    plt.title("bestpop convergence process show")
    plt.xlabel('Number of iterations',color=mainColor)
    plt.ylabel('Best'+Q_flag,color=mainColor)             
    plt.tick_params(axis='x',colors=mainColor)  # 坐标轴颜色
    plt.tick_params(axis='y',colors=mainColor)                           
    data_x = np.arange(1,Gen+1)    
    plt.plot(
            data_x,
            nmm_best_Q_dict[nmmlist[0]],
            data_x,
            nmm_best_Q_dict[nmmlist[1]],
            data_x,
            nmm_best_Q_dict[nmmlist[2]],
            data_x,
            nmm_best_Q_dict[nmmlist[3]],
            marker='.',
            color=mainColor,
            lineWidth=2
            )                                                
    plt.grid(True) #设置背景栅格
#    plt.show() # 显示图形  
#    plt.savefig('image/'+'bestpop_convergence_process.png',dpi=800)
        