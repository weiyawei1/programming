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
import copy
import math

# C函数
import cython_function as cfunc

# =============================================================================
#     fit_Qs: 计算种群中每个个体的模糊重叠社区划分的模块度函数Q值
#     pop: 种群
#     Qs: 根据pop计算的模块度值
#     adj: 网络邻接矩阵
#     c: 社区划分的数目
#     n: 网络节点数目
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
    return Qs

# =============================================================================
#     fit_Q: 计算单个个体的模糊重叠社区划分的模块度函数Q值
#     X: 种群中的个体
#     adj: 加权邻接矩阵
#     n: 网络节点数目
#     c: 社区划分的数目
#     W: 加权网络邻接矩阵的总权值之和
#     m: 邻接矩阵各行的权值之和
#     flag: 0:Q 1:Qg 2:Qc_FCD 3:Qc_OCD 4:Qov
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
        #Q =  fit_Qov(X,adj,n,c,W,m)
    else:
        Q=-1
     
    return Q

def fit_Qov(X,adj,n,c,W,m):
    Qov = 0.0
    for k in range(c):
        pointk = np.nonzero(X[k,:])[0]#获得第K个社区的所有节点标号(非零元素的下标)
        nk = len(pointk) #第k个社区的规模
        r = np.zeros((nk,nk))
        w = np.zeros((nk,nk))
        
        for i in range(nk):
            for j in range(nk):
                fuic = 60*X[k,pointk[i]]-30
                fujc = 60*X[k,pointk[j]]-30
                r[i,j] = 1.0/((1+math.exp(-fuic))*(1+math.exp(-fujc)))
                
        for i in range(nk):
            for j in range(nk):
                w[i,j] = np.sum(r[i,:])*np.sum(r[:,j])/pow(nk,2)
                
        for i in range(nk):
            for j in range(nk):
                Qov = Qov + (r[i,j]*adj[pointk[i],pointk[j]] - w[i,j]*(m[0,pointk[i]]*m[0,pointk[j]])/W)
    Qov = Qov/W
    return Qov

# =============================================================================
#     init_pop: 种群初始化
#     c: 社区划分的数目
#     n: 网络节点数目
#     NP： 种群个体数目
# =============================================================================
def init_pop(n,c,NP):
    pop = np.empty((c,n,NP), dtype = float) 
    for N in range(NP):
        for i in range(n):
            membershipList = []
            for k in range(c):
                # rd.seed(N+i+k)
                membershipList.append(rd.random())
            memberships = np.asarray(membershipList)
            memberships = memberships/sum(memberships)  #归一化
            pop[:,i,N]=memberships
    return pop

# =============================================================================
#     bias_init_pop: 对种群有偏操作
#     c: 社区划分的数目
#     n: 网络节点数目
#     NP： 种群个体数目
# =============================================================================
def bias_init_pop(pop,n,c,NP,adj):
    bias_pop = copy.deepcopy(pop)
    for N in range(NP):
        # 在该个体中，随机选择一个节点，将其隶属度赋值给所有相邻节点
        i_rand = rd.randint(0, n-1)  
        for i in range(n):
           if i != i_rand and adj[i_rand,i]>0:
               bias_pop[:,i,N] = bias_pop[:,i_rand,N]

    return bias_pop
        

# =============================================================================
#     bound_check_revise: 边界约束检查与修正
#     c: 社区划分的数目
#     n: 网络节点数目
#     NP： 种群个体数目
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
#     NMM_func: 基于邻域社区的节点社区修正操作函数【1:"NOMM",2:"NMM",3:"MNMM",4:"NWMM"】
#     pop: 种群
#     c: 社区划分的数目
#     n: 网络节点数目
#     NP： 种群中的个体数
#     adj: 无权网络的邻接矩阵
#     motif_adj: 权重矩阵
#     threshold_value: 阈值
#     return: nmm_pop, nmm_fit nmm 种群, 适应度值
# =============================================================================
def NMM_func(pop, n, c, NP, adj, motif_adj, threshold_value, Q_flag, nmm_flag):
    nmm_pop = copy.deepcopy(pop)
    nmm_fit = []
    if nmm_flag=="NOMM":
        ###NOMM###
        fit_Qs(nmm_fit,nmm_pop,adj,n,c,NP,Q_flag)   #适应度函数值计算 
    elif nmm_flag=="NMM":
        ###NMM###
        NMM(pop, n, c, NP, adj, motif_adj, threshold_value, Q_flag, nmm_pop, nmm_fit)
    elif nmm_flag=="MNMM":
        ###MNMM###
        MNMM(pop, n, c, NP, adj, motif_adj, threshold_value, Q_flag, nmm_pop, nmm_fit)
    elif nmm_flag=="NWMM":   
        ###NWMM###
        NWMM(pop, n, c, NP, adj, motif_adj, threshold_value, Q_flag, nmm_pop, nmm_fit)
    # 返回种群和对应的模块度值
    return (nmm_pop, nmm_fit)

# =============================================================================
#     NMM: 基于邻居节点的社区修正（仅基于边邻居节点）
#     nmm_pop: 种群
#     c: 社区划分的数目
#     n: 网络节点数目
#     NP： 种群中的个体数
#     adj: 无权网络的邻接矩阵
#     threshold_value: 阈值
#     return: nmm_pop, nmm_fit nmm种群, 适应度值
# =============================================================================
def NMM(pop, n, c, NP, adj, motif_adj, threshold_value, Q_flag, nmm_pop, nmm_fit):
    for i in range(NP):
        seeds = [i for i in range(n)]
#        rd.shuffle(seeds)
#        pick = seeds[:rd.randint(1, n)] # 随机选择一定数量的节点
        pick = seeds
        # 寻找不合理划分的节点和其对应的邻居节点
        unreasonableNodes = []
        NMM_CD_func(unreasonableNodes, pick,nmm_pop[:,:,i],adj,c,n,threshold_value)
        # 获得该节点应划分的社区号
        node_cno_list=[]
        NMM_P_func(node_cno_list,unreasonableNodes,nmm_pop[:,:,i],adj)
        # 修改该节点的隶属度值，对该节点重新划分社区
        unreasonableNodes_revise(node_cno_list,nmm_pop,i)
    
    # 计算该种群的适应度函数值
    fit_Qs(nmm_fit,nmm_pop,adj,n,c,NP,Q_flag)   #适应度函数值计算           

# =============================================================================
#     MNMM: 基于邻居节点的社区修正（基于边邻居节点和模体邻居节点,使用了模体权重信息）
#     nmm_pop: 种群
#     nmm_fit: 当前种群中个体的模块度值
#     c: 社区划分的数目
#     n: 网络节点数目
#     NP： 种群中的个体数
#     adj: 无权网络的邻接矩阵
#     motif_adj: 模体加权网络的邻接矩阵
#     threshold_value: 阈值
# =============================================================================
def MNMM(pop, n, c, NP, adj, motif_adj, threshold_value, Q_flag, nmm_pop, nmm_fit):
    for i in range(NP):
        seeds = [i for i in range(n)]
#        rd.shuffle(seeds)
#        pick = seeds[:rd.randint(1, n)] # 随机选择一定数量的节点
        pick = seeds
        # 寻找不合理划分的节点和其对应的邻居节点
        unreasonableNodes = []
        find_unreasonableNodes_motifadd_V2(unreasonableNodes,pick,nmm_pop[:,:,i],adj,motif_adj,c,n,threshold_value)
        # 获得该节点应划分的社区号
        node_cnos = []
        find_node_cno_motifadd(node_cnos,unreasonableNodes,nmm_pop[:,:,i],adj,motif_adj)
#        print("len=",len(node_cnos))
        # 修改该节点的隶属度值，对该节点重新划分社区
        unreasonableNodes_revise(node_cnos,nmm_pop,i)
    # 计算该种群的适应度函数值    
    fit_Qs(nmm_fit,nmm_pop,adj,n,c,NP,Q_flag)   #适应度函数值计算 

# =============================================================================
#     NWMM: 基于邻居节点的社区修正（基于边邻居节点和模体邻居节点,使用了模体权重信息）
#     nmm_pop: 种群
#     c: 社区划分的数目
#     n: 网络节点数目
#     NP： 种群中的个体数
#     adj: 无权网络的邻接矩阵
#     motif_adj: 模体加权网络的邻接矩阵
#     threshold_value: 阈值
#     return: nmm_pop, nmm_fit nmm种群, 适应度值
# =============================================================================
def NWMM(pop, n, c, NP, adj, motif_adj, threshold_value, Q_flag, nmm_pop, nmm_fit):
    for i in range(NP):
        seeds = [i for i in range(n)]
#        rd.shuffle(seeds)
#        pick = seeds[:rd.randint(1, n)] # 随机选择一定数量的节点
        pick = seeds
        # 寻找不合理划分的节点和其对应的邻居节点
        unreasonableNodes = []
        find_unreasonableNodes_motifadd_V2(unreasonableNodes,pick,nmm_pop[:,:,i],adj,motif_adj,c,n,threshold_value)
        # 获得该节点应划分的社区号
        node_cnos = []
        find_node_cno_motifadd(node_cnos,unreasonableNodes,nmm_pop[:,:,i],adj,motif_adj)
#        print("len=",len(node_cnos))
        # 修改该节点的隶属度值，对该节点重新划分社区
        unreasonableNodes_revise(node_cnos,nmm_pop,i)
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
#     find_node_cno: 寻找节点应划分的社区号
#     node_cno_list:
#     nodes: 未正确划分的节点
#     Xi: 第i个个体的隶属度矩阵
#     adj: 加权网络的临界矩阵
#     c: 社区划分的数目
#     n: 网络节点数目
# =============================================================================
def NMM_P_func(node_cno_list,nodes,Xi,adj):
    for i in nodes:
        # 获得 i 基于边的邻接节点
        j_nodes = np.nonzero(adj[i,:])[1]
        # 获得邻居节点 j 所在的社区       
        j_nodes_c = np.argmax(Xi[:,j_nodes], axis=0)
        # print("j_nodes_c=",j_nodes_c)
#        node_cno_list.append((i,rd.choice(j_nodes_c)))  # choice() 依概率选择
        i_c = np.argmax(np.bincount(j_nodes_c)) # 直接选择概率最大的社区作为i节点划分的社区
        node_cno_list.append((i,i_c))
        
#def NMM_P_func(node_cno_list,nodes,Xi,adj):
#    for i in nodes:
#        # 获得 i 基于边的邻接节点
#        j_nodes = np.nonzero(adj[i,:])[1]
#        # 获得邻居节点 j 所在的社区       
#        j_nodes_c = np.argmax(Xi[:,j_nodes], axis=0)
#        # print("j_nodes_c=",j_nodes_c)
#        node_cno_list.append((i,rd.choice(j_nodes_c)))  # choice() 依概率选择


# =============================================================================
#     find_unreasonableNodes_motifadd_V1: 寻找基于边的不合理划分的节点
#     pick: 一定数量的随机节点
#     Xi: 第i个个体的隶属度矩阵
#     adj: 加权网络的临界矩阵
#     c: 社区划分的数目
#     n: 网络节点数目
#     threshold_value: 阈值
#     reutrn: unreasonableNodes 划分不合理的节点集合
# =============================================================================
def find_unreasonableNodes_motifadd_V1(unreasonableNodes,pick,Xi,adj,motif_adj,c,n,threshold_value):
    for i in pick:
        # 获得节点 i 所在的社区
        i_node_c = np.argmax(Xi[:,i])
        # 寻找节点 i 基于边的邻居节点 j_e_nodes
        j_e_nodes = np.nonzero(adj[i,:])[1]
        # 寻找节点 i 基于模体的邻居节点 j_m_nodes
        j_m_nodes = np.nonzero(motif_adj[i,:])[1]
        # 获得节点 i 的所有邻居节点 j_nodes
        j_nodes = list(set(j_e_nodes) | set(j_m_nodes))
        # 如果 i 节点无邻居节点，则跳过该节点
        if len(j_nodes) == 0:
            continue
        # 获得邻居节点 j 所在的社区       
        j_nodes_c = np.argmax(Xi[:,j_nodes], axis=0)
        # i_c != j_c
        cd_i = np.where(j_nodes_c != i_node_c)[0].shape[0] / len(j_nodes)
        # 如果节点 i 划分不合理程度大于阈值，则返回 i
        if cd_i > threshold_value :
            unreasonableNodes.append(i)
            
# =============================================================================
#     find_unreasonableNodes_motifadd_V2: 寻找基于模体权重的不合理划分的节点
#     pick: 一定数量的随机节点
#     Xi: 第i个个体的隶属度矩阵
#     adj: 加权网络的临界矩阵
#     c: 社区划分的数目
#     n: 网络节点数目
#     threshold_value: 阈值
#     reutrn: unreasonableNodes 划分不合理的节点集合
# =============================================================================
def find_unreasonableNodes_motifadd_V2(unreasonableNodes,pick,Xi,adj,motif_adj,c,n,threshold_value):
    for i in pick:
        # 获得节点 i 所在的社区
        i_node_c = np.argmax(Xi[:,i])
        # 寻找节点 i 基于边的邻居节点 j_e_nodes
        j_e_nodes = np.nonzero(adj[i,:])[1]
        # 寻找节点 i 基于模体的邻居节点 j_m_nodes
        j_m_nodes = np.nonzero(motif_adj[i,:])[1]
        # 获得节点 i 的所有邻居节点 j_nodes
        j_nodes = list(set(j_e_nodes) | set(j_m_nodes))
        # 获得邻居节点 j 所在的社区       
        j_nodes_c = np.argmax(Xi[:,j_nodes], axis=0)
        # 如果 i 节点无邻居节点，且该节点无邻居社区，则跳过该节点
        Ck = len(set(j_nodes_c) | set([i_node_c]))
        if len(j_nodes) == 0 or Ck == 1:
            continue
        # 计算i与邻居节点的权值总和
        wij_sum = 0
        for j in j_nodes:
            wij_sum += motif_adj[i,j]

        # i_c == j_c ,获取与i节点同一社区的邻居节点集,并求与i节点同一社区的邻居节点的权重总和
        wij = 0
        ij_nodes_c = np.where(j_nodes_c == i_node_c)[0]
        for j_c in ij_nodes_c:
            wij += motif_adj[i,j_nodes[j_c]]
        # 如果节点 i 与同一社区的邻居节点之间的权重总和 <= 其与所有邻居节点权重总和的平均值，则返回 i
        
        if wij <= wij_sum/Ck:
            unreasonableNodes.append(i)

# =============================================================================
#     find_node_cno_motifadd: 寻找节点应划分的社区号
#     nodes: 未正确划分的节点
#     Xi: 第i个个体的隶属度矩阵
#     adj: 加权网络的临界矩阵
#     c: 社区划分的数目
#     n: 网络节点数目
#     return: 节点社区归属度字典{i:[(ck,attrk)]}
# =============================================================================
def find_node_cno_motifadd(node_cnos,nodes,Xi,adj,motif_adj):
        
    for i in nodes:
        # 初始化 i 节点对社区 c 的概率
        c_ps = []
        # 获得节点 i 所在的社区
#        i_node_c = np.argmax(Xi[:,i])
        # 寻找节点 i 基于边的邻居节点 j_e_nodes
        j_e_nodes = np.nonzero(adj[i,:])[1]
        # 寻找节点 i 基于模体的邻居节点 j_m_nodes
        j_m_nodes = np.nonzero(motif_adj[i,:])[1]
        # 获得节点 i 的所有邻居节点 j_nodes
        j_nodes = list(set(j_e_nodes) | set(j_m_nodes))
        
        # 获得邻居节点 j 所在的社区       
        j_nodes_c = np.argmax(Xi[:,j_nodes], axis=0)
        
        clist = set(j_nodes_c)
        # 获得社区及对应的节点
        c_jnodes = {}
        for c in clist:
            nodes = []
            for index, j_c in enumerate(j_nodes_c):
                if j_c == c:
                    nodes.append(j_nodes[index])
            c_jnodes[c] = nodes         
        # print("j_nodes_c=",j_nodes_c)
        attr_sum = 0.0  # C 个社区的归属度之和
        for c in clist:
            # 计算节点 i 对社区 c 的归属度
            Si = np.sum(motif_adj[i,:])
            if Si == 0:
                c_ps.append((c, 1))
                break
                
            c_j_nodes = c_jnodes[c]
            W = 0
            for j in c_j_nodes:
                W += motif_adj[i,j]
            # 计算归属度
            attr = W / Si
            attr_sum += attr
            p = 0.0000
            if attr_sum > 0.0001:
                p += attr/attr_sum
                
            c_ps.append((c, p))

        #   依概率选择
        c = choice_by_probability(c_ps)
        node_cnos.append((i,c))  
        
# =============================================================================
#     unreasonableNodes_revise: 修正节点社区编号
#     node_cno_list: 节点和社区编号
#     Xi: 第i个个体的隶属度矩阵
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
 #     pop_inherit: 继承以前的种群
 #     node_cnos_attrs: 节点_社区编号_归属程度
 #     nmm_pop: 种群隶属度矩阵
 #     c:  社区数
 #     N:  种群中的第 N 个个体
 #     adj: 加权网络的临界矩阵
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


"""
# =============================================================================
#     MNMM_V1: 基于邻居节点的社区修正（基于边邻居节点和模体邻居节点）
#     nmm_pop: 种群
#     c: 社区划分的数目
#     n: 网络节点数目
#     NP： 种群中的个体数
#     adj: 无权网络的邻接矩阵
#     motif_adj: 模体加权网络的邻接矩阵
#     threshold_value: 阈值
#     return: nmm_pop, nmm_fit nmm种群, 适应度值
# =============================================================================
def MNMM_V1(pop, n, c, NP, adj, motif_adj, threshold_value,Q_flag):
    nmm_pop = copy.deepcopy(pop)
    for i in range(NP):
        seeds = [i for i in range(n)]
        rd.shuffle(seeds)
        pick = seeds[:rd.randint(1, n)] # 随机选择一定数量的节点
        # 寻找不合理划分的节点和其对应的邻居节点
        unreasonableNodes = []
        find_unreasonableNodes_motifadd_V1(unreasonableNodes,pick,nmm_pop[:,:,i],adj,motif_adj,c,n,threshold_value)
        # 获得该节点应划分的社区号
        node_cnos = []
        find_node_cno_motifadd(node_cnos,unreasonableNodes,nmm_pop[:,:,i],adj,motif_adj)
        # 修改该节点的隶属度值，对该节点重新划分社区
        unreasonableNodes_revise(node_cnos,nmm_pop,i)
    # 计算该种群的适应度函数值    
    nmm_fit = fit_Qs(nmm_pop,adj,n,c,NP,Q_flag)   #适应度函数值计算 
    return (nmm_pop, nmm_fit)
"""


























