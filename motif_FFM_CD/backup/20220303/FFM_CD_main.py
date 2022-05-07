# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 09:55:19 2022

@author: WYW
"""
"""
    FFM_CD_main
   使用各种优化算法，对加权网络的社区检测
"""

import numpy as np
import igraph as ig
import networkx as nx
import os
import time
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
zhang_network = path + r'/zhang.txt'
karate_network = path + r'/karate.txt'
dolphins_network = path + r'/dolphins.txt'
football_network = path + r'/football.txt'
polbooks_network = path + r'/polbooks.txt'

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
c = 3  #社区的真实划分数
Gen = 60  #进化代数
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

# # 设置全局变量
pop_best_history = np.zeros((c,n,Gen)) # 用于保存历史最优的个体记录
best_in_history_Q = [] # 用于保存历史最优Q值

# 初始化NMi
nmilist = [] # 用于保存每一代的NMI值
# 获取真实社区划分列表
real_mem = []
with open(path + "/real/" + 'zhang_groundtruth.txt', mode='r',encoding='UTF-8') as f:
    real_mem = list(map(int,f.read().splitlines()))

start = time.process_time()
# =============================================================================
# 种群初始化，有偏操作
# =============================================================================
#种群初始化
pop = func.init_pop(n, c, NP)  #初始化种群
fit_values = func.fit_Qs(pop,adj,n,c,NP)   #适应度函数值计算 
# end = clock()
# print("spend_time=",end - start)

#有偏操作
bias_pop = func.bias_init_pop(pop, c, n, NP, adj) # 对初始化后的种群进行有偏操作
bias_fit_values = func.fit_Qs(bias_pop,adj,n,c,NP)   #适应度函数值计算 
#选择优秀个体并保留到种群
for index in range(NP):
    if bias_fit_values[index] > fit_values[index]:
        pop[:,:,index] = bias_pop[:,:,index]    #保存优秀个体
        fit_values[index] = bias_fit_values[index] #保存优秀个体的适应度函数
        
#Qs = []
# =============================================================================
# Main
#【使用优化算法进行社区检测】
# =============================================================================
for gen in range(Gen):
    # SOSFCD 算法
    new_pop, new_fit = alg_func.SOSFCD(pop, fit_values, n, c, NP, adj)      
    
    # NMM 操作
    nmm_pop, nmm_fit = func.NMM(new_pop, n, c, NP, adj, adj, threshold_value) 

    # MNMM 操作
    # nmm_pop, nmm_fit = func.MNMM(new_pop, n, c, NP, adj, motif_adj, threshold_value) 
    
    # 选择优秀个体并保留到种群
    better_number = 0
    for index in range(NP):
        if nmm_fit[index] > new_fit[index]:
            new_pop[:,:,index] = nmm_pop[:,:,index]    #保存优秀个体
            new_fit[index] = nmm_fit[index] #保存优秀个体的适应度函数值
            # better_number+=1
    # print("better_number={}".format(better_number))
    
    #test
    # best_Q = max(new_fit)
    # bestx = new_pop[:,:,new_fit.index(best_Q)]
    # membership_c = np.argmax(bestx, axis=0)
    # print("best_Q={}".format(best_Q))
    # print("membetrship_c={}".format(membership_c))


    # 更新pop,fit
    pop = new_pop
    fit_values = new_fit
    
    # 记录当代最优个体Xbest，并记录最优个体对应得Q
    # print("best_Q={}".format(max(fit_values)))
    best_Q = max(fit_values)
    bestx = pop[:,:,fit_values.index(best_Q)]
    best_in_history_Q.append(best_Q)
    pop_best_history[:,:,gen] = bestx
    membership_c = np.argmax(bestx, axis=0)
    nmi=ig.compare_communities(real_mem, membership_c, method='nmi', remove_none=False)    
    nmilist.append(nmi)
    if (gen+1) % 20==0:
        print("#####SOSFCD#####")
        print("gen=",gen+1)
        print(membership_c)
        print("bestNMI=",nmi)
        print("bestQw=",best_Q)
        
end = time.process_time()
print("spend_time=",end - start)

# =============================================================================
# 有选择地保存历史信息到文件
# =============================================================================
bestQ = max(best_in_history_Q)
bestQ_path = r"./paras/SOSFCD/zhang/bestQ.txt"
flag = 0
with open(bestQ_path, mode="r+") as f:
    tempQ = eval(f.read())
    if bestQ > tempQ:
        flag = 1
        
if flag == 1:
    ## 保存最优值
    with open(bestQ_path, mode='w',encoding='UTF-8') as f:
        f.write(str(bestQ))
    ## 更新最优解    
    pop_best_history_path = r"result/SOSFCD/zhang/Q_pop_best_history.txt"
    fit_best_history_path = r"result/SOSFCD/zhang/Q_fit_best_history.txt"
    nmi_path = r"result/SOSFCD/zhang/NMI_Q_history.txt"
    try:
        if not os.path.exists(r"result/SOSFCD/zhang"):
            os.makedirs(r"result/SOSFCD/zhang") 
            
        with open(pop_best_history_path, mode='w',encoding='UTF-8') as pop_f:
            for gen in range(Gen):
                pop_f.write("gen_Xbest[" + str(gen) + "]:\n")
                X = pop_best_history[:,:,gen]
                for k in range(c):
                    pop_f.write(str(list(X[k,:])) + '\n')
                pop_f.writelines("\n")

    
        with open(fit_best_history_path, mode='w',encoding='UTF-8') as fit_f:
            for gen in range(Gen):
                fit_f.writelines("gen_Fitbest[" + str(gen) + "]=" + str(best_in_history_Q[gen]) + "\n")
    
        with open(nmi_path, mode='w',encoding='UTF-8') as nmi_f:
            for gen in range(Gen):
                nmi_f.writelines("gen_NMI[" + str(gen) + "]=" + str(nmilist[gen]) + "\n")
    
    except IOError as e:
        print("Operattion failed:{}".format(e.strerror))













