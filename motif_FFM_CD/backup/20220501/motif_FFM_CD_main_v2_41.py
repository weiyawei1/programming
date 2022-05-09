# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 09:55:19 2022

@author: WYW
"""
"""
    FFM_CD_main_v1_NMM
   使用各种优化算法，基于模体的加权网络的社区检测
"""

import numpy as np
import igraph as ig
import networkx as nx
import os
import time
import copy
# 各模块函数
import motif_network_construct as net_stru_func
import algorithm_FCD_function as alg_func
import motif_FFM_CD_function as func

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
network = beican_9_network
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
Gen = 200  #进化代数
threshold_value = 0.25  #阈值
Mlist = [1,2,3,4,5,6,7,8] #模体选择【1:M1,2:M2,3:M3,4:M4,5:M5,6:M6,7:M7,8:M8】
Q_flags = [0,1,2,3,4]  # Q_flag: 0:Q 1:Qg 2:Qc_FCD 3:Qc_OCD 4:Qov 
Q_flag=Q_flags[2] # Qc
Independent_Runs = 1 # 本次实验独立运行次数
 
# =============================================================================
# 构建基于模体M1的加权网络
# =============================================================================
G = net_stru_func.construct_weighted_network(Gi,n,Mlist[0]) # 基于M1模体

# 获得无权网络邻接矩阵
G2 = nx.Graph() 
G2.add_nodes_from([i for i in range(n)])
G2.add_edges_from(edge_all)
adj= nx.adjacency_matrix(G2)
adj=adj.todense() 

# 构建基于模体的加权网络邻接矩阵motif_adj
#motif_adj = nx.adjacency_matrix(G)
#motif_adj = motif_adj.todense() 
# 获得模体邻接矩阵
motif_adjs = [] # 模体邻接矩阵列表
for M in Mlist:
    motif_adj = net_stru_func.get_motifadd_adj(G, edge_all, M)
    motif_adjs.append(np.matrix(motif_adj))

run = 0
while (run < Independent_Runs):
# # 设置全局变量
    pop_best_history = np.zeros((c,n,Gen)) # 用于保存历史最优的个体记录
    best_in_history_Q = [] # 用于保存历史最优Q值
    
#    # 初始化NMi
#    nmilist = [] # 用于保存每一代的NMI值
#    # 获取真实社区划分列表
#    real_mem = []
#    with open(path + "/real/" + 'beican_9_groundtruth_4.txt', mode='r',encoding='UTF-8') as f:
#        real_mem = list(map(int,f.read().splitlines()))

    start = time.process_time()
    # =============================================================================
    # 种群初始化，有偏操作
    # =============================================================================
    #种群初始化
    pop = func.init_pop(n, c, NP)  #初始化种群
    fit_values = func.fit_Qs(pop,adj,n,c,NP,Q_flag)   #适应度函数值计算 
    # end = clock()
    # print("spend_time=",end - start)
    
    #有偏操作
    bias_pop = func.bias_init_pop(pop, c, n, NP, adj) # 对初始化后的种群进行有偏操作
    bias_fit_values = func.fit_Qs(bias_pop,adj,n,c,NP,Q_flag)   #适应度函数值计算 
    #选择优秀个体并保留到种群
    for index in range(NP):
        if bias_fit_values[index] > fit_values[index]:
            pop[:,:,index] = bias_pop[:,:,index]    #保存优秀个体
            fit_values[index] = bias_fit_values[index] #保存优秀个体的适应度函数值
    
    # =============================================================================
    # Main
    #【使用优化算法进行社区检测】
    # =============================================================================
    nmm_flag = 1
    motif_adj_index = 0
    best_Q_tmp = 0
    for gen in range(Gen):
        # SOSFCD 算法
        (new_pop, new_fit) = alg_func.SOSFCD(pop, fit_values, n, c, NP, adj,Q_flag)   
    
        # 使用所有模体进行 MNMM_v2 操作
        motif_M = motif_adj_index % len(Mlist)
        (nmm_pop, nmm_fit) = func.MNMM_V2(new_pop, n, c, NP, adj, motif_adjs[motif_M], threshold_value, Q_flag)
        
        # 选择优秀个体并保留到种群
        better_number = 0
        for index in range(NP):
            if nmm_fit[index] > new_fit[index]:
                new_pop[:,:,index] = nmm_pop[:,:,index]    #保存优秀个体
                new_fit[index] = nmm_fit[index] #保存优秀个体的适应度函数值
                better_number+=1
    
        # MNMM_v1 操作
#        (nmm_pop), nmm_fit = func.MNMM_V1(new_pop, n, c, NP, adj, motif_adj, threshold_value)

        #test
        # best_Q = max(new_fit)
        # bestx = new_pop[:,:,new_fit.index(best_Q)]
        # membership_c = np.argmax(bestx, axis=0)
        # print("best_Q={}".format(best_Q))
        # print("membetrship_c={}".format(membership_c))
    
    
        # 更新pop,fit
        pop = copy.deepcopy(new_pop)
        fit_values = copy.deepcopy(new_fit)
        
        # 记录当代最优个体Xbest，并记录最优个体对应的Q值及NMI
        # print("best_Q={}".format(max(fit_values)))
        best_Q = max(fit_values)
        bestx = pop[:,:,fit_values.index(best_Q)]
        best_in_history_Q.append(best_Q)
        pop_best_history[:,:,gen] = bestx
        membership_c = np.argmax(bestx, axis=0)
#        nmi=ig.compare_communities(real_mem, membership_c, method='nmi', remove_none=False)    
#        nmilist.append(nmi)
        
        if better_number == 0 or (best_Q - best_Q_tmp) == 0:
#            print("========换M{}=======".format((motif_M+2)%len(Mlist)))
            motif_adj_index += 1
            
        best_Q_tmp = best_Q
#        if len(set(membership_c)) <4:
#            break
        if (gen+1) % 20 ==0:
            print("#####motif_SOSFCD_Qov_nmm_v3#####")
            print("gen=",gen+1)
            print("c_count=",len(set(membership_c)))
#            print("NMI=",nmi)
            print("bestQov_nmm_v3=",best_Q)
            print("better_number={}".format(better_number))
            
    end = time.process_time()
    
    print("spend_time=",end - start)
    run+=1
    
    # =============================================================================
    # 有选择地保存历史信息到文件
    # =============================================================================
#    bestQ = best_in_history_Q[-1]
#    bestX = pop_best_history[:,:,Gen-1]
#    bestX_membership = np.argmax(bestx, axis=0)
#    # 保存每次独立实验的最优值到日志文件
#    if len(set(bestX_membership)) == 3:
#        logs_path = r"./logs/motif_SOSFCD/zhang/zhang_Qov_nmm_v3_log.txt"
#        with open(logs_path, mode='a+',encoding='UTF-8') as log_f:
#            log_f.writelines("run[" + str(run) + "]=" + str(bestQ) + "\n")
#            log_f.writelines("run[" + str(run) + "]=" + str(bestX_membership) + "\n")
#            print("===========running is {0}==============".format(run))
#        run += 1
#        # 保存目前n次的独立实验中的最优值
#        bestQ_path = r"./paras/motif_SOSFCD/zhang/bestQov_nmm_v3.txt"
#        flag = 0
#        with open(bestQ_path, mode="r+") as f:
#            tempQ = eval(f.read())
#            if bestQ > tempQ:
#                flag = 1
#                
#        if flag == 1:
#            ## 保存最优值
#            with open(bestQ_path, mode='w',encoding='UTF-8') as f:
#                f.write(str(bestQ))
#            ## 更新最优解    
#            pop_best_history_path = r"result/motif_SOSFCD/zhang/pop_best_Qov_nmm_v3_history.txt"
#            fit_best_history_path = r"result/motif_SOSFCD/zhang/fit_best_Qov_nmm_v3_history.txt"
#            nmi_path = r"result/motif_SOSFCD/zhang/NMI_Qov_nmm_v3_history.txt"
#            
#            try:
#                if not os.path.exists(r"result/motif_SOSFCD/zhang"):
#                    os.makedirs(r"result/motif_SOSFCD/zhang") 
#                    
#                with open(pop_best_history_path, mode='w',encoding='UTF-8') as pop_f:
#                    for gen in range(Gen):
#                        pop_f.write("gen_Xbest[" + str(gen) + "]:\n")
#                        X = pop_best_history[:,:,gen]
#                        for k in range(c):
#                            pop_f.write(str(list(X[k,:])) + '\n')
#                        pop_f.writelines("\n")
#            
#                with open(fit_best_history_path, mode='w',encoding='UTF-8') as fit_f:
#                    for gen in range(Gen):
#                        fit_f.writelines("gen_Fitbest[" + str(gen) + "]=" + str(best_in_history_Q[gen]) + "\n")
#            
#                with open(nmi_path, mode='w',encoding='UTF-8') as nmi_f:
#                    for gen in range(Gen):
#                        nmi_f.writelines("gen_NMI[" + str(gen) + "]=" + str(nmilist[gen]) + "\n")
#            
#            except IOError as e:
#                print("Operattion failed:{}".format(e.strerror))
#    
#    
    
    
    
    
    
    
    
    
    
    
    
