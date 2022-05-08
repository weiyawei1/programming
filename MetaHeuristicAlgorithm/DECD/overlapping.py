# -*- coding: utf-8 -*-
"""
Created on Wed Nov  3 10:05:37 2021

@author: WYW
"""
import random
from numpy import random
import copy

"""
交叉操作
    pop: 初始种群
    crossover_pop: 实验种群【交叉后的种群】
    NP:  代数
    Fit: 用于分数存储
    G:   网络
"""
def overlapping_operation(pop,NP,D,CR,mutation_pop):
    crossover_pop = copy.deepcopy(pop)  # 深拷贝，两者是完全独立的
    # 根据DE算法的交叉操作，以概率CR，保留变异种群mutation_pop中的社区性状
    for i in range(NP):
        # 在[0，n-1]范围内，随机选择一维分量
        rand_j = random.randint(0, D)  # rand_j in [0,D-1] 与文章表述不一致
        # rand_j = random.randint(1, D+1) # jrand is 从1到n中随机选择的整数
        for j in range(D):
            # random.rand返回一个或一组服从“0~1”均匀分布的随机样本值
            if random.rand(0, 1) <= CR or j == rand_j:  
                # 变异个体i中第j维分量对应的值
                comm_id_j = mutation_pop[i][j]
                # all_nodes_j是变异个体i中社团编号都为V_ij节点位置
                all_nodes_j = []
                all_nodes_j.append(j)
                for k in range(D):
                    if k != j and mutation_pop[i][k] == comm_id_j:
                        all_nodes_j.append(k)
                # 交叉个体i中上述节点集合的社区标号全部改为comm_id_j
                for k in range(D):
                    if k in all_nodes_j:
                        crossover_pop[i][k] = comm_id_j
                        
    return crossover_pop