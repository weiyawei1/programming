# -*- coding: utf-8 -*-
"""
Created on Tue Nov  2 21:28:55 2021

@author: WYW
"""
from numpy import random
import igraph as ig

"""
种群初始化
    pop: 初始种群
    n:  种群中个体的分量数【网络的节点数】
    NP: 种群中的个体数[网络划分的社区数]
    G:  网络G
    D:  参与变异、交叉的种群个体的分量数
    domain: 种群搜索范围的边界约束[Lbound,Ubound]
    Fit:  记录分数 
"""
# 构建初始种群，每个个体代表一个社区划分，每个元素对应节点所属社区标号
def pop_init(pop,n,NP,G,D,domain,Fit):
    
    # 限定种群个体中各维取值范围
    xmin = 1
    xmax = n
    # 每一维分量的取值范围[1,n]
    for i in range(0, n):
        domain.append((xmin, xmax))
    # 构建初始种群
    # 使用列表解析
    # 注意：randint函数对于取值范围，包括起始值，但是不包括终值
    ## 对DECD 算法再初始化过程中设置偏好操作，提高收敛速率
    for j in range(NP):
        vec = [random.randint(domain[j][0], domain[j][1] + 1)  # 生成 j in [1,n]
               for j in range(D)  # 每个个体都有j维分量
               ]
        pop.append(vec)
    # 计算个体适应度值
    for i in range(NP):
        fit = ig.GraphBase.modularity(G, pop[i])
        Fit.append(fit)
        
        