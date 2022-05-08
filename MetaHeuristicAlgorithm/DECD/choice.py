# -*- coding: utf-8 -*-
"""
Created on Wed Nov  3 10:13:33 2021

@author: WYW
"""
import igraph as ig

"""
选择操作
    pop: 初始种群
    crossover_pop: 实验种群【交叉后的种群】
    NP:  代数
    Fit: 用于分数存储
    G:   网络
"""
def choice_operation(pop,crossover_pop,NP,Fit,G):
    # 将crossover_pop中的优秀个体保留至下一代种群pop
    for i in range(NP):
        score = ig.GraphBase.modularity(G, crossover_pop[i])  # 计算每一个试验个体的Q
        if score > Fit[i]:  # 比较每一个新个体和老个体的Q
            pop[i] = crossover_pop[i]  # 把Q高的给初始种群作为第二代（其实这个[:]可以不用）
            Fit[i] = score  # 储存新个体的Q值