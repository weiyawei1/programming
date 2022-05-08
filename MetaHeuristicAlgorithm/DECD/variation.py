# -*- coding: utf-8 -*-
"""
Created on Wed Nov  3 09:45:00 2021

@author: WYW
"""
import random

"""
变异操作
    pop: 初始种群
    NP: 种群中的个体数[网络划分的社区数]
    D:  参与变异、交叉的种群个体的分量数
    F:  缩放因子
    domain: 种群搜索范围的边界约束[ Lbound,Ubound ]
"""
def variation_operation(pop,NP,D,F,domain):
    mutation_pop = []
    for i in range(NP):
        a = randomIntX(NP,i)  #x in [0,NP-1] 
        b = randomIntX(NP,i,a)
        c = randomIntX(NP,i,a,b)
        # 构造第i个个体对应的变异个体V
        V = []
        for j in range(D):
            vec = int(pop[a][j] + F * (pop[b][j] - pop[c][j]))
            # 限制每一维分量的取值范围(是否违反边界约束条件)
            if vec < domain[j][0]:
                vec1 = max(domain[j][0],int(2*domain[j][0] - vec))
            elif vec > domain[j][1]:
                vec1 = min(domain[j][1],int(2*domain[j][1] - vec))
            else:
                vec1 = vec
            V.append(vec1)
        # 将第i个个体对应的变异个体V存入变异种群mutation_pop
        mutation_pop.append(V)
    # 返回变异种群
    return mutation_pop

"""
用于产生不同于*param中任何元素的随机整数
"""
def randomIntX(maxRandInt, *param):
    x = random.randint(0,maxRandInt-1)  # x in [0,NP-1]
    while 1:
        if x not in param:
            return x
        else :
            x = random.randint(0,maxRandInt-1)