# -*- coding: utf-8 -*-
"""
Created on Mon Nov 29 18:18:55 2021

@author: WYW
"""
# 差分进化算法
import numpy as np
import matplotlib.pyplot as plt
import math
import random
import sys
import time


# =============================================================================
# 各个函数
# =============================================================================
# Rastrigr 函数
# 一种标准测试函数，全局最优点：Xi = 0,f(x) = 0；可有效验证算法的全局优化性能
def rastrigr_function(x):
    f = 0
    for i in range(0, len(x)):
        f = f + (x[i] ** 2 - (10 * math.cos(2 * np.pi * x[i])) + 10)
    return f


# 评价
def evaluate(x):
    return [rastrigr_function(i) for i in x]


"""
用于产生不同于*param中任何元素的随机整数
"""


def randomIntX(maxRandInt, *param):
    x = random.randint(0, maxRandInt - 1)  # x in [0,NP-1]
    while 1:
        if x not in param:
            return x
        else:
            x = random.randint(0, maxRandInt - 1)


# 参数
def initpara():
    NP = 100  # 种群数量
    F = 0.3  # 缩放因子
    CR = 0.45  # 交叉概率
    generation = 5000  # 遗传代数
    gen_number = 30  # 基因数量（对应种群中每个个体的分量数）
    pop = np.zeros((NP, gen_number), dtype=np.float, order='c')  # 种群，染色体
    up_range = 5.12
    down_range = -5.12
    return pop, NP, F, CR, generation, gen_number, up_range, down_range


# 种群初始化
def initialtion(NP, pop, gen_number, up_range, down_range):
    for i in range(0, NP):
        for j in range(0, gen_number):
            pop[i][j] = down_range + random.random() * (up_range - down_range)


# 变异、修正
def mutation(pop, v_pop, NP, F, gen_number, down_range, up_range):
    v_list = []
    for i in range(0, NP):
        xr1 = randomIntX(NP, i)  # x in [0,NP-1]
        xr2 = randomIntX(NP, i, xr1)
        xr3 = randomIntX(NP, i, xr1, xr2)
        # 构造第i个个体对应的变异个体V
        vi = pop[xr1] + F * (pop[xr2] - pop[xr3])
        # 对所有变异个体的分量进行边界约束检查与修正
        for index, vij in enumerate(vi):
            if vij < down_range:
                vi[index] = max(down_range, 2 * down_range - vij)
            elif vij > up_range:
                vi[index] = min(up_range, 2 * up_range - vij)
            else:
                pass
        v_pop[i] = vi


# 交叉
def crossover(pop, v_pop, u_pop, NP, CR, gen_number):
    for i in range(0, NP):
        for j in range(0, gen_number):
            if (random.random() <= CR) or (j == random.randint(0, gen_number - 1)):
                u_pop[i][j] = v_pop[i][j]
            else:
                u_pop[i][j] = pop[i][j]


# 选择
def selection(pop, u_pop, NP):
    for i in range(0, NP):
        if rastrigr_function(u_pop[i]) <= rastrigr_function(pop[i]):
            pop[i] = u_pop[i]
        else:
            pop[i] = pop[i]


start_time = time.time()
pop, NP, F, CR, generation, gen_number, up_range, down_range = initpara()  # 初始化各参数
best_f = []
best_x = []
# 种群初始化
initialtion(NP, pop, gen_number, up_range, down_range)
evaluate_result = evaluate(pop)  # 对种群中的每个个体进行评价
best_f.append(min(evaluate_result))
best_x.append(pop[evaluate_result.index(min(evaluate_result))])

# 进化，搜索
for i in range(generation):
    v_pop = np.zeros((NP, gen_number), dtype=np.float, order='c')  # 初始化变异种群，染色体
    mutation(pop, v_pop, NP, F, gen_number, down_range, up_range)  # 变异、修正
    u_pop = np.zeros((NP, gen_number), dtype=np.float, order='c')  # 初始化试验种群，染色体
    crossover(pop, v_pop, u_pop, NP, CR, gen_number)  # 交叉
    selection(pop, u_pop, NP)  # 选择
    # 评价
    evaluate_result = evaluate(pop)  # 对种群中的每个个体进行评价，产生评价结果
    best_f.append(min(evaluate_result))
    best_x.append(pop[evaluate_result.index(min(evaluate_result))])
    # 实时打印进化代数
    print('\r', ("generation=" + str(i + 1)).ljust(4), end='', flush=True)
end_time = time.time()
print("spend_time=", end_time - start_time)
# 输出
best_ff = min(best_f)
best_xx = best_x[best_f.index(best_ff)]
print('the minimum point is x ')
print(best_xx)
print('the minimum value is y ')
print(best_ff)

# 画图
x_label = np.arange(0, generation + 1, 1)
plt.plot(x_label, best_f, color='blue')
plt.xlabel('iteration')
plt.ylabel('fx')
plt.show()
plt.savefig('./iteration-f:[F_' + str(F) + ' CR_' + str(CR) + '].png')
