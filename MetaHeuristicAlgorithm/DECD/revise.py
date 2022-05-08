# -*- coding: utf-8 -*-
"""
Created on Tue Nov  2 21:44:46 2021

@author: WYW
"""
from numpy import random

"""
修正操作
    pop_x: 用于修正的种群【变异种群、交叉后的种群】
    n:  种群中个体的分量数【网络的节点数】
    NP: 种群中的个体数[网络划分的社区数]
    G:  网络G
    threshold_value:  阈值【用于控制修正程度】
"""

def revise_operation(pop_x, n, NP, G, threshold_value):
    for i in list(range(NP)):
        # 所有节点标号
        all_node_index = list(range(n))
        # 确定选择节点标号的数目
        get_num = random.randint(1, n)
        # 在1-n号节点中随机选择get_num个不一样的节点标号
        use_node_index = []
        # 在个体i中随机选择get_num个不同的节点，保存在use_node_index
        for cu_i in range(get_num):
            # 在1-n节点中随机选取一个节点序号
            cur_rand_index = random.randint(0, len(all_node_index) - 1)
            # 添加至use_node_index
            use_node_index.append(all_node_index[cur_rand_index])
            # 将all_node_index中对应的元素删除
            all_node_index.remove(all_node_index[cur_rand_index])

        # 对use_node_index中的节点进行纠错
        for rand_i in range(get_num):
            # 针对use_node_index中的每一个节点进行社区标号纠错
            node = use_node_index[rand_i]
            # 确定节点node的所有邻域个体，包括其自身，如node=16，那么all_adj_node=【16,33,34】
            neigh_node = G.neighbors(node)

            # 构建节点node自身及邻域集合列表
            # 例如：[2, 0, 1, 3, 7, 8, 9, 13, 27, 28, 32]
            all_adj_node = []
            all_adj_node.append(node)
            all_adj_node.extend(neigh_node)#修改过的语句，原句在上面

            # node及其邻域节点所属的社区编号
            node_comm = pop_x[i][node]
            # node邻域节点所属的社区编号
            node_neigh_comm = []
            for k in range(len(neigh_node)):
                node_neigh_comm.append(pop_x[i][neigh_node[k]])
            # 计算CV
            # 节点node与邻域个体属于不同社区的数目
            different_comm_number = 0
            for k in range(len(node_neigh_comm)):
                if node_comm != node_neigh_comm[k]:
                    different_comm_number += 1
                else:  # 新增部分
                    different_comm_number += 0
            # 节点node的度
            degree_node = len(node_neigh_comm)
            # 节点node的CV值
            CV_node = float(different_comm_number) / degree_node
            # 判断CV是否大于阈值
            # 若是，则说明节点node与邻域节点不在同一社区的概率较大
            # 节点社区标号错误,选择邻域节点中出现次数最多的社区标号
            if CV_node > threshold_value:
                # 邻域节点所属社区标号
                temp_comm = node_neigh_comm
                # 邻域节点所归属最多的社区数目
                max_num = 0
                # 邻域节点所归属最多的社区标号
                max_comm_id = 0
                # 找到node_neigh_comm中邻域节点归属最多的社区
                while len(temp_comm) > 0:
                    # 选取第一个邻域节点所属社区cur_comm
                    cur_comm = temp_comm[0]
                    # 归属cur_comm的所有邻域节点的序号集合
                    all_node = []
                    for k in range(len(temp_comm)):
                        if temp_comm[k] == cur_comm:
                            all_node.append(k)
                    # 归属cur_comm的所有邻域节点数目
                    cur_num = len(all_node)
                    # 比较cur_num与max_num，更新max_num和max_comm_id
                    if cur_num > max_num:
                        # 属于当前社区cur_comm的邻居节点>已知属于同一社区的最多邻居节点数目max_num
                        max_num = cur_num
                        max_comm_id = cur_comm
                    elif cur_num == max_num:
                        # 以50%的概率决定是否更改max_num和max_comm_id
                        if random.rand(1, 1) > 0.5:
                            max_num = cur_num
                            max_comm_id = cur_comm
                    # 删除temp_comm中归属于cur_comm的邻域节点\
                    del temp_comm[0] #删除列表中的元素 

                    # 将pop中node的社区标号更新为max_comm_id
                pop_x[i][node] = max_comm_id  # 纠正该节点的社团编号
    # 返回纠错后的新种群
    return pop_x