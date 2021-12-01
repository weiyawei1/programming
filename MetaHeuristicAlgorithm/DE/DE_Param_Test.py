# -*- coding: utf-8 -*-
"""
Created on Wed Nov 24 10:35:54 2021

@author: WYW
"""
from multiprocessing import Process, Lock, Queue
import numpy as np
import time
import src.DE as de


# =============================================================================
#     用于测试 DE 中 F、CR 参数对 DE算法的影响
# =============================================================================
# =============================================================================
# # 设置线程池
# =============================================================================
class MyProcess(Process):
    def __init__(self, name):
        super().__init__()
        self.name = name

    def run(self):
        print("{:*^20} is running".format(self.name))
        self.function(self.params, self.queue)
        print("{:*^20} is running end".format(self.name))

    def setParam(self, function, params, queue):
        self.function = function
        self.params = params
        self.queue = queue


if __name__ == '__main__':
    # 初始化进程锁
    process_num = 10
    lock = Lock()  # 初始化进程锁
    # 创建队列
    queue = Queue(process_num)

    clc = 19
    F_list = []
    CR_list = []
    init_v = 0
    for i in range(clc):
        init_v += 0.05
        F_list.append(init_v)
        CR_list.append(init_v)

    F_CR_fx = np.zeros((clc ** 2, 3), dtype=np.float, order='c')  # 用于绘制最优个体三维图
    F_CR_c = np.zeros((clc ** 2, 3), dtype=np.float, order='c')  # 用于绘制收敛三维图

    index_clc = 0
    start_time = time.time()
    for f in F_list:
        for cr in CR_list:
            ### 使用多线程，运行十次取平均值
            # 创建进程池
            pools = [MyProcess("Process" + str(index + 1)) for index in range(process_num)]  # 创建进程池
            fx_c_everage = [0, 0]  # fx，c
            params = [f, cr]
            # 多进程执行    
            for po in pools:
                po.setParam(de.de_function, params, queue)
                po.start()
            for p in pools:
                p.join()

            print("*****************第{}次运行完成*****************".format(index_clc + 1))
            while (not queue.empty()):
                q = queue.get()
                fx_c_everage[0] += q[0]
                fx_c_everage[1] += q[1]
            fx_c_everage[0] /= 10
            fx_c_everage[1] //= 10
            F_CR_fx[index_clc] = [f, cr, fx_c_everage[0]]
            F_CR_c[index_clc] = [f, cr, fx_c_everage[1]]
            index_clc += 1
            # pool.close() # 关闭进程池，不再接收新的进程
    # pool.join() # 主进程阻塞等待子进程的退出
    end_time = time.time()
    print("\n {:*^40}".format("USEDTIME=" + str(end_time - start_time)))
    # =============================================================================
    # 将数据保存到本地磁盘         
    # =============================================================================
    # 创建数据存储文件路径
    # 将数据保存到磁盘，第一行为fx，第二行为c
    fo = open("result_data.txt", "w")
    for fx in F_CR_fx:
        fo.write(str(fx[2]) + ",")
    fo.write('\n')
    for c in F_CR_c:
        fo.write(str(c[2]) + ",")
    fo.close()

    print("{:*^40}".format("已完成！"))
    ### 绘制三维图形
