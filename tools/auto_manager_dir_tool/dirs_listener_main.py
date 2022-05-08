# -*- coding: utf-8 -*-
"""
Created on Mon Oct 25 19:21:15 2021

@author: WYW
"""

'''
 文件夹监听程序；
 用于监听文件的动作；
'''

# 根据日期设置监听目录
import threading
from queue import *
import os
import time
import deal_function1 as df1,deal_function2 as df2,deal_function3 as df3


'''
配置:
'''
root_dir_path = r"F:\qiyeweixintemp\WXWork\1688850016872018\Cache\File"
local_date = time.strftime("%Y-%m", time.localtime())
local_time = time.strftime("%Y%m%d", time.localtime())

students_path = r"F:\研究生工作文件夹\pythonClassWork\sno_sname.txt"
snames=[]
sno = []
with open(students_path, mode='r',encoding='UTF-8') as f: 
    data = f.read().splitlines()
    sno = data[0].split(",")
    snames = data[1].split(",")
# 初始化学号：姓名
sno_sname = {}
for no in sno:
    sno_sname[no] = snames[sno.index(no)]
# 目的文件夹
dest_dir_path = "F:\研究生工作文件夹\pythonClassWork" + os.path.sep + local_time
# 要监控的源文件夹
soure_dir_path = root_dir_path + os.path.sep + local_date
# 监控前后间隔时间（s）
time_span = 3
# 扫描监控自n分钟前开始
time_before = 20
# 初始化最小时间戳 
min_time_stmp = time.time() - time_before * 60
# 设置大小为200的先进先出的队列
q = Queue(maxsize=200)




# 设置线程类
class myThread (threading.Thread):
    def __init__(self, threadID, name, counter, deal_function, deal_data):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        self.counter = counter
        self.deal_function = deal_function
        self.deal_data = deal_data

    def run(self):
        print("开启线程： " + self.name)
        self.deal_function(self.deal_data)



# 初始化线程锁
threadLock = threading.Lock()
# 初始化线程队列
threads = []
# 初始化参数数据
deal_data = {}
deal_data["queue"] = q
deal_data["lock"] = threadLock
deal_data["soure_dir_path"] = soure_dir_path
deal_data["dest_dir_path"] = dest_dir_path
deal_data["time_span"] = time_span
deal_data["min_time_stmp"] = time_before
deal_data["sno_sname"] = sno_sname

# 创建新线程
thread1 = myThread(1, "dir_lister_monitor", 1, df1.deal_function, deal_data)
thread2 = myThread(2, "file_deal_tool", 2, df2.deal_function, deal_data)
thread3 = myThread(3, "unfinished_job_count", 3, df3.deal_function, deal_data)

# 开启新线程
thread1.start()
thread2.start()
thread3.start()

# 添加线程到线程列表
threads.append(thread1)
threads.append(thread2)
threads.append(thread3)

# 等待所有线程完成
for t in threads:
    t.join()
print("退出主线程")
