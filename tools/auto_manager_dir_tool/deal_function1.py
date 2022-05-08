# -*- coding: utf-8 -*-
"""
Created on Mon Oct 25 20:11:38 2021

@author: WYW
"""
'''
处理函数1（生产者线程函数）
用于监控源文件夹
筛选出待处理的数据
将筛选出的文件名放入队列
'''
import time
import os

def deal_function(deal_data):
    while 1:
        time.sleep(deal_data["time_span"])
        # 初始化本轮筛选列表
        fnames = []
        # 初始化本轮筛选的最小时间戳
        min_time_stmp =  deal_data["min_time_stmp"]
        fnames = os.listdir(deal_data["soure_dir_path"])
        # 本轮初始化fnames结束，更新最小时间戳
        deal_data["min_time_stmp"] = time.time()
        
        for fname in fnames:
            fpath = deal_data["soure_dir_path"] + os.path.sep + fname 
            fCreateTime = os.path.getctime(fpath)
            if fCreateTime >= min_time_stmp:
                str_index = fname.find("20211105") 
                if str_index != -1 :
                    # 获取锁，用于线程同步
                    threadLock = deal_data["lock"]
                    threadLock.acquire()
                    deal_data["queue"].put(fname)
                    # 释放锁
                    threadLock.release()
        
                
            
        

