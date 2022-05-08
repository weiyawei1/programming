# -*- coding: utf-8 -*-
"""
Created on Mon Oct 25 20:16:43 2021

@author: WYW
"""

'''
    处理函数3
'''
import os
import time

def deal_function(deal_data):
    while 1:
		# 5分钟统计一次
        time.sleep(deal_data["time_span"] * 100)
        
        # 人员名单文件路径
        watch_dir_path = deal_data["dest_dir_path"]
        snames = deal_data["sno_sname"].values()
        if not os.path.exists(watch_dir_path):
            os.makedirs(watch_dir_path)                #创建路径
        # 查找文件的路径
        fnames = []
        for fname in os.listdir(watch_dir_path):
            name = fname.split(".")[0].replace(" ","")[11:]
            fnames.append(name)
        # 找出差集
        unfinish_job_count = len(list(set(snames).difference(set(fnames))))
        if unfinish_job_count == 0 :
            print("\n{:*^60}\n".format("作业已收齐！"))
            break
        else :
            print("\n***********还有{}人没交作业!***********\n".format(unfinish_job_count))        