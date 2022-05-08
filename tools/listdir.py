# -*- coding: utf-8 -*-
"""
Created on Mon Oct 25 13:55:57 2021

@author: WYW
"""

'''
 文件夹监听程序；
 用于监听文件的动作；
'''
import os, time
from Queue import Queue

# 根据日期设置监听目录
root_dir_path = "F:/qiyeweixintemp/WXWork/1688850016872018/Cache/File"
local_date = time.strftime("%Y-%m",time.localtime())
# 要监控的文件夹
dir_to_watch = root_dir_path + os.path.sep + local_date
# 监控前后间隔时间（s）
time_span = 5
# 扫描监控自n分钟前开始
n = 20




before = dict ([(f, None) for f in os.listdir (dir_to_watch)])
while (True):
    time.sleep (time_span)
    after = dict ([(f, None) for f in os.listdir (dir_to_watch)])
    added = [f for f in after if not f in before]
    removed = [f for f in before if not f in after]
    if added: 
       print ("Added: ", ", ".join (added))
    if removed: 
       print ("Removed: ", ", ".join (removed))
    before = after