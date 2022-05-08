# -*- coding: utf-8 -*-
"""
Created on Sat Feb 26 13:34:55 2022

@author: l
"""

import time

import os,shutil

'''
    复制文件
    srcfile: 源文件路径(全路径)
    dstfile: 目的文件路径（全路径）
'''   
def copyfile(srcfile,dstfile):
    if not os.path.isfile(srcfile):
        print ("%s not exist!"%(srcfile))
    else:
        fpath,fname=os.path.split(dstfile)    #分离文件名和路径
        if not os.path.exists(fpath):
            os.makedirs(fpath)                #创建路径
        shutil.copyfile(srcfile,dstfile)      #复制文件
        print ("copy %s -> %s"%( srcfile,dstfile))
        
# 查找文件的路径
path = r"C:\Users\l\Desktop\tmp"
rePath = r"E:\weiyawei\tmp\dayToDayWork\对比算法\有向真实网络\test_result\CMLFR"
for dirName in os.listdir(path):
    for dirName1 in os.listdir(path + "\\" + dirName):
        for fname in os.listdir(path + "\\" + dirName + "\\" + dirName1):
            if fname == 'tp':
                copyfile(path + "\\" + dirName + "\\" + dirName1 + "\\" + fname, rePath + "\\" + dirName1.split('.')[0] + "\\" + fname + dirName + ".txt")