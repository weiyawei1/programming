# -*- coding: utf-8 -*-
"""
Created on Mon Oct 25 20:12:18 2021

@author: WYW
"""
'''
    处理函数2（消费者线程函数）
    按照先进先出原则取出队列中的文件名
    转移并重命名文件
'''
import time

import os,shutil

def deal_function(deal_data):
    while 1:
        if deal_data["queue"].empty() :
            time.sleep(deal_data["time_span"])
        else :
            # 初始化文件名列表
            fnames = []
            # 获取锁，用于线程同步
            threadLock = deal_data["lock"]
            threadLock.acquire()
            while not deal_data["queue"].empty() :
                fnames.append(deal_data["queue"].get())
            # 释放锁
            threadLock.release()
            # 转移并重命名文件
            soure_dir_path = deal_data["soure_dir_path"]
            dest_dir_path = deal_data["dest_dir_path"]
            for fname in fnames:
                fileName = rename(fname,deal_data["sno_sname"])
                srcfile = soure_dir_path + os.path.sep + fname
                dstfile = dest_dir_path + os.path.sep + fileName
                movefile(srcfile, dstfile)

        
'''
    移动文件
    srcfile: 源文件路径(全路径)
    dstfile: 目的文件路径（全路径）
'''        
def movefile(srcfile,dstfile):
    if not os.path.isfile(srcfile):
        print ("%s not exist!"%(srcfile))
    else:
        fpath,fname=os.path.split(dstfile)    #分离文件名和路径
        if not os.path.exists(fpath):
            os.makedirs(fpath)                #创建路径
        shutil.move(srcfile,dstfile)          #移动文件
        print("{} 提交成功！".format(fname.split(".")[0]))
#        print ("move %s -> %s"%( srcfile,dstfile))
    

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
'''
    对文件重新命名
    指定命名规则
    fname:文件名
    return:返回新文件名
'''
def rename(fname,sno_sname = {}):
    index_str = fname.find("20211105")
    end_index = index_str + 11
    sno = fname[ index_str : end_index ]
    sname = sno_sname[sno]
    postfix = fname.split(".")[1]
    fileName = sno + sname + "." + postfix
    return fileName

