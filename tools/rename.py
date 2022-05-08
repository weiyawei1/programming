# -*- coding: utf-8 -*-
"""
Created on Thu Dec  2 10:29:50 2021

@author: WYW
"""
import os
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

students_path = r"F:\研究生工作文件夹\pythonClassWork\sno_sname.txt"
snames=[]
sno = []
class_no = []
with open(students_path, mode='r',encoding='UTF-8') as f: 
    data = f.read().splitlines()
    sno = data[0].split(",")
    snames = data[1].split(",")
    class_no = data[2].split(",")
# 初始化学号：姓名
sno_sname = {}
for no in sno:
    sno_sname[no] = snames[sno.index(no)]

path = r"C:\Users\WYW\Desktop\期中测试"
repath = r"C:\Users\WYW\Desktop\期中测试\test"
fnames = os.listdir(path)
for index,name in enumerate(snames):
    for fname in fnames:
        if name in fname:
            fn = sno[index] + "-" + name + "-" + class_no[index] + "." + fname.split('.')[1]
            copyfile(path + "\\" + fname, repath + "\\" + fn)
           
            