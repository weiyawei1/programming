# -*- coding: utf-8 -*-
"""
Created on Sat Oct 23 13:49:10 2021

@author: WYW
"""

import os

# 人员名单文件路径
students_path = r"F:\\研究生工作文件夹\\pythonClassWork\\students.txt"
students_names=[]
with open(students_path, mode='r',encoding='UTF-8') as f:
    students_names = f.read().splitlines()[0].split(",")

# 查找文件的路径
path = r"C:\Users\WYW\Desktop\期中测试\test"
fnames = []
for fname in os.listdir(path):
    name = fname.split(".")[0].split("-")[1]
    fnames.append(name)
        
# 找出差集
unfinish_job_status = list(set(students_names).difference(set(fnames)))
print("还有{}人没交作业".format(len(unfinish_job_status)))
print(unfinish_job_status)