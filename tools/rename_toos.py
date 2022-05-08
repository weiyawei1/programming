# -*- coding: utf-8 -*-
"""
Created on Tue Oct 26 14:08:58 2021

@author: WYW
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Oct 23 13:49:10 2021

@author: WYW
"""

import os

# 查找文件的路径
path = r"F:\研究生工作文件夹\pythonClassWork\20211023_第一次上机实验测试"
sno_fname = {}
for fname in os.listdir(path):
    sno = fname.split(".")[0].replace(" ","")[0:11]
    sname = fname.split(".")[0].replace(" ","")[11:]
    sno_fname[sno] = sname
infolines = []
keys = str(list(sno_fname.keys()))
values = str(list(sno_fname.values()))

wfpath = r"F:\研究生工作文件夹\workspace\tools\sno_sname.txt"
with open(wfpath, mode='w',encoding='UTF-8') as f:
    f.writelines(keys + "\n" + values)
    