# -*- coding: utf-8 -*-
"""
Created on Sat Mar  5 09:38:20 2022

@author: l
"""
import linecache
import os
import numpy as np

# =============================================================================
# 读取指定行
# =============================================================================
Qcs = []
file = r"E:\weiyawei\workspace\motif_FFM_CD\logs\motif_SOSFCD\zhang\zhang_Qc_FCD_nmm_v3_log.txt"
#path = r"E:\weiyawei\workspace\motif_FFM_CD\logs\motif_SOSFCD\karate\Qc_nmm_logs"
#for dirName in os.listdir(path):
#    mem = set(linecache.getline(path + '\\' + dirName , 250)[1:-2].split(' '))
#    Qc = eval(linecache.getline(path + '\\' + dirName , 252).split('=')[1][1:-1])
#    if len(mem) == 3:
#        Qcs.append(Qc)



tmp = []
cs = []
with open(file, mode='r+',encoding='UTF-8') as log_f:
    temp = log_f.readlines()
    for tem in temp:
        if '0.' in tem:
            Qcs.append(eval(tem.split("=")[1][:-1]))
        else:
            cs.append(len(set(tem.split("=")[1][1:-2].split(' '))))
#indexs = []
#for index,c in enumerate(cs):
#    if c != 3:
#       indexs.append(index)
#indexs.reverse()
#
#for index in indexs:
#    Qcs.pop(index)

Qcs.sort()
Qcs_arr = np.asarray(Qcs)
average = Qcs_arr.mean()
print("average=",average)
print("std=",Qcs_arr.std())
print("max=",max(Qcs))
print("min=",min(Qcs))