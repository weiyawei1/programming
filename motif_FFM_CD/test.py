# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 09:55:19 2022

@author: WYW
"""
"""
    FFM_CD_main_v1_NMM
   使用各种优化算法，基于模体的加权网络的社区检测
"""
import os
import numpy as np

file_path = r"E:\weiyawei\workspace\programming\motif_FFM_CD\logs\lesmis\5_Qc_FCD_log.txt"
NWMMlist = []
MNMMlist = []
NMMlist = []
NOMMlist = []
with open(file_path, mode='r') as f:
        datas = f.readlines()
        for line in datas:
            if line.find('NWMM') != -1:
                NWMMlist.append(line.split('=')[1][:-1])
            elif line.find('MNMM') != -1:
                MNMMlist.append(line.split('=')[1][:-1])
            elif line.find('NMM') != -1:
                NMMlist.append(line.split('=')[1][:-1])
            elif line.find('NOMM') != -1:
                NOMMlist.append(line.split('=')[1][:-1])
                
NWMM_mean = np.asarray(NWMMlist, dtype=float).mean()
MNMM_mean = np.asarray(MNMMlist,dtype=float).mean()
NMM_mean = np.array(NMMlist,dtype=float).mean()
NOMM_mean = np.array(NOMMlist, dtype=float).mean()
print("NWMM_mean=",NWMM_mean)
print("MNMM_mean=",MNMM_mean)
print("NMM_mean=",NMM_mean)
print("NOMM_mean=",NOMM_mean)















