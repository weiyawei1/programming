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













