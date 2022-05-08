# -*- coding: utf-8 -*-
"""
Created on Fri Oct 29 13:28:37 2021

@author: WYW
"""

data_path = "F:\研究生工作文件夹\workspace\C++\\mdf1.txt"
write_path = "F:\研究生工作文件夹\workspace\C++\\mdf02.txt"
dataList=[]
data=[]
with open(data_path, mode='r',encoding='UTF-8') as f:
    dataList = f.read().split("\n")
index =0
for line in dataList:
    le = line[1:-2] + "\n"
    data.append(le)
    index += 1
    if index >20:
        break
with open(write_path, mode='w',encoding='UTF-8') as f:
    f.writelines(data)