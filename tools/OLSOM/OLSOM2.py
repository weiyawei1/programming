# -*- coding: utf-8 -*-
"""
Created on Tue Feb  8 14:33:06 2022

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

# 查找文件的路径
path = r"E:\weiyawei\dayToDayWork\对比算法\有向真实网络\test_result\test"
for fname in os.listdir(path):
    if not fname.endswith('.txt') :
        copyfile(path + "\\" + fname, path + "\\" + fname + ".txt")
        os.remove(path + "\\" + fname)

datas = []
for fname in os.listdir(path):
    with open(path + "\\" + fname, mode='r',encoding='UTF-8') as f:
        data = f.read().splitlines()
        for line in data:
            if line.startswith('#'):
                data.remove(line)
        datas.append(data)
        
# 设置模糊节点
bs_datas = []
for fname in os.listdir(path):
    with open(path + "\\" + fname, mode='r',encoding='UTF-8') as f:
        bs = []
        data = f.read().splitlines()
        for line in data:
            if not line.startswith('#'):
                data.remove(line)
        for d in data:
            bs.append(float(d.split(':')[-1].strip()))
        bs_datas.append(bs)
                     

nums = []
repeats = []
for index1, data in enumerate(datas):
    tmp_data = []
    for line in data:
        numbers = line.split(' ')
        numbers.remove('')
        tmp_data += numbers
    size = max(map(int,tmp_data))
    num = [0]*size
    repeat = []
    for index2, line in enumerate(data):
        numbers = line.split(' ')
        numbers.remove('')
        for number in numbers:
            if num[int(number)-1] != 0:
                repeat.append(number)
                if  bs_datas[index1][index2] < bs_datas[index1][int(num[int(number)-1].split(',')[0])]:
                   num[int(number)-1] = str(index2) + ','
                   print("t=",(index1,number,int(num[int(number)-1].split(',')[0])))
                else:
                     print("t=",(index1,number,index2))
                     continue
            else:
                num[int(number)-1] = str(index2) + ','  
    repeats.append(repeat)
    num[-1] = num[-1].split(',')[0]
    nums.append(num)




result_path = r"E:\weiyawei\dayToDayWork\对比算法\有向真实网络\test_result"
file_name = "test.txt"
with open(result_path + "\\" + file_name, mode='w',encoding='UTF -8') as f:
    for num in nums:
        f.writelines(num)
        f.writelines('\n')
