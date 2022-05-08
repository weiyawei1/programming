# -*- coding: utf-8 -*-
"""
Created on Wed Nov  3 16:43:30 2021

@author: WYW
"""
class Solution:
    def isPerfectSquare(self, num: int):
        # 使用1~num 二分查找法
        if findX(1,num,num):
            return True
        else :
            return False
def findX(min,max,Y):
        middle = (min + max) // 2
        if middle == min:
            return False
        Y_temp = middle ** 2
        if Y_temp == Y:
            return True
        elif Y_temp < Y:
            return findX(middle,max,Y)
        else:
            return findX(min,middle,Y)
        
sol = Solution()
a=sol.isPerfectSquare(1)
print(a)