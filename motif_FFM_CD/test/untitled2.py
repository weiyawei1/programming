# -*- coding: utf-8 -*-
"""
Created on Tue Jan 25 22:13:53 2022

@author: WYW
"""

# coding: utf-8
# 这个import会先找hello.py，找不到就会找hello.so
import hello  # 导入了hello.so

a = hello.say_hello_to(2,5)
print(a)