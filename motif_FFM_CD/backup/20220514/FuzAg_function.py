# -*- coding: utf-8 -*-
"""
Created on Wed Jan 26 19:52:17 2022

@author: WYW
"""
"""
    FuzAg_function: FuzAg 各功能函数
"""

import numpy as np
import random
from random import shuffle

# =============================================================================
# 初始化全局变量
# threshold: 阈值
# n: 节点数
# motif_adj： 加权网络的邻接矩阵
# u： 各分区的隶属度矩阵
# anchors：锚点列表
# =============================================================================
def init(threshold,n,motif_adj,u,anchors):
    # 初始化全局变量
    global N 
    global W
    global U
    global anchorList
    global phi
    
    N = n
    W = motif_adj
    U = u
    anchorList = anchors
    phi = threshold

# =============================================================================
# 随机定义第一个锚点
# =============================================================================
def randomFirstAnchor(numberOfNodes):
	randomVertex=random.randrange(0,numberOfNodes,1)
	# anchorList.add(randomVertex)
	addAnchor(randomVertex)

# 返回一个节点的邻接节点
def adjacentNodes(node):
    adjacentNodesList=[]
    # print(node)
    for i in range(0,N):
        if(W[node,i]>0):
            adjacentNodesList.append(i) 
    return np.asarray(adjacentNodesList)

# 返回与其余锚点的隶属度值总和1
def sumOfMembershipWithRemainingCommunities(currentNode,anchorNode):
	# print("Calculating Sum of Membership with remaining communities excpet Anchor ",anchorNode)
	val=0
	# print(U)
	for i in U:
		if(i!=anchorNode and i!=N+1):
			# print("AnchorNode-> ",i," CurrentNode-> ",currentNode)
			val+=U[i][currentNode]
	return val


# 返回对于锚点h的重要节点列表
def vitalNodes(anchorNode):
    vitalNodesList = []
    adjacentNodeList=adjacentNodes(anchorNode)
	# print("Adjacent Nodes of AnchorNode",anchorNode,": ",vitalNodesList)
    for i in range(0,N):
        if i in adjacentNodeList and U[anchorNode][i] > 0:
            vitalNodesList.append(i)
        # if i in adjacentNodeList:
        #     vitalNodesList.append(i)
        if i not in adjacentNodeList and i!=anchorNode:
            if(U[anchorNode][i]>=sumOfMembershipWithRemainingCommunities(i,anchorNode)):
                vitalNodesList.append(i)
    vitalNodesList.append(anchorNode)
    return np.asarray(vitalNodesList)


# return list of nodes common between vitalnodes and adjacent nodes of a given node
def commonVitalAdjacentNodes(givenNode,anchorNode):
	neighbours=adjacentNodes(givenNode)
	currentVitalNodes=vitalNodes(anchorNode)
	return np.intersect1d(neighbours,currentVitalNodes)



# Calculate all membership values of nodes with respect to the current anchors
def calculateMembership(node,community):
    numr=0
    commonNodes = commonVitalAdjacentNodes(node,community)
    for i in commonNodes:
        numr+=(W[i,node])
    denmr=0
    # print(node," -> ",adjacentNodes(node))
    adjcentNodeList = adjacentNodes(node)
    for j in adjcentNodeList:
        denmr+=(W[j,node])
    if(denmr==0):
        return 0
    return numr/denmr

# 返回所有独立节点的列表
def independentNodes():
	listOfAllNodes=[i for i in range(0,N)]
	npArrayOfAllNodes=np.array(listOfAllNodes)
	npArrayOfAllVitalNodes=allVitalNodes()
	ret=np.setdiff1d(npArrayOfAllNodes,np.intersect1d(npArrayOfAllNodes,npArrayOfAllVitalNodes))
	# print("independentNodes",ret)
	return ret


# 计算节点node的自隶属度
def selfMembership(node):
    numr=0
    allIndependentNodes=independentNodes()
    neighbours=adjacentNodes(node)
    commonNodes=np.intersect1d(allIndependentNodes,neighbours)
    # print("####################")
    # print("node={}".format(node))
    # print("independentNodes={}".format(allIndependentNodes))
    # print("neighbours={}".format(neighbours))
    # print("commonNodes={}".format(commonNodes))
    # print("####################")
    for i in commonNodes:
        numr+=(W[i,node])
    denmr=0
    for j in neighbours:
        denmr+=(W[j,node])
    return numr/denmr

# 鉴定新锚点
def newAnchors():
    listOfNewAnchors=[]
    oldAnchorList=[]
    for i in anchorList:
        oldAnchorList.append(i)
    # print("oldAnchorList-> ",oldAnchorList)
    for i in range(N):
        sumOfMembershipWithAllCommunties=0
        for j in oldAnchorList:
            # print("oldAnchor-> ",j)
            sumOfMembershipWithAllCommunties+=U[j][i]
        # print("Node-> ",i," Self membership-> ",U[N+1][i],"Membership at others",sumOfMembershipWithAllCommunties)
        if(U[N+1][i]>=sumOfMembershipWithAllCommunties):
            listOfNewAnchors.append(i)
            # anchorList.add(i)
    return np.asarray(listOfNewAnchors)

# 任意锚点的所有重要节点列表
def allVitalNodes():
    setOfAllVitalNodes=set({})
    for i in anchorList:
        for j in vitalNodes(i):
                setOfAllVitalNodes.add(j)
    return np.asarray(list(setOfAllVitalNodes))

# set of all vital nodes of all anchors except this anchors
def allVitalNodesMinusThisAnchor(anchor):
	setOfAllVitalNodesExceptThisAnchor=set({})
	# print("anchor",anchor)
	for i in anchorList:
		if i!=anchor:
			for j in vitalNodes(i):
				setOfAllVitalNodesExceptThisAnchor.add(j)
	return list(setOfAllVitalNodesExceptThisAnchor)


# identify redundant anchor
def identifyRedundantAnchor(anchorNode):
	listOfVitalNodes=vitalNodes(anchorNode)
	flag=0
	setOfAllVitalNodes=np.array(allVitalNodesMinusThisAnchor(anchorNode))
	# print("All vital Nodes except ",anchorNode, " Node",setOfAllVitalNodes)
	# print("List of All Vital Nodes of ",anchorNode," -> ",listOfVitalNodes)
	intersection=np.intersect1d(listOfVitalNodes,setOfAllVitalNodes)
	if(len(intersection)<len(listOfVitalNodes)):
		# not redundant anchor
		return False
	else:
		# redundant anchor identified
# 		print("Identified redundant anchor-> ",anchorNode)
		return True

# identify false anchor
def identifyFalseAnchor(anchorNode):
	flag=0
	for i in U[anchorNode]:
		if i>=phi:
			flag=1
			break
	if(flag==1):
		# not False Anchor
		return False
	else:
		# False Anchor Identified
# 		print("Identified False Anchor-> ",anchorNode)
		return True

# remove Anchor
def removeAnchor(anchor):
	anchorList.remove(anchor)
	del U[anchor]

# 添加锚点
def addAnchor(anchor):
	anchorList.add(anchor)
	U[anchor]=np.zeros(N)
	# membershipOfNodes=[]
	# for i in range(N):
	# 	membershipOfNodes.append(calculateMembership(i,anchor))
	# U[anchor]=np.array(membershipOfNodes)

# 规范化隶属度矩阵
def normalize():
	for i in range(N):
		tsum=0
		for j in anchorList:
			tsum+=U[j][i]
		if tsum==0:
			continue
		for j in anchorList:
			U[j][i]=U[j][i]/tsum

