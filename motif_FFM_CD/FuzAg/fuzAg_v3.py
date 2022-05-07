import networkx as nx
import random
import numpy as np
import igraph as ig

# 各模块函数
import motif_network_construct as net_stru_func
import FuzAg_function as FuzAg_func


# =============================================================================
# 网络信息
# network
# =============================================================================
path = r"F:\研究生工作文件夹\data\经典数据集"
beican_9_network = path + r'\9_beican.txt'
karate_network = path + r'\karate.txt'
dolphins_network = path + r'\dolphins.txt'
football_network = path + r'\football.txt'
polbooks_network = path + r'\polbooks.txt'

# 选择网络
network = karate_network
G1 = nx.read_edgelist(network)
G1 = G1.to_undirected()

# 获取网络数据中的边列表，并根据其使用igraph创建网络
Gi=ig.Graph.Read_Edgelist(network)
Gi=Gi.subgraph(map(int,G1.nodes()))          
Gi=Gi.as_undirected()

edge_all = Gi.get_edgelist()

# =============================================================================
# 各参数设置
# =============================================================================
n=G1.number_of_nodes()
NP = 100
c = 9   #社区的真实划分数
Gen = n  #进化代数
threshold_value = 0.25  #阈值
M = 1  #模体选择【1:M1,2:M2,3:M3,4:M4,51:M5,6:M6,7:M7,8:M8】 

# =============================================================================
# 构建基于模体M1的加权网络
# =============================================================================
G = net_stru_func.construct_weighted_network(Gi,n,M)

# 获得无权网络邻接矩阵
G2 = nx.Graph() 
G2.add_nodes_from([i for i in range(n)])
G2.add_edges_from(edge_all)
adj= nx.adjacency_matrix(G2)
adj=adj.todense() 

# 构建基于模体的加权网络邻接矩阵motif_adj
motif_adj = nx.adjacency_matrix(G)
motif_adj = motif_adj.todense() 
# 获得模体邻接矩阵
# motif_adj = net_stru_func.get_motifadd_adj(G, edge_all, M)

# # 设置全局变量
pop_best_history = np.zeros((c,n,Gen)) # 用于保存历史最优的个体记录
best_in_history_Qg = [] # 用于保存历史最优Qg值

# 初始化NMi
nmilist=np.zeros((1,Gen),dtype = float, order = 'C')

# =============================================================================
# ##################################Test#######################################
# FuzAg: 基于自隶属度搜索概念的模糊聚类社区检测
# n: 网络节点数
# itermax: 最大迭代次数
# phi: 获得任何社区成员资格的节点阈值
# motif_adj: 加权网络邻接矩阵
# return: U, K 隶属度矩阵，社区划分数
# =============================================================================
RUbest = {}
Qs = []
for gen in range(Gen):
    print("gen=",gen)
    RU, RK = FuzAg_func.FuzAg(n, 10, threshold_value, adj,gen)
    
    # 设置社区 (离散划分)
    membership = [0]*n
    keyList = RU.keys()
    for i in range(n):
        i_ships = []
        for key in keyList:
            i_ships.append(RU[key][i])
        c = i_ships.index(max(i_ships))
        membership[i] = c
        
    Q_modularity = ig.GraphBase.modularity(Gi, membership) 
    Qs.append(Q_modularity)
    
print ('Qbest  is',max(Qs)) 



# threshold value for acquiring membership of a particular community
phi=0.25

# maximum number of iterations
itermax=100

# Total Number of Nodes
N=50

# Current Number of Communities
K=1

# adjacency matrix Wij, weight of edge connecting node i and j
W=np.zeros((N,N))
# src=np.random.choice(N,5,replace=False)
# dest=np.random.choice(N,5,replace=False)
for i in range(N):
	dest=np.random.choice(N,5)
	for j in dest:
		if i!=j:
			W[i][j]=round(random.uniform(0,1),3)
			W[j][i]=W[i][j];
            
# Partitioning Matrix (Python Dictionary)
U={}

U[N+1]=np.zeros(N)

# anchorList
anchorList=set({})


# Returns a random vertex as the first anchor
def randomFirstAnchor(numberOfNodes):
	randomVertex=random.randrange(0,numberOfNodes,1)
	# anchorList.add(randomVertex)
	addAnchor(randomVertex)

# 返回一个节点的邻接节点
def adjacentNodes(node):
	adjacentNodesList=[]
	# print(node)
	for i in range(0,len(W[node])):
		if(W[node][i]>0):
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


# 返回一个社区中重要节点列表
def vitalNodes(anchorNode):
	vitalNodesList=adjacentNodes(anchorNode)
	# print("Adjacent Nodes of AnchorNode",anchorNode,": ",vitalNodesList)
	for i in range(0,N):
		if i not in vitalNodesList and i!=anchorNode:
			if(U[anchorNode][i]>=sumOfMembershipWithRemainingCommunities(i,anchorNode)):
				# vitalNodesList.append(i)
				np.concatenate([vitalNodesList,[i]])
	return np.asarray(vitalNodesList)


# return list of nodes common between vitalnodes and adjacent nodes of a given node
def commonVitalAdjacentNodes(givenNode,anchorNode):
	neighbours=adjacentNodes(givenNode)
	currentVitalNodes=vitalNodes(anchorNode)
	return np.intersect1d(neighbours,currentVitalNodes)



# Calculate all membership values of nodes with respect to the current anchors
def calculateMembership(node,community):
	numr=0
	for i in commonVitalAdjacentNodes(node,community):
		numr+=(W[i][node])
	denmr=0
	# print(node," -> ",adjacentNodes(node))
	for j in adjacentNodes(node):
		denmr+=(W[j][node])
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
        numr+=(W[i][node])
    denmr=0
    for j in adjacentNodes(node):
        denmr+=(W[j][node])
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
			anchorList.add(i)
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
		# print("Identified redundant anchor-> ",anchorNode)
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
		# print("Identified False Anchor-> ",anchorNode)
		return True

# remove Anchor
def removeAnchor(anchor):
	anchorList.remove(anchor)
	del U[anchor]

# add Anchor
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


# main working Loop
randomFirstAnchor(N)
rflag=1
aflag=1
iterVal=0
while(1):
	print("ITERATION:",iterVal)
	iterVal+=1
	trflag=0
	taflag=0
	for i in range(N):
		U[N+1][i]=selfMembership(i)
	for i in range(N):
		for j in anchorList:
			U[j][i]=calculateMembership(i,j)
	normalize()    #规范化隶属度矩阵
    # 初始化锚点移除集合
	anchorsToBeRemoved=[]
	for i in anchorList:
		if identifyFalseAnchor(i) or identifyRedundantAnchor(i):
			# removeAnchor(i)
			anchorsToBeRemoved.append(i)
			trflag=1
	for i in anchorsToBeRemoved:
		# print("Removing Anchor-> ",i)
		removeAnchor(i)
	# print(U)
	if(trflag==1):
		for i in anchorList:
			tmpList=[]
			for j in range(N):
				tmpList.append(calculateMembership(j,i))
			U[i]=np.array(tmpList)
		# for i in range(N):
		# 	tmpList=[]
		# 	for j in anchorList:
		# 		tmpList.append(calculateMembership(i,j))
		# 	U[j]=np.array(tmpList)
		for i in range(N):
			U[N+1][i]=selfMembership(i)
		for i in range(N):
			for j in anchorList:
				U[j][i]=calculateMembership(i,j)
		normalize()
		# print(U)
		rflag=1
	else:
		rflag=0

	listOfNewAnchors=newAnchors()
	print("Adding Anchors->",listOfNewAnchors)
	if(len(listOfNewAnchors)>0):
		aflag=1
	else:
		aflag=0
	for i in listOfNewAnchors:
		addAnchor(i)
	if(iterVal>=itermax):
		break
	if(aflag==0 and rflag==0):
		break
