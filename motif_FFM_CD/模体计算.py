# -*- coding: utf-8 -*-
"""
Created on Thu May  5 20:24:28 2022

@author: wyw
"""
import find_motifs as fm
import igraph as ig
import networkx as nx
import time

# =============================================================================
# Test
# ===========================================================================
path = r"data/经典数据集"
beican_9_network = path + r'/9_beican.txt'
zhang_network = path + r'/zhang.txt'
karate_network = path + r'/karate.txt'
dolphins_network = path + r'/dolphins.txt'
football_network = path + r'/football.txt'
polbooks_network = path + r'/polbooks.txt'
## 功能网络
func_path = r"data/功能网络"
brain47_network = func_path + r'/brain47.txt'

# 选择网络
network = football_network
# 构建网络
G = nx.read_edgelist(network,create_using=nx.Graph())
G = G.to_undirected()

# 获取网络数据中的边列表，并根据其使用igraph创建网络
Gi=ig.Graph.Read_Edgelist(network)
Gi=Gi.subgraph(map(int,G.nodes()))          
Gi=Gi.as_undirected()
n=G.number_of_nodes()
edge_all = Gi.get_edgelist()

# 获得无权网络邻接矩阵
G2 = nx.Graph() 
G2.add_nodes_from([i for i in range(n)])
G2.add_edges_from(edge_all)
adj= nx.adjacency_matrix(G2)
adj=adj.todense() 

# 创建需要计算的模体
g = nx.Graph()
g.add_nodes_from([1, 2, 3])
g.add_edges_from([(1, 2), (2, 3), (3, 1)])  # 连通
nx.draw(g,with_labels=True)
# 计算模体在网络中的数量
number3 = fm.total_motif_num(G, g, directed=False, weighted=False)#总模体数
print('总模体数量:',number3)
#模体覆盖率
rate1=fm.node_coverage_rate_of_motif(G,g,directed=False, weighted=False)
'''
G为网络，g为模体，directed是否为有向网络，weighted是否为加权网络，
'''
print('模体的节点覆盖率', rate1)

rate2=fm.edge_coverage_rate_of_motif(G,g,directed=False, weighted=False)
'''
G为网络，g为模体，directed是否为有向网络，weighted是否为加权网络，
'''
print('模体的边覆盖率', rate2)

#节点或边参与构成模体g的数量
node_list = list(G.nodes)
edge_list = list(G.edges)

start = time.process_time()
for i in node_list:
    number4=fm.node_in_motif(G, g,i, directed=False, weighted=False)
#    print("节点参与构成模体的数量：",number4)
for edge in edge_list:
    number5=fm.edge_in_motif(G, g, edge, directed=False, weighted=False)
#    print("边参与构成模体的数量：",number5)

#节点参与构成模体的集合
#for i in node_list:
#    Node_set,edge_set=fm.node_in_motif_list(G, g, i, directed=False, weighted=False)
##    print("节点参与构成模体：点集合：",Node_set)
##    print("节点参与构成模体：边集合：",edge_set)
for edge in edge_list:
    Node_set1,edge_set1=fm.edge_in_motif_list(G, g, edge, directed=False, weighted=False)
    print("edge=",edge)
    print("边参与构成模体：点集合：",Node_set1)
    print("边参与构成模体：边集合：",edge_set1)
end = time.process_time()
print("spend_time=",end-start)








































