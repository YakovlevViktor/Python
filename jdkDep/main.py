import collections
import networkx as nx, os, matplotlib.pyplot as plt, numpy as np

from operator import itemgetter
from scipy import optimize

#https://networkx.github.io/documentation/networkx-1.10/reference/algorithms.traversal.html
#Check for spreading

dir = os.path.dirname(__file__)
dataFilePath = dir + '/data/out.subelj_jdk_jdk'
classFilePath = dir + '/data/ent.subelj_jdk_jdk.class.name'

classDict = {}
with open(classFilePath) as f:
    for i, line in enumerate(f):
        classDict[i + 1] = line.rstrip()

# print(classDict.keys(), '\n', classDict.items())
# print(len(classDict))
#
# print(dataFilePath)

G = nx.read_edgelist(dataFilePath, create_using=nx.MultiDiGraph())
# print(edges.number_of_edges())

print("Nnodes: ", G.number_of_nodes(), "\n")
print("Nedges: ", G.number_of_edges(), "\n")


deg3 = sorted(G.degree_iter(), key=itemgetter(1), reverse=True)
dd = deg3[0]

tree = nx.dfs_tree(G, dd[0])
def hierarchy_pos(G, root, width=1., vert_gap = 0.2, vert_loc = 0, xcenter = 0.5,
                  pos = None, parent = None):
    '''If there is a cycle that is reachable from root, then this will see infinite recursion.
       G: the graph
       root: the root node of current branch
       width: horizontal space allocated for this branch - avoids overlap with other branches
       vert_gap: gap between levels of hierarchy
       vert_loc: vertical location of root
       xcenter: horizontal location of root
       pos: a dict saying where all nodes go if they have been assigned
       parent: parent of this branch.'''
    if pos == None:
        pos = {root:(xcenter,vert_loc)}
    else:
        pos[root] = (xcenter, vert_loc)
    neighbors = G.neighbors(root)
    if parent != None:
        neighbors.remove(parent)
    if len(neighbors)!=0:
        dx = width/len(neighbors)
        nextx = xcenter - width/2 - dx/2
        for neighbor in neighbors:
            nextx += dx
            pos = hierarchy_pos(G,neighbor, width = dx, vert_gap = vert_gap,
                                vert_loc = vert_loc-vert_gap, xcenter=nextx, pos=pos,
                                parent = root)
    return pos

pos = hierarchy_pos(tree, dd[0])
nx.draw(tree, pos=pos, with_labels=True)
plt.savefig('hierarchy.png')
plt.show()
#G_ud = G.to_undirected()
#print(nx.average_shortest_path_length(G))


#
# def plotDegreeDistribution(G):
#     from collections import defaultdict
#     import numpy as np
#     import matplotlib.pyplot as plt
#     degs = defaultdict(int)
#
#     # Fit the first set
#     fitfunc = lambda p, x: 1 / (x ** p)  # Target function
#     errfunc = lambda p, x, y: fitfunc(p, x) - y  # Distance to the target function
#
#     for i in G.out_degree().values(): degs[i]+=1
#     items = sorted ( degs.items () )
#     x, y = np.array(items).T
#     y = [float(i) / sum(y) for i in y]
#
#
#     plt.plot(x, y, 'bo')
#     plt.xscale('log')
#     plt.yscale('log')
#     plt.legend(['Degree'])
#     plt.xlabel('$K$', fontsize = 20)
#     plt.ylabel('$P_K$', fontsize = 20)
#     plt.title('$out-Degree\,Distribution$', fontsize = 20)
#
#     #x = np.delete(x, [0])
#     #print(x)
#     x[0] = 1
#
#     p0 = 1  # Initial guess for the parameters
#     p1, success = optimize.leastsq(errfunc, p0, args=(x, y))
#
#     # fit = np.polyfit(x, y, 1)
#     # fit_fn = np.poly1d(fit)
#     print(p1)
#     plt.plot(x, fitfunc(p1, x), color='#ff0000')
#     plt.xscale('log')
#     plt.yscale('log')
#
#     plt.show()
#
# plotDegreeDistribution(G)
#print (nx.diameter(G_ud))

#BA = nx.barabasi_albert_graph(6434, 50)

#print ("AvPath: ", nx.diameter(BA), "AvClustering", nx.average_clustering(BA))

# G_simple = nx.Graph()
# for u, v in G_ud.edges_iter():
#     if not G_simple.has_edge(u, v):
#         G_simple.add_edge(u,v)

# clustCoeff = nx.clustering(BA)
# avg_clust = sum(clustCoeff.values()) / len(clustCoeff)

#print("Average clustering: ", avg_clust, nx.average_clustering(BA), nx.diameter(BA))
# print(G.degree())



# def most_important(G):
#  """ returns a copy of G with
#      the most important nodes
#      according to the pagerank """
#  ranking = nx.betweenness_centrality(G).items()
#  print(ranking)
#  r = [x[1] for x in ranking]
#  m = sum(r)/len(r) # mean centrality
#  t = m*3 # threshold, we keep only the nodes with 3 times the mean
#  Gt = G.copy()
#  for k, v in ranking:
#   if v < t:
#    Gt.remove_node(k)
#  return Gt
#
# Gt = most_important(G) # trimming

# # create the layout
# pos = nx.spring_layout(G)
# # draw the nodes and the edges (all)
# nx.draw_networkx_nodes(G,pos,node_color='b',alpha=0.2,node_size=8)
# nx.draw_networkx_edges(G,pos,alpha=0.1)
#
# # draw the most important nodes with a different style
# nx.draw_networkx_nodes(Gt,pos,node_color='r',alpha=0.4,node_size=254)
# # also the labels this time
# nx.draw_networkx_labels(Gt,pos,font_size=12,font_color='b')
# plt.show()

# pos = nx.spring_layout(G)
# nx.draw_networkx_nodes(G, pos, cmap=plt.get_cmap('jet'))
# nx.draw_networkx_edges(G, pos, edge_color='r', arrows=True)
# plt.show()

# 0.07532792644547627

# deg1 = sorted(G.degree_iter(),key=itemgetter(0),reverse=True)
# deg = sorted(G.degree_iter(),key=itemgetter(1),reverse=True)
# print(deg)
# print(deg1)
#
# x = []
# y = []
# counter = 0
# G1 = G.copy()
# G1 = G1.to_undirected()
# G2 = G.copy()
# G2 = G2.to_undirected()
#
#
# for i in deg:
#     G1.remove_node(i[0])
#     counter += 1
#     x.append(counter)
#     degrees = nx.degree(G1).values()
#     #print(sum(degrees), len(degrees), "1111111")
#     #y.append((sum(degrees)) / len(degrees))
#     Gc = max(nx.connected_component_subgraphs(G1), key=len)
#     y.append(Gc.number_of_nodes())
#     print(counter)
#     if counter >= 1000:
#         break
#
# x1 = []
# y1 = []
# counter = 0
#
# for i in deg1:
#     G2.remove_node(i[0])
#     counter += 1
#     x1.append(counter)
#     degrees = nx.degree(G2).values()
#     #print(sum(degrees), len(degrees), "!@$@!", sum(degrees)/len(degrees))
#     #y1.append((sum(degrees)) / len(degrees))
#     Gc = max(nx.connected_component_subgraphs(G2), key=len)
#     y1.append(Gc.number_of_nodes())
#     #print(y1)
#     print(counter)
#     if counter >= 1000:
#         break
#
# notRandom = plt.plot(x, y, color='#ff0000', label='NR')
# Random = plt.plot(x1, y1, color='#000000', label='R')
# #plt.legend([notRandom, Random],['Not Random Nodes', 'Random Nodes'])
# #plt.legend(handles=[notRandom, Random])
# plt.xlabel('Deleted nodes', fontsize = 20)
# plt.ylabel('Giant component', fontsize = 20)
# plt.title('Robustness', fontsize = 20)
# plt.show()