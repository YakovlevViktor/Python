import collections
import networkx as nx, matplotlib.pyplot as plt

G= nx.Graph()
G.add_nodes_from([1, 10])

G.add_edges_from([(1,2),(2,3),(2,4),(3,4),(3,5),(4,5),(4,6),(5,6),(6,7),(6,8),(6,9),(7,8),(7,9),(8,9),(9,10)])

print("nVert: ", G.number_of_nodes(), "nEdges: ", G.number_of_edges())

nodeDegr = nx.degree(G).values()

print("NodeDegrees: ", nodeDegr)
print("Average degree: ", sum(nodeDegr) / 10)

nx.draw(G)
plt.show()

print(nx.average_shortest_path_length(G))
print(nx.diameter(G))

degree_sequence=sorted(nx.degree(G).values(),reverse=True) # degree sequence
#print "Degree sequence", degree_sequence
dmax=max(degree_sequence)
#
# plt.loglog(degree_sequence,'b-',marker='o')
# plt.title("Degree rank plot")
# plt.ylabel("degree")
# plt.xlabel("rank")
#
# # draw graph in inset
# plt.axes([0.45,0.45,0.45,0.45])
# Gcc=sorted(nx.connected_component_subgraphs(G), key = len, reverse=True)[0]
# pos=nx.spring_layout(Gcc)
# plt.axis('off')
# nx.draw_networkx_nodes(Gcc,pos,node_size=20)
# nx.draw_networkx_edges(Gcc,pos,alpha=0.4)
#
# plt.savefig("degree_histogram.png")
# plt.show()

print(nx.clustering(G))

clusteringAverage = nx.clustering(G).values()
print("AverrageCloef: ", sum(clusteringAverage) / 10)