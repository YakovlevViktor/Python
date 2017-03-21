from audioop import reverse

import networkx as nx, random, matplotlib.pyplot as plt

graph = 2
nVerticles = 10000

# Probability for neighbours to be infected
infectionProbability = 0.01
# Period for being cured
curePeriod = 7
# Probability for edge creation
erdosRenyiProbability = 0.02
# Number of edges to attach from a new node to existing nodes
barabasiAlbertIncNodes = 10
# Each node is connected to k nearest neighbors in ring topology
wattsStrogatzKNearest = 50
# The probability of rewiring each edge
wattsStrogatzProbability = 0.1

erdosRenyiGraph = nx.erdos_renyi_graph(nVerticles, erdosRenyiProbability)
barabasiAlbertGraph = nx.barabasi_albert_graph(nVerticles, barabasiAlbertIncNodes)
wattsStrogatzGraph = nx.watts_strogatz_graph(nVerticles, wattsStrogatzKNearest, wattsStrogatzProbability)


def makeSusceptible(G):
    return nx.set_node_attributes(G, 'SIR', 'S')


def makeInfected(G, nodeIndex, infectiousList, iterationN):
    while G.neighbors(nodeIndex) == []:
        nodeIndex = random.randint(0, nVerticles - 1)
    G.node[nodeIndex]['SIR'] = 'I'
    infectiousList.append((nodeIndex, iterationN))


def tryToInfect(graph, nodeIndex, toMakeInfectious):
    # print("Trying")
    for index in graph.neighbors(nodeIndex):
        if graph.node[index]['SIR'] == 'S':
            if random.random() <= infectionProbability:
                graph.node[index]['SIR'] = 'I'
                toMakeInfectious.append(index)


def getKey(item):
    return item[1]


if graph == 1:
    G = erdosRenyiGraph
elif graph == 2:
    G = barabasiAlbertGraph
else:
    G = wattsStrogatzGraph

makeSusceptible(G)

infectiousList = list()
toChangeToInfectious = list()
rList = list()

timelineI = list()
timelineR = list()

# First infected
makeInfected(G, random.randint(0, nVerticles - 1), infectiousList, 0)

currentIteration = 1

while True:
    # Check infected list to remove cured
    infectiousList = sorted(infectiousList, key=getKey)
    while (currentIteration - infectiousList[0][1] > curePeriod):
        infectiousList = sorted(infectiousList, key=getKey)
        nodeIndex = infectiousList.pop(0)[1]
        G.node[nodeIndex]['SIR'] = 'R'
        if len(infectiousList) == 0:
            break
        rList.append(nodeIndex)
    # Quit if everybody are OK
    if len(infectiousList) == 0:
        print("Everybody are fine!")
        break

    toChangeToInfectious = []

    # Try to infect neighbours
    for i in range(0, len(infectiousList)):
         tryToInfect(G, infectiousList[i][0], toChangeToInfectious)
    # Put infected neighbours to the infectious list
    if len(toChangeToInfectious) > 0:
        for i in range(0, len(toChangeToInfectious)):
            makeInfected(G, toChangeToInfectious[i], infectiousList, currentIteration)

    print("End of iteration", currentIteration)
    print("Now infected==========", len(infectiousList), "==========nodes")

    timelineI.append((currentIteration, len(infectiousList)))
    timelineR.append((currentIteration, len(rList)))
    currentIteration += 1

plt.plot(*zip(*timelineI))
plt.plot(*zip(*timelineR))
plt.show()
