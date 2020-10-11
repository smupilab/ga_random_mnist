import networkx as nx
import matplotlib.pyplot as plt
import sys

G = nx.read_gpickle(sys.argv[1])
nx.draw(G)
plt.show()
