import numpy as np
import networkx as nx

G = nx.karate_club_graph()
degree = [d for n, d in G.degree()]
nx.draw(G, with_labels=True)
a = G.degree()
print(a)
