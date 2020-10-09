import networkx as nx
import random
import copy

def perturb(G):
    DROP_EDGE = 0
    ADD_EDGE = 1
    DROP_NODE = 2
    ADD_NODE = 4
    P_DROP_NODE = 0.001

    H = copy.deepcopy(G)

    for _ in range(100):
        c = random.randrange(2)
        node_list = list(H.nodes())
        edge_list = list(H.edges())
        N = len(node_list)
        E = len(edge_list)

        if c == DROP_EDGE:     ## remove an edge
            e = random.randrange(E)
            H.remove_edge(edge_list[e][0], edge_list[e][1])                    
        
        elif c == DROP_NODE:     ## remove a hidden node
            done = False
            while not done:
                n = random.randrange(N)
                if (node_list[n][0] == "H"):
                    H.remove_node(node_list[n])
                    done = True
        
        elif c == ADD_EDGE:    ## add an edge
            done = False
            while not done:
                u = random.randrange(N)
                v = random.randrange(N)
                if not H.has_edge(node_list[u], node_list[v]):
                    H.add_edge(node_list[u], node_list[v])
                    if len(list(nx.simple_cycles(H))) == 0:
                        done = True
                    else:
                        H.remove_edge(node_list[u], node_list[v])
    
    return H
