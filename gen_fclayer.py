import networkx as nx

num_inputs = 784
num_outputs = 10

def gen_fclayer(*arg):
    hidden_layers = arg
    G = nx.DiGraph()

    # node generation
    for n in range(num_inputs):
        G.add_node("X_{}".format(n))
    for l in range(len(hidden_layers)):
        for v in range(hidden_layers[l]):
            G.add_node("H_{}_{}".format(l, v))
    for n in range(num_outputs):
        G.add_node("O_{}".format(n))

    # input layer
    for u in range(num_inputs):    
        for v in range(int(hidden_layers[0])):
            G.add_edge("X_{}".format(u), "H_0_{}".format(v))

    # hidden layers
    for l in range(len(hidden_layers)-1):
        for u in range(int(hidden_layers[l])):
            for v in range(int(hidden_layers[l+1])):
                G.add_edge("H_{}_{}".format(l, u), "H_{}_{}".format(l+1, v))

    # output layer
    for u in range(int(hidden_layers[-1])):
        for v in range(num_outputs):
            G.add_edge("H_{}_{}".format(len(hidden_layers)-1, u), "O_{}".format(v))

    return G
#-------------------------------
