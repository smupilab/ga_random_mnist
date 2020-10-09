import networkx as nx

def gen_nn(G, py_fname, weight_fname, prev_weights):
    batch_size = 100
    numepochs =  2#100
    num_inputs = 28 * 28
    num_outputs = 10
    num_hiddennodes = G.number_of_nodes() - num_inputs - num_outputs
    num_edges = G.number_of_edges()
    num_nodes = num_inputs + num_outputs + num_hiddennodes
    
    output = open(py_fname, "w")

    output.write("import tensorflow as tf\n")
    output.write("import pickle\n")
    output.write("\n")
    output.write("num_classes = 10\n")
    output.write("img_rows, img_cols = 28, 28\n")
    output.write("num_channels = 1\n")
    output.write("input_shape = (img_rows, img_cols, num_channels)\n")
    output.write("\n")
    output.write("(x_train, y_train),(x_test, y_test) = tf.keras.datasets.mnist.load_data()\n")
    output.write("x_train, x_test = x_train / 255.0, x_test / 255.0\n")
    output.write("\n")
    output.write("### num_inputs, num_outputs, num_hiddennodes, num_edges\n")
    output.write("### {} {} {} {}\n".format(num_inputs, num_outputs, num_hiddennodes, num_edges))
    output.write("\n")
    output.write("model_input = tf.keras.layers.Input(shape=input_shape)\n")
    output.write("flat_input = tf.keras.layers.Flatten()(model_input)\n")
    output.write("\n")

    # node generation
    for n in nx.topological_sort(G):
        if (n[0] == "X"):
            output.write("{} = tf.gather(flat_input, [{}], axis=1)\n".format(n, int(n[2:])))
        else:
            output.write("{}_node = tf.keras.layers.Dense(1, activation='relu')\n".format(n))
            output.write("{} = {}_node(tf.concat([".format(n, n))
            # edge u -> v
            E = list(G.in_edges(n))
            for u in range(len(E)-1):
                output.write("{}, ".format(E[u][0]))
            if len(E) != 0:
                output.write("{}], 1))\n".format(E[-1][0]))

    # final layer
    output.write("final_nodes = tf.concat([")
    for o in range(9):
        output.write("O_{}, ".format(o))
    output.write("O_9], 1)\n")
    output.write("\n")

    output.write("final_output = tf.nn.softmax(final_nodes, axis=1)\n")
    output.write("\n")
    output.write("model = tf.keras.Model(model_input, final_output)\n")
    output.write("model.compile(optimizer='sgd', loss='sparse_categorical_crossentropy', metrics=['accuracy'])\n")
    if prev_weights != None:
        output.write("prev_file = open('{}', 'rb')\n".format(prev_weights))
        output.write("nodes_weights = pickle.load(prev_file)\n")
        output.write("nodes = nodes_weights[0]\n")
        output.write("weights = nodes_weights[1]\n")
        output.write("nodes_set = 0\n")
        for n in nx.topological_sort(G):
            if (n[0] != "X"):
                output.write("if '{}' in nodes:\n".format(n))
                output.write("    w = weights[nodes.index('{}')]\n".format(n))
                output.write("    in_w = len(w[0])\n")
                output.write("    in_n = {}_node.input_shape[1]\n".format(n))
                output.write("    if in_w == in_n:\n")
                output.write("        {}_node.set_weights(w)\n".format(n))
                output.write("        nodes_set = nodes_set + 1\n")

        output.write("\n")
        output.write("print('## Weight loaded: {}', nodes_set, '/', len(nodes))\n".format(prev_weights))
        output.write("prev_file.close()\n")


    output.write("## model.summary()\n")
    output.write("print('Training started')\n")
    output.write("callback = tf.keras.callbacks.EarlyStopping(patience=3)\n")
    output.write("model.fit(x_train, y_train, epochs={}, verbose=2, validation_data=(x_test, y_test), callbacks=[callback])\n".format(numepochs))

    ## save weights
    output.write("nodes = []\n")
    output.write("weights = []\n")
    for n in nx.topological_sort(G):
        if (n[0] != "X"):
            output.write("w = {}_node.get_weights()\n".format(n))
            output.write("nodes.append('{}')\n".format(n))
            output.write("weights.append(w)\n")
    output.write("weight_file = open('{}', 'wb')\n".format(weight_fname))
    output.write("pickle.dump([nodes, weights], weight_file)\n")
    output.write("weight_file.close()\n")
    output.write("print('## Weight saved: {}')\n".format(weight_fname))

    output.write("test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)\n")
    output.write("print('Final Accuracy =', test_acc)\n")

    output.close()
