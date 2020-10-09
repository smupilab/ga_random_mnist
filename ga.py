import networkx as nx
from gen_fclayer import gen_fclayer
from perturb import perturb
from gen_nn import gen_nn
import os

N = 10 # number of population in each generation
PERTURB_RATIO = 0.5
PATIENCE = 3

# generate N entities
fcl = gen_fclayer(64, 32)

curr_gen = [fcl]
prev_model = [None]
for _ in range(N-1):
    g = perturb(fcl)
    curr_gen.append(g)
    prev_model.append("gen0_ent0")
    print("## Perturbation done...")
print("## Initial population generated")

termination_condition = False

no_improvements = 0
prev_best = 0
gen = 0
while not termination_condition:
    # evaluation
    acc_list = []
    for i in range(len(curr_gen)):
        # run python        
        py_fname = "run/gen{}_ent{}.py".format(gen, i)
        ## ckpt_fname = "weights/gen{}_ent{}.ckpt".format(gen, i)
        weight_fname = "weights/gen{}_ent{}.weights".format(gen, i)
        if prev_model[i] == None:
            prev_weights = None
        else:
            prev_weights = "weights/{}.weights".format(prev_model[i])
        pklname = "run/gen{}_ent{}.pickle".format(gen, i)
        logname = "log/gen{}_ent{}.log".format(gen, i)
        gen_nn(curr_gen[i], py_fname, weight_fname, prev_weights)
        #nx.write_gpickle(curr_gen[i], pklname)
        stream = os.popen("python3 {} | tee {}".format(py_fname, logname))
        result = stream.read()
        x = result.find("Final Accuracy =")
        acc = float(result[x+17:])
        acc_list.append({"index": i, "acc": acc})

    acc_list = sorted(acc_list, key=lambda k:k["acc"], reverse=True)
    curr_best = acc_list[0]["acc"]
    
    print("## Generation {} ##".format(gen))
    for a in acc_list:
        print("    entity{}, acc={}".format(a["index"], a["acc"]))

    if (prev_best < curr_best):
        prev_best = curr_best
        # save the best solution        
        nx.write_gpickle(curr_gen[acc_list[0]["index"]], "mnist_best_graph.pickle")
        no_improvements = 0
    else:
        no_improvements = no_improvements + 1

    print("## No Improvements: {}".format(no_improvements))
        
    if (no_improvements >= PATIENCE):
        termination_condition = True
    else:
        # next generation
        survival = acc_list[:int((1-PERTURB_RATIO)*N)]
        next_gen = [curr_gen[a["index"]] for a in survival]

        # perturbation
        prev_model = ["gen{}_ent{}".format(gen, a["index"]) for a in survival]
        for n in range(int(PERTURB_RATIO*N)):
            next_gen.append(perturb(curr_gen[acc_list[n]["index"]]))
            prev_model.append("gen{}_ent{}".format(gen, acc_list[n]["index"]))
        print("## Perturbation for next generation done")
        curr_gen = next_gen

        print("## previous model index = ", prev_model)
    gen = gen + 1
    
# print result
print("Best accuracy = {}".format(prev_best))

