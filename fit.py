# allows model fitting

import node2vec
import grarep
import line
import matfact
import sbm

def model(config, G):
    if config['method'] == "node2vec" or config['method'] == "deepwalk":
        node2vec.run(config, G)

    elif config['method'] == "grarep":
        grarep.run(config, G)

    elif config["method"] == "matfact":
        matfact.run(config, G)

    elif config["method"] == "line":
        line.run(config, G)

    elif config["method"] == "sbm":
        sbm.run(config, G)

    else:
        raise Exception("Method not supported. ")

