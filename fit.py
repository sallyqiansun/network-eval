# allows model fitting

import node2vec
import grarep
import line
import MatrixFactorization

def model(config, G):
    if config['method'] == "node2vec" or config['method'] == "deepwalk":
        node2vec.simulate_and_embed(config, G)

    elif config['method'] == "grarep":
        grarep.simulate_and_embed(config, G)

    elif config["method"] == "MatrixFactorization":
        MatrixFactorization.mf(config, G)

    elif config["method"] == "line":
        pass

    else:
        raise Exception("Method not supported. ")

