# tasks: NodeClassification | LinkPrediction | Recommendation

import NodeClassification
import LinkPrediction


def train(data_path, graph, config):
    if config['task'] == "NodeClassification":
        return NodeClassification.run(graph, config)


    elif config['task'] == "LinkPrediction":
        return LinkPrediction.run(graph, config)


    else:
        raise Exception("Not yet supported.")


