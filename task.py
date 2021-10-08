# tasks: NodeClassification | LinkPrediction | Recommendation

import NodeClassification
import LinkPrediction


def train(data_path, config):
    if config['task'] == "NodeClassification":
        NodeClassification.run(data_path, config)


    elif config['task'] == "LinkPrediction":
        LinkPrediction.run(data_path, config)


    else:
        raise Exception("Not yet supported.")


