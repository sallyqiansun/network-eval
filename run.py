import json
import argparse
import fit
import task
import evaluate
import visualize
import networkx as nx
from collections import defaultdict
#parse args
parser = argparse.ArgumentParser(description="Pipeline of network-eval.")
parser.add_argument('--config', help="Configuration file path. ")
args = parser.parse_args()


config_file = open(args.config, "r")
config = json.load(config_file)



# load data
data_path = "examples/" + config["data"] + ".gpickle"
graph = nx.read_gpickle(data_path)
print("Graph loaded successfully from ", data_path)


# fit model and save embeddings
fit.model(config, graph)
print("Learning finished for model", config['method'])

# task
result = task.train(data_path, graph, config)
print("Completed training on task", config["task"])

#visualization
visualize(config)

# evaluate
evaluate.save(result, config)
print("Evaluation results saved to", config['eval-path'])






