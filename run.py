import json
import argparse
import fit
import task
import evaluate
import visualize
import networkx as nx
#parse args
parser = argparse.ArgumentParser(description="Pipeline of network-eval.")
parser.add_argument('--config', help="Configuration file path. ", required=True)

# parameters to change
parser.add_argument('--data', help="dataset")
parser.add_argument('--method', help="embedding method")
parser.add_argument('--task', help="task to perform")
parser.add_argument('--train_percent', help="a list of training percentage")
parser.add_argument('--dimensions', help="embedding dimensions")
parser.add_argument('--iter', help="iterations")
parser.add_argument('--seed', help="random seed")
parser.add_argument('--p', help="p in node2vec")
parser.add_argument('--q', help="q in node2vec")
parser.add_argument('--window-size', help="window size in node2vec")
parser.add_argument('--num-walks', help="number of walks in n2v")
parser.add_argument('--walk-size', help="the size of the walks in n2v")
parser.add_argument('--num-shuffles', help="random shuffles in node2vec and graphsage")
parser.add_argument('--order', help="order in grarep")
parser.add_argument('--edge-feature', help="method in link prediction, hadamard/average/l1/l2")
parser.add_argument('--K', help="K in line")
parser.add_argument('--batch-size', help="batch size")
parser.add_argument('--batch', help="number of batches")
parser.add_argument('--layers', help="number of layers")
parser.add_argument('--agg-func', help="aggregate function in graphsage")
parser.add_argument('--proximity', help="first/second-order proximity in line")
parser.add_argument('--learning-rate', help="learning rate in training")

args = parser.parse_args()

config_file = open(args.config, "r")
config = json.load(config_file)

# update the config parameters that are newly passed into run.py
for key in vars(args):
    if vars(args)[key] is not None:
        config[key] = vars(args)[key]

config['emb-path'] = "embedding/"+config['data']+'-'+config['method']+".emb"
config['eval-path'] = "evaluation/"+config['data']+'-'+config['method']+".txt"
config['fig-path'] = "plots/"+config['data']+'-'+config['method']+".png"


# load data
data_path = "data/" + config["data"] + ".gpickle"
graph = nx.read_gpickle(data_path)
if config['directed'] == "false":
    graph.to_undirected()
print("Graph loaded successfully from ", data_path)

# fit model and save embeddings
fit.model(config, graph)
print("Learning finished for model", config['method'])

# task
# result = task.train(graph, config)
# print("Completed training on task", config["task"])

# visualization
# visualize.visualize(result, config)

# evaluate
# evaluate.save(result, config)
# print("Evaluation results saved to", config['eval-path'])






