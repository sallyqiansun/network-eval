import json
import argparse
import data
import fit
import task
import networkx as nx
#parse args
parser = argparse.ArgumentParser(description="Pipeline of network-eval.")
parser.add_argument('--config', help="Configuration file path. ")
args = parser.parse_args()


config_file = open(args.config, "r")
config = json.load(config_file)



# load data
data_path = "examples" + config["data"] + ".gpickle"
graph = nx.read_gpickle(data_path)

# fit model and save embeddings
fit.model(config, graph)

# task
task.train(data_path, config)

# evaluate







