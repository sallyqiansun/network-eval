import json
import argparse
import data
import fit
import task

#parse args
parser = argparse.ArgumentParser(description="Pipeline of network-eval.")
parser.add_argument('--config', help="Configuration file path. ")
args = parser.parse_args()


config_file = open(args.config, "r")
config = json.load(config_file)

# load data
data_path = 'examples/' + config['data'] + '.' + config['format']
print("Data retrived from ", data_path)
graph = data.read_graph(config['format'], data_path, config['directed'], config['weighted'], config['mat-variable-name'])

# fit model and save embeddings
fit.model(config, graph)

# task
task.train(data_path, config)

# evaluate







