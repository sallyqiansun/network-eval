import json
config_file = open("config.json", "r")
config = json.load(config_file)
data_path = "examples/" + config["data"] + ".gpickle"
graph = nx.read_gpickle(data_path)
mf(config, graph)