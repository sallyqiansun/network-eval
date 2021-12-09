from sparsebm import SBM


def run(config, G):
    number_of_clusters = config["K"]

    model = SBM(number_of_clusters)
    model.fit(G, symmetric=True)
    f = open(config['emb-path'], "w")
    f.write(model.labels)
    f.close()
    print("Embedding saved to {}.".format(config['emb-path']))