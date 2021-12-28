import networkx as nx
import numpy as np
import pymc
import json

#reference https://github.com/aburnap/Mixed-Membership-Stochastic-Blockmodel/blob/master/MMSB_mcmc/MMSB_model.py

def run(G):
    num_nodes = G.number_of_nodes()
    nodes = list(G.nodes())
    labels = list(nx.get_node_attributes(G, 'label').values())
    num_labels = len(set(labels))



    alpha = np.ones(num_nodes).ravel() * 0.1
    B = np.eye(num_labels) * 0.8
    B = B + np.ones([num_labels, num_labels]) * 0.2 - np.eye(num_labels) * 0.2

    model = create_model(nodes, labels, alpha, B)
    model_instance = pymc.Model(model)
    pymc.MAP(model_instance).fit(method='fmin_powell')
    M = pymc.MCMC(model)
    M.sample(100000, 50000, thin=5, verbose=0)

def create_model(nodes, labels, alpha, B):
    num_nodes = len(nodes)

    pi_list = np.empty(num_nodes, dtype=object)
    for person in range(num_nodes):
        person_pi = pymc.Dirichlet('pi_%i' % person, theta=alpha)
        pi_list[person] = person_pi

    z_pTq_matrix = np.empty([num_nodes,num_nodes], dtype=object)
    z_pFq_matrix = np.empty([num_nodes,num_nodes], dtype=object)
    for p_person in range(num_nodes):
        for q_person in range(num_nodes):
            z_pTq_matrix[p_person,q_person] = pymc.Multinomial('z_%dT%d_vector' % (p_person,q_person), n=1, p=labels[p_person])
            z_pFq_matrix[p_person,q_person] = pymc.Multinomial('z_%dF%d_vector' % (p_person,q_person), n=1, p=labels[q_person])

    #---------------------------- Data Level ---------------------------------#
    # Combination of Priors to build the scalar parameter for y~Bernoulli
    @pymc.deterministic
    def bernoulli_parameters(z_pTq=z_pTq_matrix, z_pFq=z_pFq_matrix, B=B):
        """
        Takes in the two z_lists of Categorical Stochastic Objects
        Take their values (using Deterministic class)
        Dot Product with z'Bz
        """
        bernoulli_parameters = np.empty([num_nodes, num_nodes], dtype=object)
        for p in range(num_nodes):
            for q in range(num_nodes):
                bernoulli_parameters[p,q] = np.dot(np.dot(z_pTq[p,q], B), z_pFq[p,q])
        return bernoulli_parameters.reshape(1,num_nodes*num_nodes)

    y_vector = pymc.Bernoulli('y_vector', p=bernoulli_parameters, value=nodes, observed=True)

    return locals()

config_file = open('config.json', "r")
config = json.load(config_file)
data_path = "data/" + config["data"] + ".gpickle"
graph = nx.read_gpickle(data_path)
run(graph)