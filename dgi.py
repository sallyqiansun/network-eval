import torch
import torch.nn as nn
import numpy as np
import networkx as nx
import scipy.sparse as sp

# (reference: https://github.com/PetarV-/DGI)

class AvgReadout(nn.Module):
    def __init__(self):
        super(AvgReadout, self).__init__()

    def forward(self, seq, msk):
        if msk is None:
            return torch.mean(seq, 1)
        else:
            msk = torch.unsqueeze(msk, -1)
            return torch.sum(seq * msk, 1) / torch.sum(msk)
class Discriminator(nn.Module):
    def __init__(self, n_h):
        super(Discriminator, self).__init__()
        self.f_k = nn.Bilinear(n_h, n_h, 1)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Bilinear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, c, h_pl, h_mi, s_bias1=None, s_bias2=None):
        c_x = torch.unsqueeze(c, 1)
        c_x = c_x.expand_as(h_pl)

        sc_1 = torch.squeeze(self.f_k(h_pl, c_x), 2)
        sc_2 = torch.squeeze(self.f_k(h_mi, c_x), 2)

        if s_bias1 is not None:
            sc_1 += s_bias1
        if s_bias2 is not None:
            sc_2 += s_bias2

        logits = torch.cat((sc_1, sc_2), 1)

        return logits
class DGI(nn.Module):
    def __init__(self, n_in, n_h, activation):
        super(DGI, self).__init__()
        self.gcn = GCN(n_in, n_h, activation)
        self.read = AvgReadout()

        self.sigm = nn.Sigmoid()

        self.disc = Discriminator(n_h)

    def forward(self, seq1, seq2, adj, msk, samp_bias1, samp_bias2):
        h_1 = self.gcn(seq1, adj)

        c = self.read(h_1, msk)
        c = self.sigm(c)

        h_2 = self.gcn(seq2, adj)

        ret = self.disc(c, h_1, h_2, samp_bias1, samp_bias2)

        return ret

    # Detach the return variables
    def embed(self, seq, adj, msk):
        h_1 = self.gcn(seq, adj)
        c = self.read(h_1, msk)

        return h_1.detach(), c.detach()


class GCN(nn.Module):
    def __init__(self, in_ft, out_ft, act, bias=True):
        super(GCN, self).__init__()
        self.fc = nn.Linear(in_ft, out_ft, bias=False)
        self.act = nn.PReLU() if act == 'prelu' else act

        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_ft))
            self.bias.data.fill_(0.0)
        else:
            self.register_parameter('bias', None)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    # Shape of seq: (batch, nodes, features)
    def forward(self, seq, adj):
        seq_fts = self.fc(seq)
        out = torch.bmm(adj, seq_fts)
        if self.bias is not None:
            out += self.bias

        return self.act(out)



def sparse_to_tuple(sparse_mx, insert_batch=False):
    """Convert sparse matrix to tuple representation."""
    """Set insert_batch=True if you want to insert a batch dimension."""
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        if insert_batch:
            coords = np.vstack((np.zeros(mx.row.shape[0]), mx.row, mx.col)).transpose()
            values = mx.data
            shape = (1,) + mx.shape
        else:
            coords = np.vstack((mx.row, mx.col)).transpose()
            values = mx.data
            shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx

def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = (1 / rowsum).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features.todense(), sparse_to_tuple(features)

def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()

def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
    return sparse_to_tuple(adj_normalized)

def run(config, G):
    adj = nx.adjacency_matrix(G)
    adj = normalize_adj(adj + sp.eye(adj.shape[0]))
    feat = nx.get_node_attributes(G, "feature")
    for key in feat:
        feat[key] = [int(k) for k in feat[key]]
    a = np.array(list(feat.values()))
    features, _ = preprocess_features(sp.csr_matrix(a))
    ft_size = features.shape[1]
    # lab = nx.get_node_attributes(G, "label")
    # labels = np.array(lab.values())

    adj = (adj + sp.eye(adj.shape[0])).todense()
    adj = torch.FloatTensor(adj[np.newaxis])
    features = torch.FloatTensor(features[np.newaxis])

    model = DGI(ft_size, config['dimensions'], activation='prelu')
    optimiser = torch.optim.Adam(model.parameters(), lr=config['learning-rate'])
    xent = nn.CrossEntropyLoss()

    for epoch in range(config['iter']):
        model.train()
        optimiser.zero_grad()

        idx = np.random.permutation(G.number_of_nodes())
        shuf_fts = features[:, idx, :]

        lbl_1 = torch.ones(1, G.number_of_nodes())
        lbl_2 = torch.zeros(1, G.number_of_nodes())
        lbl = torch.cat((lbl_1, lbl_2), 1)

        logits = model(features, shuf_fts, adj, None, None, None)
        loss = xent(logits, lbl)

        print('Loss:', loss)

        loss.backward()
        optimiser.step()

    embeds, _ = model.embed(features, adj, None)
    emb = embeds[0]

    f = open(config['emb-path'], "w")
    f.write(str(len(emb))+" ")
    f.write(str(len(emb[0])) + "\n")
    for e in emb:
        e = e.tolist()
        for i in e:
            f.write(str(i)+" ")
        f.write("\n")
    f.close()

    print("Embedding saved to {}.".format(config['emb-path']))

# import json
# config_file = open('config.json', "r")
# config = json.load(config_file)
# data_path = "data/" + config["data"] + ".gpickle"
# graph = nx.read_gpickle(data_path)
# run(graph, config)