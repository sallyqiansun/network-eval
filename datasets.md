# Datasets to use

1. Cora (subset)
    - Type of dataset: Citation network
    - Description: The Cora dataset consists of 2708 scientific publications classified into one
    of seven classes. The citation network consists of 5429 links. Each publication
    in the dataset is described by a 0/1-valued word vector indicating the 
    absence/presence of the corresponding word from the dictionary. The dictionary 
    consists of 1433 unique words. 
    - Summary: 2708 vertices, 5429 edges, 1433 dimensional node covariates, 7 classes (single-class).
    - Tasks: node classification; link prediction (via artificial splits)
    - Examples of processing code (code includes download links):
        - https://stellargraph.readthedocs.io/en/v1.2.1/_modules/stellargraph/datasets/datasets.html
        - https://github.com/pyg-team/pytorch_geometric/blob/master/torch_geometric/datasets/planetoid.py


2. Cora (full)
    - Type of dataset: Citation network
    - Description: An extended version of the Cora dataset, with more
    papers, edges, and classes.
    - Summary: 19,793 vertices, 130,622 edges, 8,710 dimensional node covariates, 70 classes (single-class)
    - Comments: may be benficial to just perform some dimension reduction on the covariates if they are 0/1-valued.   
    - Tasks: node classification; link prediction (via artificial splits)
    - Examples of processing code (code includes download links): 
        - https://docs.dgl.ai/_modules/dgl/data/gnn_benchmark.html#CoraFullDataset (may need to dig around to get the exact bits of code)


3. CiteSeer
    - Type of dataset: Citation network
    - Description: The CiteSeer dataset consists of 3312 scientific publications classified 
    into one of six classes. The citation network consists of 4732 links, although 17 of these 
    have a source or target publication that isn't in the dataset and only 4715 are included in the graph.
    Each publication in the dataset is described by a 0/1-valued word vector indicating the absence/presence 
    of the corresponding word from the dictionary. The dictionary consists of 3703 unique words.
    - Summary: 3312 vertices, 4732 edges, 3703 dimensional node covariates, 6 classes (single-class)
    - Tasks: node classification; link prediction (via artificial splits)
    - Examples of processing code (code includes download links):
        - https://github.com/pyg-team/pytorch_geometric/blob/master/torch_geometric/datasets/planetoid.py
        - https://stellargraph.readthedocs.io/en/v1.2.1/_modules/stellargraph/datasets/datasets.html


4. PubMedDiabetes
    - Type of dataset: Citation network
    - Description: The Pubmed Diabetes dataset consists of 19717 scientific publications
    from PubMed database pertaining to diabetes classified into one of three classes. 
    The citation network consists of 44338 links. Each publication in the dataset 
    is described by a TF/IDF weighted word vector from a dictionary which consists 
    of 500 unique words.
    - Summary: 19717 vertices, 44338 edges, 500 dimensional node covariates, 3 classes (single-class)
    - Tasks: node classification; link prediction (via artificial splits)
    - Examples of processing code (code includes download links):
        - https://stellargraph.readthedocs.io/en/v1.2.1/_modules/stellargraph/datasets/datasets.html
        - https://github.com/pyg-team/pytorch_geometric/blob/master/torch_geometric/datasets/planetoid.py


5. Amazon co-purchase network
    - Type of dataset: Co-purchasing network
    - Description: Amazon Computers and Amazon Photo are segments of the 
    Amazon co-purchase graph [McAuley et al., 2015], where nodes represent goods, 
    edges indicate that two goods are frequently bought together, node features 
    are bag-of-words encoded product reviews, and class labels are given by the 
    product category.
    - Summary: 13,752 vertices, 574418 edges, 767 dimensional node covariates, 10 classes (single-class)
    - Tasks: node classification; link prediction (via artificial splits)
    - Examples of processing code: 
        - https://docs.dgl.ai/api/python/dgl.data.html#amazon-co-purchase-dataset (you may need to 
        dig around in the different code bases to get everything here, sorry!)


6. Homo-sapiens PPI
    - Type of dataset: protein-protein interaction network(s)
    - Description:
    - Summary: A collection of 24 graphs, where vertices have 121
    dimensional features, with 50 classes (multi-class). Note that
    the graph used in the evaluation for node2vec has 3890 vertices,
    76583 edges. 
    - Tasks: mutli-class prediction (both on individual graphs, and by
    inductive methods to generalize to different graphs); link
    prediction (via artificial splits on individual graphs, or via
    inductive methods to generalize to new graphs)
    - Examples of processing code (code includes download links):
        - (node2vec paper dataset) https://github.com/wooden-spoon/relational-ERM/blob/master/src/relational_erm/data_processing/node2vec-datasets/homo_sapiens.py
        - https://github.com/pyg-team/pytorch_geometric/blob/master/torch_geometric/datasets/ppi.py


7. BlogCatalog
    - Type of dataset: Social network
    - Description: This is a network of social relationships
    of the bloggers listed on the BlogCatalog website. The labels
    represent blogger interests inferred through the metadata
    provided by the bloggers. The network has 10,312 nodes,
    333,983 edges, and 39 different labels.
    - Summary: 10312 nodes, 333983 edges, no node covariates, 39 classes (multi-class)
    - Tasks: multi-class prediction; link prediction (via artificial splits)
    - Examples of processing code:
        - https://github.com/wooden-spoon/relational-ERM/blob/master/src/relational_erm/data_processing/node2vec-datasets/blog_catalog_3.py


8. GitHub
    - Type of dataset: Social network
    - Description: The GitHub Web and ML Developers dataset introduced in the “Multi-scale Attributed Node Embedding” paper. Nodes represent developers on GitHub and edges are mutual follower relationships.
    - Summary: 37,300 nodes, 578,006 edges, 128 node features and 2 classes
    - Note: This could be interesting purely because of the small number of classes; I think this is a good candidate for a dataset where we may be able
    to get the information purely from the node features.
    - Examples of processing code:
        - https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/datasets/github.html#GitHub


9. Reddit
    - Type of dataset: Social network
    - Note: This is very large! We're not gonna be able to do
    everything on this. 
    - Description: The Reddit dataset from the [Inductive 
    Representation Learning on
    Large Graphs](https://arxiv.org/abs/1706.02216) paper, containing
    Reddit posts belonging to different communities. The Reddit posts 
    in the network were made in the month of September, 2014. The node 
    label in this case is the “subreddit”, that a post belongs to. Posts
    are connected if the same user comments on both. The training/test split
    specifies the first 20 days for training,
    and the remaining days for testing.
    - Summary: 233k nodes, > 100 million edges, 300 dimensional node
    features, 50 classes
    - Tasks: node classification; link prediction (inductive, due to
    training/test splits induced across time points)
    - Examples of processing code: 
        - https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/datasets/reddit.html#Reddit


10. Reddit (but sparser)
    - Type of dataset: Social network
    - Description: This is a sparser version of the Reddit network.
    - Summary: 233k nodes, 11.6 million edges, 300 dimensional node
    features, 50 classes (single class)
    - Tasks: node classification; link prediction (inductive, due to
    training/test splits induced across time points)
    - Examples of processing code:
        - https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/datasets/reddit2.html#Reddit2

# Some papers re: datasets and evaluation of network methods

- https://arxiv.org/abs/1811.05868
- https://arxiv.org/abs/2005.00687
- https://arxiv.org/abs/2103.09430
- https://arxiv.org/abs/1909.13021 (this is just moreso because of Table 1 being a nice summary of some particular things)
- https://arxiv.org/abs/1902.07153 (including this so we have some
references as to how papers will try and do different evaluations)

# Ideas for some fun experiments

- Nonesense classes? What if we randomly create classes across the 
network, and use these instead?
- Robustness to pertubation of classes? Corrupt some small fixed 
percentage of the different edges?
- For these very large networks - how much sparser can we make them
before the information they contain substantially degrades?
- For datasets with temporal aspects - how does performance change if
the splits no longer reflect the proper direction of time? e.g
randomly splitting, rather than doing so probably w.r.t to the time
dimension