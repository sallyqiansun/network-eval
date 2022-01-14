# Node classification experiments

As a brief summary, suppose we have a network $A_{uv}$ denoting the presence/lack of an edge between vertices $u$ and $v$, and suppose moreover that each vertex has a class label $c_u \in [K]$. (For some of the datasets we have multi-class prediction problems - we can handle these in a similar way, but for the sake of writing down notation I'll just stick with the single class case.) Note that we may also have a vector of node covariates $x_u \in \mathbb{R}^p$ for some $p \geq 1$; denote the collection of these as $X$. We then apply an embedding method which has access to both $A$ and $X$, to produce a per-node embedding vector $\omega_u \in \mathbb{R}^d$. For now supposing that we are in the single class label case, the goal is to then use the embedding vectors $\omega_u$ to make a prediction of the class label $c_u \in [K]$. 

In order to do so, we proceed as follows:

- We create a subset of nodes $U_{tr}$ and $U_{te}$, corresponding to nodes which we will use for training purposes, and one for testing purposes.
- We use $\{ (\omega_u, c_u) : u \in U_{tr} \}$ to build a classifier $\hat{f}$ which we can use to make predictions for the embedding vectors for nodes in $U_{tr}$.
- We then evaluate the performance of the classifier $\hat{f}$. Usually, there are a few metrics we can consider:
    - Accuracy and average precision
    - Macro and micro F1 scores
    - For the multi-class problems (the BlogCatalog and PPI datasets), we have $K$ classes, and any node can belong to some number of them. We can therefore think of this as a multi-output binary classification problem with class labels $c_u \in \{0, 1\}^K$. TODO: see what metrics people use for these in papers, and then use these for these tasks. 

Note that in the above outline, we have three decisions to make:

- What is the form of the training set?
- What is the form of the testing set?
- What is our choice of family for the classifier $\hat{f}$? (Note that usually a multinomial one-vs-all classifier is used.)

Our experiments will probe the choices which are usually made here without much consideration.

## Experiments

Our experiments will seek to answer the following questions:

- To what extent does using the network information benefit us for downstream tasks?
- How informative are the embeddings learnt by these algorithms? 
- Are there any biases in how informative they are across different nodes?
- Is node classification a useful tool for measuring the quality of embedding methods?

To answer these questions, we will examine the choices which are made implicitly when evaluating these types of algorithms.

### Forming the training set

In building the training set, there are a few considerations which come into play:

- How many nodes per class should we sample? In papers, this is sometimes done as a percentage of the size of the network, but I have also seen papers which will just use e.g 20 or 50 nodes, regardless of the size of the network.

- How do we sample the nodes across vertices? Usually this is done just by sampling at random. However, while we are sampling uniformly at random the nodes to use for classification, we note that (due to the power law structure of networks), nodes are not equi-equivalent with regards to the amount of information they contain. 

Note that we could also think about the usual issues with regard to class balance, but there's nothing which particularly relates to the network setting here. As a result, there are a few things we can do here:

1. When forming the training set, select different numbers of vertices (e.g 20, 50, 100, 200) for each class in order to form the testing set.
2. Change the sampling distribution of vertices within each class - we can do so according to e.g the following distributions:
    - $P(u) \propto 1$ (i.e, uniform sampling)
    - $P(u) \propto \mathrm{deg}(u)$ (i.e, sampling proportional to degree)
    - $P(u) \propto \mathrm{deg}(u)^{-1}$ (i.e, sampling inversely proportional to degree)

### Forming the testing set and evaluation

When forming the testing set, we can just pick the testing set to be sufficiently large (e.g a percentage, say 25%, of the nodes) that we do not need to consider the size to be an issue. However, there is still a question of the sampling scheme - as we usually obtain single valued metrics which are averaged over the testing set uniformly, this may not provide a significant amount of oversight. As a result, there are a few things we can do for the testing set:

3. Use one of the following sampling distributions to create the testing set, and then compute the average precision and macro/micro F1 scores:
    - $P(u) \propto 1$ (i.e, uniform sampling)
    - $P(u) \propto \mathrm{deg}(u)$ (i.e, sampling proportional to degree)
    - $P(u) \propto \mathrm{deg}(u)^{-1}$ (i.e, sampling inversely proportional to degree)

4. To understand more carefully the relationship between the degree of a node (which we will use as a proxy for how informative it is) and the classification performance, we can bin the vertices by their degree, and then compute the metrics over each bin of nodes. Note that in order to do this, we need to make sure we choose the bins sensibly - I think probably the best way to do this is to get the degree distributions for a few of the datasets, and then figure out manually the best way of doing so. (My impression is that if you choose a log scale for the degree, then the histogram defaults will give you something sensible.)

### The choice of classifier

As I've said, usually the default is to use a multinomial one-vs-all classifier for the vertices. I think it's worth exploring different possibilities for this and use some different classifiers, such as a gradient boosting method (there should be one in sklearn), and a shallow neural network.

### Robustness to noise in the labels

Note that currently we are assuming that vertices are always correctly labelled, and so there is a natural question as to whether the use of embeddings is robust when some of the class labels are incorrectly labelled. To do so, we can do the following:

- Corrupt $\gamma \times 100\%$ of the labels in the training set by permuting, for $\gamma \in \{0.05, 0.1, 0.2, 1\}$, and keep the test labels correct.
- Compute our metrics on both the training and test sets.

(The purpose of using $\gamma = 1$ is to explore whether there is a similar phenomenon to as in neural networks where you can "fit" anything to the training data - obviously, the training performance will be wrecked.)

## Workflow for experiments

Note that once we have obtained a collection of embedding vectors, in order to perform the experiments above, we actually need nothing else from the network other than the class labels and the degrees of the nodes within the network, and so we do not need to spend a large amount of time constantly retraining embeddings for every experiment. We can therefore break the workflow up into the following:

### Obtaining an embedding and the degree structure

- Pick a dataset from the following list:
    - **Single-class datasets:**
    - Cora
    - Cora (full)
    - CiteSeer
    - PubMedDiabetes
    - Amazon co-purchase dataset
    - GitHub dataset
    - **Multi-class datasets:**
    - BlogCatalog (note: this has no nodal covariates)
    - PPI dataset

- Pick an embedding method from the following list, learn the embedding of the above network, and then save it somewhere:
    - Using the node covariates directly
    - Matrix Factorization (MF), and MF with node covariates concatenated
    - DeepWalk, and DeepWalk with node covariates concatenated
    - node2vec, and node2vec with node covariates concatenated
    - [...other methods to add here]
    - GraphSAGE (trained without the node covariates, and with the node covariates)
    - DGI (must be trained with node covariates)

- Compute and save the degree distribution of the network (this is something which could be saved, along with e.g the class label). 

**NOTE:** Some methods may not be suited to larger datasets (such as e.g the matrix factorization). This is fine - just make a note of it when doing so. Note that for something like DGI, if this uses a GCN as part of training, you may struggle to get enough memory to fit it, in which case you should use the GraphSAGE pooling layers for DGI. (If this makes no sense, tell me and I'll go into and explain what is happening in more detail.)

### Using the embeddings for experiments

Experiments investigating sampling distributions and performance conditional on the degree:

- For each choice of {nodes per class for training, training sampling distribution}, learn:
    - **(single class)**
        - a multinomial one-vs-all logistic classifier;
        - a gradient boosted classification team;
        - a shallow (one hidden layer) neural network classifier
    - **(multi-class)**
        - TODO

- Conditional on this last choice, then for every choice of {test sampling distribution}, compute:
    - **(single class)**
        - the accuracy, average precision, macro F1 and micro F1 scores (on the test set)
        - bin the degree distribution as described above, and then compute these metrics for each bin and save them (again, on the test set)
    - **(multi class)**
        - TODO

Experiments exploring robustness **(single-class datasets only)**:

- Using the uniform sampling schemes for both training and testing, for each choice of {nodes per class for training, percentage of corruption}, go through the process of building a classifier and then compute the performance of the metrics on both the training and test sets. (No need to do the binning here.)

This describes one "loop" of experiments; we can then:

- Repeat this process some number of times (e.g 10 or 20);

- With this, go back and fit the embedding again (e.g, 10 or 20) times, and repeat this procedure.

(We can always do this more times, depending on how much time we have.)