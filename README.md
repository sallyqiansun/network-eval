
# Model Fitting

## Probabilistic Models

### 1. Exchangeable Type Models

- SBM
    
- Mixed-Membership SBM

### 2. Exponential Random Graph Models

- ERGM

## Embedding Methods

### 1. Transductive Methods
- DeepWalk (source: https://github.com/aditya-grover/node2vec access-date 7/31/2021)
  
    python node2vec.py --method deepwalk --format mat --input examples/blogcatalog.mat --output embedding/blogcatalog-dw.emd --num-walks 80 --workers 20 --representation-size 128 --walk-length 40 --window-size 10

- Node2Vec (source: https://github.com/aditya-grover/node2vec access-date 7/31/2021)

    python node2vec.py --method node2vec --format edgelist --input examples/karate.edgelist --output embedding/karate-n2v.emd --num-walks 80 --workers 20 --representation-size 128 --walk-length 40 --window-size 10 --p 0.9 --q 0.9

- GraRep (source: https://github.com/benedekrozemberczki/GraRep access-date 9/8/2021)

    python grarep.py --format edgelist --input examples/cora.csv --output embedding/cora-grarep.emd --dimensions 16 --seed 1 --order 5 --iter 1

- LINE

### 2. Inductive Methods
- GraphSAGE

- DGI


# Tasks & Evaluation Metrics
### 1. Node Classification / Community Detection
- node classification (supervised)

    - zero/one loss, FP/FN rates, macro F1 scores, cross entropy loss
    
    python NodeClassification.py --emb embedding/blogcatalog-n2v.emd --network examples/blogcatalog.mat --format mat --num-shuffle 10 --all

    (code source: https://github.com/phanein/deepwalk access-date: 7/31/2021)

- community detection (unsupervised)

### 2. Link Prediction / Link Recommendation

- link prediction

    - zero/one loss, cross entropy loss
    
    python LinkPrediction.py --emb embedding/blogcatalog-n2v.emd --network examples/blogcatalog.mat --format mat --num-shuffle 5 --all

    (code source: https://github.com/lucashu1/link-prediction DOI: 10.5281/zenodo.1408472 access-date: 9/16/2021)

- link recommendation
    
    - L<sup>p</sup> penalties, top-K metrics

### 3. Inference of Generative Models



# Datasets
- Cora
    - node classification
    - link prediction

- Reddit

- Blogcatalog

- Karate

