
# Pipeline
- install dependencies in requirements.txt
- run the program: 
`python run.py --config config.json`
![example config file](config.png?raw=true "example config file")

- to convert graph data format, run data.py:
1. read_graph: read graph in the format of edgelist/adglist/matfile/csv
2. save_graph: save graph to pickle format

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

- Node2Vec (source: https://github.com/aditya-grover/node2vec access-date 7/31/2021)

- GraRep (source: https://github.com/benedekrozemberczki/GraRep access-date 9/8/2021)

- LINE (source: https://github.com/snowkylin/line access-date 9/29/2021)

### 2. Inductive Methods
- GraphSAGE

- DGI


# Tasks & Evaluation Metrics
### 1. Node Classification / Community Detection
- node classification (supervised)

    - zero/one loss, FP/FN rates, macro F1 scores, cross entropy loss
    
    (code source: https://github.com/phanein/deepwalk access-date: 7/31/2021)

- community detection (unsupervised)

### 2. Link Prediction / Recommendation

- link prediction

    - zero/one loss, cross entropy loss

    (code source: https://github.com/lucashu1/link-prediction DOI: 10.5281/zenodo.1408472 access-date: 9/16/2021)

- recommendation
    
    - L<sup>p</sup> penalties, top-K metrics

### 3. Inference of Generative Models



# Datasets
- Cora

- Reddit

- Blogcatalog

- Karate


