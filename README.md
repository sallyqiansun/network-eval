
# Pipeline
- Install dependencies in `requirements.txt`. 
```yaml
six~=1.15.0
gensim==4.1.2
networkx==2.6.3
numpy~=1.19.2
scipy==1.7.1
tqdm==4.62.3
pandas~=1.3.4
tensorflow==2.7.0
scikit-learn~=1.0
matplotlib==3.1.1
sklearn==0.0
pymc3==3.11.4
torch==1.10.0
```

- Run the program: 
`python run.py --config config.json`. 

An example config file: 
```yaml
config = {
    "data": "cora",
    "weighted": "false",
    "directed": "false",
    "method": "node2vec",
    "task": "NodeClassification",
    "emb-path": "embedding/cora-n2v.emb",
    "eval-path": "evaluation/cora-n2v.txt",
    "train_percent": [
        0.1,
        0.5,
        0.8
    ],
    "dimensions": 128,
    "iter": 1,
    "seed": 1,
    "p": 1,
    "q": 1,
    "window-size": 10,
    "num-walks": 10,
    "walk-size": 80,
    "num-shuffles": 1,
    "order": 10,
    "edge-feature": "hadamard",
    "K": 5, 
    "batch-size": 128, 
    "batch": 300,
    "proximity": "first-order", 
    "learning-rate": 0.05, 
    "fig-path": "plots/cora-n2v.png"
}
```

- The Jupyter Notebook demo for the run-through is in:
```
    run_through.ipynb
```

# Data
Details of the datasets in the `/data` folder can be found in `datasets.md`.

We provide the code for data conversion to `.gpickle` format in `data_process.ipynb`. 

Processing of external datasets is supported by `data_process.py`:
1. `read_graph`: read graph in the format of edgelist/adglist/matfile/csv
2. `save_graph`: save graph to pickle format

# Model Fitting
## Probabilistic Models

### 1. Exchangeable Type Models

- SBM
    
- Mixed-Membership SBM

### 2. Exponential Random Graph Models

- ERGM

## Embedding Methods
### 1. Transductive Methods
- Matrix Factorization 

- DeepWalk (reference: https://github.com/aditya-grover/node2vec access-date 7/31/2021)

- Node2Vec (reference: https://github.com/aditya-grover/node2vec access-date 7/31/2021)

- GraRep (reference: https://github.com/benedekrozemberczki/GraRep access-date 9/8/2021)

- LINE (reference: https://github.com/snowkylin/line access-date 9/29/2021)

### 2. Inductive Methods
- GraphSAGE (reference: https://github.com/twjiang/graphSAGE-pytorch)

- DGI (reference: https://github.com/PetarV-/DGI)


# Tasks & Evaluation Metrics
### 1. Node Classification / Community Detection
- node classification (supervised)

    - zero/one loss, FP/FN rates, macro F1 scores, cross entropy loss
```
    NodeClassification.py
    (reference: https://github.com/phanein/deepwalk access-date: 7/31/2021)
```

- community detection (unsupervised)

### 2. Link Prediction / Recommendation

- link prediction

    - zero/one loss, cross entropy loss
```
    LinkPrediction.py
    (reference: https://github.com/lucashu1/link-prediction DOI: 10.5281/zenodo.1408472 access-date: 9/16/2021)
```

- recommendation
    
    - L<sup>p</sup> penalties, top-K metrics

### 3. Inference of Generative Models


# Visualization
```
visualize.py
```


