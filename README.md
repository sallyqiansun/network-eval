

# Embedding Methods:
##### Tasks Performed on Datasets(for reference):

  ![table.JPG](table.JPG?raw=true "https://ieeexplore.ieee.org/abstract/document/8392745")


#### 1. random-walk based methods
- DeepWalk (source: https://github.com/aditya-grover/node2vec access-date 7/31/2021)
  
    python node2vec.py --method deepwalk --format mat --input examples/blogcatalog.mat --output embedding/blogcatalog-dw.emd --num-walks 80 --workers 20 --representation-size 128 --walk-length 40 --window-size 10

- Node2Vec (source: https://github.com/aditya-grover/node2vec access-date 7/31/2021)

    python node2vec.py --method node2vec --format edgelist --input examples/karate.edgelist --output embedding/karate-n2v.emd --num-walks 80 --workers 20 --representation-size 128 --walk-length 40 --window-size 10 --p 0.9 --q 0.9


#### 2. matrix factorization methods

- GraRep (source: https://github.com/benedekrozemberczki/GraRep access-date 9/8/2021)

    python grarep.py --format edgelist --input examples/cora.csv --output embedding/cora-grarep.emd --dimensions 16 --seed 1 --order 5 --iter 1

- LINE

- HOPE


#### 3. deep neural networks methods

- SDNE

# Generative Models:

- SBM

- ERGM


# Community Detection Algorithms:

- 

# Datasets

- blogcatalog

- karate

- cora

# Evaluation Metrics:

- node classification (source: https://github.com/phanein/deepwalk access-date: 7/31/2021)
    cd evaluate
    python NodeClassification.py --emb embedding/blogcatalog-n2v.emd --network examples/blogcatalog.mat --format mat --num-shuffle 10 --all

- link prediction  (source: https://github.com/lucashu1/link-prediction DOI: 10.5281/zenodo.1408472 access-date: 9/16/2021)
    cd evaluate
    python LinkPrediction.py --emb --network --format --num-shuffle --all

- link recommendation
