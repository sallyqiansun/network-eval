
# **Tasks** Performed on **Datasets**:

  ![table.JPG](table.JPG?raw=true "https://ieeexplore.ieee.org/abstract/document/8392745")


# Embedding Methods:

- deepwalk (source: https://github.com/phanein/deepwalk access-date: 7/31/2021)

    python deepwalk.py --input examples/karate.edgelist --output output/karate-dw.emd --method deepwalk
  
    python deepwalk.py --format mat --input examples/blogcatalog.mat --output output/blogcatalog-dw.emd --method deepwalk --num-walks 80 --workers 20 --representation-size 128 --walk-length 40 --window-size 10

- node2vec (source: https://github.com/aditya-grover/node2vec access-date 7/31/2021)

    python node2vec.py --input examples/karate.edgelist --output output/karate-n2v.emd --method node2vec

    python node2vec.py --format mat --input examples/blogcatalog.mat --output output/blogcatalog-n2v.emd --method node2vec --workers 20

-


# Evaluation Metrics:

- node classification
    
    python evaluate-NodeClassification.py --emb output/blogcatalog-dw.emd --network examples/blogcatalog.mat --num-shuffle 10 --all

- link prediction

    python evaluate-LinkPrediction.py --operator hadamard --method n2v --p 1 --q 1 --workers 1 --all --network examples/blogcatalog.mat

- link recommendation