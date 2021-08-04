Network-eval

- deepwalk (source: https://github.com/phanein/deepwalk access-date: 7/31/2021)

    python deepwalk.py --input examples/karate.edgelist --output output/karate-dw.emd --method deepwalk
  
    python deepwalk.py --format mat --input examples/blogcatalog.mat --output output/blogcatalog-dw.emd --method deepwalk --num-walks 80 --workers 20 --representation-size 128 --walk-length 40 --window-size 10

- node2vec (source: https://github.com/aditya-grover/node2vec access-date 7/31/2021)

    python node2vec.py --input examples/karate.edgelist --output output/karate-n2v.emd --method node2vec

    python node2vec.py --format mat --input examples/blogcatalog.mat --output output/blogcatalog-n2v.emd --method node2vec --workers 20

