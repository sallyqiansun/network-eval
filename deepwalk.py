from six.moves import range, zip, zip_longest
import random
from gensim.models import Word2Vec
from io import open
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from os import path
import run

from multiprocessing import cpu_count


class Graph():

    def __init__(self):
        super(Graph, self).__init__(list)

def build_deepwalk_corpus(G, num_paths, path_length, alpha=0,
                          rand=random.Random(0)):
    walks = []

    nodes = list(G.nodes())

    for cnt in range(num_paths):
        rand.shuffle(nodes)
        for node in nodes:
            walks.append(random_walk(G, path_length, rand=rand, alpha=alpha, start=node))

    return walks

def build_deepwalk_corpus_iter(G, num_paths, path_length, alpha=0,
                               rand=random.Random(0)):
    walks = []

    nodes = list(G.nodes())

    for cnt in range(num_paths):
        rand.shuffle(nodes)
        for node in nodes:
            yield random_walk(G, path_length, rand=rand, alpha=alpha, start=node)

def random_walk(G, path_length, alpha=0, rand=random.Random(), start=None):
    """ Returns a truncated random walk.

        path_length: Length of the random walk.
        alpha: probability of restarts.
        start: the start node of the random walk.
    """
    if start:
      path = [start]
    else:
      # Sampling is uniform w.r.t V, and not w.r.t E
      path = [rand.choice(list(G.nodes()))]

    while len(path) < path_length:
      cur = path[-1]
      if len(G[cur]) > 0:
        if rand.random() >= alpha:
          path.append(rand.choice(list(G[cur])))
        else:
          path.append(path[0])
      else:
        break
    return [str(node) for node in path]

# http://stackoverflow.com/questions/312443/how-do-you-split-a-list-into-evenly-sized-chunks-in-python
def grouper(n, iterable, padvalue=None):
    "grouper(3, 'abcdefg', 'x') --> ('a','b','c'), ('d','e','f'), ('g','x','x')"
    return zip_longest(*[iter(iterable)] * n, fillvalue=padvalue)



class Skipgram(Word2Vec):
    """A subclass to allow more customization of the Word2Vec internals."""

    def __init__(self, vocabulary_counts=None, **kwargs):

        self.vocabulary_counts = None

        kwargs["min_count"] = kwargs.get("min_count", 0)
        kwargs["workers"] = kwargs.get("workers", cpu_count())
        kwargs["size"] = kwargs.get("size", 128)
        kwargs["sentences"] = kwargs.get("sentences", None)
        kwargs["window"] = kwargs.get("window", 10)
        kwargs["sg"] = 1
        kwargs["hs"] = 1

        if vocabulary_counts != None:
          self.vocabulary_counts = vocabulary_counts

        super(Skipgram, self).__init__(**kwargs)


__current_graph = None

# speed up the string encoding
__vertex2str = None

def count_words(file):
  """ Counts the word frequences in a list of sentences.
  Note:
    This is a helper function for parallel execution of `Vocabulary.from_text`
    method.
  """
  c = Counter()
  with open(file, 'r') as f:
    for l in f:
      words = l.strip().split()
      c.update(words)
  return c

def count_textfiles(files, workers=1):
  c = Counter()
  with ProcessPoolExecutor(max_workers=workers) as executor:
    for c_ in executor.map(count_words, files):
      c.update(c_)
  return c

def count_lines(f):
  if path.isfile(f):
    num_lines = sum(1 for line in open(f))
    return num_lines
  else:
    return 0


def _write_walks_to_disk(args):
  num_paths, path_length, alpha, rand, f = args
  G = __current_graph
  with open(f, 'w') as fout:
    for walk in build_deepwalk_corpus_iter(G=G, num_paths=num_paths, path_length=path_length,
                                                 alpha=alpha, rand=rand):
      fout.write(u"{}\n".format(u" ".join(v for v in walk)))
  return f

def write_walks_to_disk(G, filebase, num_paths, path_length, alpha=0, rand=random.Random(0), num_workers=1,
                        always_rebuild=True):
  global __current_graph
  __current_graph = G
  files_list = ["{}.{}".format(filebase, str(x)) for x in list(range(num_paths))]
  expected_size = len(G)
  args_list = []
  files = []

  if num_paths <= num_workers:
    paths_per_worker = [1 for x in range(num_paths)]
  else:
    paths_per_worker = [len(list(filter(lambda z: z!= None, [y for y in x])))
                        for x in grouper(int(num_paths / num_workers)+1, range(1, num_paths+1))]

  with ProcessPoolExecutor(max_workers=num_workers) as executor:
    for size, file_, ppw in zip(executor.map(count_lines, files_list), files_list, paths_per_worker):
      if always_rebuild or size != (ppw*expected_size):
        args_list.append((ppw, path_length, alpha, random.Random(rand.randint(0, 2**31)), file_))
      else:
        files.append(file_)

  with ProcessPoolExecutor(max_workers=num_workers) as executor:
    for file_ in executor.map(_write_walks_to_disk, args_list):
      files.append(file_)

  return files


class WalksCorpus(object):
  def __init__(self, file_list):
    self.file_list = file_list
  def __iter__(self):
    for file in self.file_list:
      with open(file, 'r') as f:
        for line in f:
          yield line.split()


def combine_files_iter(file_list):
  for file in file_list:
    with open(file, 'r') as f:
      for line in f:
        yield line.split()

def main(args):
	'''
	Pipeline for representational learning for all nodes in a graph.
	'''
	G = run.read_graph(args)
	run.simulate_and_embed(args, G=G)


if __name__ == "__main__":
	args = run.parse_args()
	main(args)