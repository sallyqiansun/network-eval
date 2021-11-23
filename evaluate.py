# evaluation metrics

import sys
from six import iteritems
from collections import defaultdict

def save(all_results, config):

    file = sys.stdout
    sys.stdout = open(config['eval-path'], "w")

    for train_percent in sorted(all_results.keys()):
        print('Train percent:', train_percent)
        score_dict = all_results[train_percent]
        print('Average score:', score_dict)
        print('-------------------')

    sys.stdout.close()
    sys.stdout = file
    return


