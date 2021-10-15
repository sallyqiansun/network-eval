# evaluation metrics

import sys
from six import iteritems
from collections import defaultdict

def save(all_results, config):

    file = sys.stdout
    sys.stdout = open(config['eval-path'], "w")

    for train_percent in sorted(all_results.keys()):
        print('Train percent:', train_percent)
        for index, result in enumerate(all_results[train_percent]):
            print('Shuffle #%d:   ' % (index + 1), result)
        avg_score = defaultdict(float)
        for score_dict in all_results[train_percent]:
            for metric, score in iteritems(score_dict):
                avg_score[metric] += score
        for metric in avg_score:
            avg_score[metric] /= len(all_results[train_percent])
        print('Average score:', dict(avg_score))
        print('-------------------')

    sys.stdout.close()
    sys.stdout = file
    return


