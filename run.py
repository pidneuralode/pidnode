import torch

from walker2d import *
import sys

from walker2d import node_rnn_walker, anode_rnn_walker, sonode_rnn_walker, hbnode_rnn_walker, ghbnode_rnn_walker, \
    pidnode_rnn_walker_rnn_walker, gpidnode_rnn_walker

run_walker = {
    'node': node_rnn_walker.main,
    'anode': anode_rnn_walker.main,
    'sonode': sonode_rnn_walker.main,
    'hbnode': hbnode_rnn_walker.main,
    'ghbnode': ghbnode_rnn_walker.main,
    'pidnode':  pidnode_rnn_walker_rnn_walker.main,
    'gpidnode': gpidnode_rnn_walker.main,
}

all_models = {
    'walker': run_walker,
}


def main(ds='walker', model='pidnode', gpu=0):
    gpu = torch.device(f"cuda:{gpu}")
    all_models[ds][model](gpu)


if __name__ == '__main__':
    args = sys.argv[1:]
    main(*args)
