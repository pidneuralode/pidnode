import argparse
from walker2d import node_rnn_walker, anode_rnn_walker, sonode_rnn_walker, hbnode_rnn_walker, ghbnode_rnn_walker, \
    pidnode_rnn_walker_rnn_walker, gpidnode_rnn_walker

parser = argparse.ArgumentParser()
parser.add_argument('--models', nargs='+', default=['pidnode'],
                    help="List of models to run")
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--batch-size', type=int, default=256)
args = parser.parse_args()

# 开始导入全部的可以使用的模型
run_walker = {
    'node': node_rnn_walker.main,
    'anode': anode_rnn_walker.main,
    'sonode': sonode_rnn_walker.main,
    'hbnode': hbnode_rnn_walker.main,
    'ghbnode': ghbnode_rnn_walker.main,
    'pidnode':  pidnode_rnn_walker_rnn_walker.main,
    'gpidnode': gpidnode_rnn_walker.main,
}

if __name__ == '__main__':
    for model in args.models:
        run_walker[model](args.gpu)
