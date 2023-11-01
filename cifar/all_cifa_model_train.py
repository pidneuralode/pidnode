import torch
import torch.nn as nn
import torch.optim as optim
import argparse

import utils
import models

parser = argparse.ArgumentParser(
        description="Train a model for the cifar classification task"
)

parser.add_argument(
    '--models',
    choices=[
        'hbnode', 'ghbnode', 'sonode',
        'node_ss', 'ghbnode_ss',
        'anode', 'node',
        'pidnode', 'gpidnode'
    ],
    default=[
        "gpidnode"
    ],
    help="Determines which Neural ODE algorithm is used"
)

parser.add_argument(
    '--tol',
    type=float,
    default=1e-5,
    help="The error tolerance for the ODE solver"
)

parser.add_argument(
    '--xres',
    type=float,
    default=1.5
)

parser.add_argument(
    '--visualize',
    type=eval,
    default=True
)

parser.add_argument(
    '--niters',
    type=int,
    default=40,
    help='The number of iterations/epochs'
)

parser.add_argument(
    '--lr',
    type=float,
    default=0.001,
    help='The learning rate for the optimizer'
)

parser.add_argument(
    '--gpu',
    type=int,
    default=0,
    help='The GPU device number'
)

parser.add_argument(
    '--weight_decay',
    type=float,
    default=0.00,
    help='Weight decay in the optimizer'
)

parser.add_argument(
    '--timescale',
    type=int,
    default=1
)

parser.add_argument(
    '--batch-size',
    type=int,
    default=128
)

parser.add_argument(
    '--nesterov-factor',
    type=int,
    default=3
)

parser.add_argument(
    '--learnable-nesterov-factor',
    type=bool,
    default=True
)

parser.add_argument(
    '--extra-name',
    default=""
)

parser.add_argument(
    '--pid-general-type',
    type=int,
    default=1
)

parser.add_argument(
    '--ss',
    default=""
)

parser.add_argument(
    '--xi',
    type=float,
    default=1.5
)

# 临时补充pidnode参数的输入
parser.add_argument('--kp', type=float, default=2.)
parser.add_argument('--ki', type=float, default=2)
parser.add_argument('--kd', type=float, default=1.5)

# make a parser
args = parser.parse_args()

def main(modelname: str):

    args.model = modelname

    # device
    device = f'cuda:{args.gpu}'

    # shape: [time, batch, derivatives, channel, x, y]
    trdat, tsdat = utils.cifar(batch_size=args.batch_size, path_to_data='./data/cifar_data')

    # Some hyperparams
    tanh_act = nn.Tanh()
    gamma = nn.Parameter(torch.tensor([0.0]))
    factor = nn.Parameter(torch.tensor([1.0])).to(device)
    nesterov_factor = None

    evaluation_times = (1.0, 2.0)

    tanh = nn.Tanh()
    hard_tanh_half = nn.Hardtanh(-0.5, 0.5)

    # create the model nodes
    if modelname == 'node':
        dim = 3
        hidden = 125
        df = models.DF(dim, hidden, args=args)
        model_layer = models.NODElayer(models.NODE(df), evaluation_times=evaluation_times, args=args)
        iv = models.anode_initial_velocity(3, aug=dim, args=args)
        # iv = models.initial_velocity(3, dim, hidden)
    elif modelname == 'anode':
        dim = 13
        hidden = 64
        df = models.DF(dim, hidden, args=args)
        model_layer = models.NODElayer(models.NODE(df), evaluation_times=evaluation_times, args=args)
        iv = models.anode_initial_velocity(3, aug=dim, args=args)
    elif modelname == 'sonode':
        dim = 12
        hidden = 50
        df = models.DF(dim, hidden, args=args)
        model_layer = models.NODElayer(models.SONODE(df), evaluation_times=evaluation_times, args=args)
        iv = models.initial_velocity(3, dim, hidden)
    elif modelname == 'hbnode':
        dim = 12
        hidden = 51
        args.xres = 0
        df = models.DF(dim, hidden, args=args)
        iv = models.initial_velocity(3, dim, hidden)
        model_layer = models.NODElayer(models.HeavyBallNODE(df, None, thetaact=None, timescale=args.timescale),
                                       evaluation_times=evaluation_times, args=args)
    elif modelname == 'ghbnode':
        dim = 12
        hidden = 51
        args.xres = 1.5
        df = models.DF(dim, hidden, args=args)
        model_layer = models.NODElayer(models.HeavyBallNODE(df, None, thetaact=tanh_act, timescale=args.timescale),
                                       evaluation_times=evaluation_times, args=args)
        iv = models.initial_velocity(3, dim, hidden)
    elif name == 'pidnode':
        # 调整隐藏层的数目
        dim = 12
        nhid = 49
        model_layer = models.NODElayer(models.PIDNODE(models.DF(dim, nhid, args=args), sign=-1, ki=args.ki,
                                                        kp=args.kp, kd=args.kd),
                                       nesterov_algebraic=False,
                                       evaluation_times=evaluation_times,
                                       args=args)
        iv = models.pidnode_initial_velocity(3, dim, nhid)
    elif name == 'gpidnode':
        dim = 12
        nhid = 49
        model_layer = models.NODElayer(models.PIDNODE(models.DF(dim, nhid, args=args), sign=-1, actv_df=tanh_act,
                                                        actv_h=tanh_act, ki=args.ki, kp=args.kp, kd=args.kd,
                                                        general_type=args.pid_general_type, corr=2.0, corrf=False),
                                       nesterov_algebraic=False,
                                       evaluation_times=evaluation_times,
                                       args=args)
        iv = models.pidnode_initial_velocity(3, dim, nhid)
    elif modelname == 'node_ss':
        dim = 3
        hidden = 125
        df = models.DF(dim, hidden, args=args)
        method = "euler"
        step_size = 0.5
        model_layer = models.NODElayer(models.NODE(df), evaluation_times=evaluation_times, args=args, method=method,
                                       step_size=step_size)
        iv = models.anode_initial_velocity(3, aug=dim, args=args)
    elif modelname == 'ghbnode_ss':
        dim = 12
        hidden = 51
        args.xres = 1.5
        df = models.DF(dim, hidden, args=args)
        iv = models.initial_velocity(3, dim, hidden)
        method = "euler"
        step_size = 0.5
        model_layer = models.NODElayer(models.HeavyBallNODE(df, None, thetaact=tanh_act, timescale=args.timescale),
                                       evaluation_times=evaluation_times, args=args, method=method, step_size=step_size)

    # 对于所有的模型进行遍历训练操作
    # create the model
    model = nn.Sequential(
        iv,
        model_layer,
        models.predictionlayer(dim)
    ).to(device=device)

    # optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # print some summary information
    print(f'Error Tolerance: {args.tol}')
    print('Model Parameter Count:', utils.count_parameters(model))

    # train the model
    utils.train(model, optimizer, trdat, tsdat, args=args, nesterov_factor=nesterov_factor)


if __name__ == "__main__":
    for name in args.models:
        main(name)
