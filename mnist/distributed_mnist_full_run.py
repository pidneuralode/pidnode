from os import path
import sys

sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
import argparse
import csv

from anode_data_loader import distributed_mnist_train
from anode_data_loader import distributed_mnist_test
from base import *
from mnist.distributed_mnist_train import Trainer

import torch
from torch.utils.data import Dataset, DataLoader
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import os

parser = argparse.ArgumentParser()
parser.add_argument('--tol', type=float, default=1e-5)
parser.add_argument('--adjoint', type=eval, default=True)
parser.add_argument('--visualize', type=eval, default=True)
parser.add_argument('--niters', type=int, default=20)
parser.add_argument('--batch-size', type=int, default=64)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--timeout', type=int, default=10000, help='Maximum time/iter to do early dropping')
parser.add_argument('--nfe-timeout', type=int, default=300,
                    help='Maximum nfe (forward or backward) to do early dropping')
# parser.add_argument('--names', nargs='+',
#                     default=['node', 'anode', 'sonode', 'hbnode', 'ghbnode', 'nnode', 'nesterovnode', 'gnesterovnode',
#                              'high_nesterovnode', 'ghigh_nesterovnode'],
#                     help="List of models to run")
parser.add_argument('--names', nargs='+',
                    default=['hbnode', 'high_nesterovnode2', 'ghigh_nesterovnode2'],
                    help="List of models to run")
parser.add_argument('--log-file', default="outdat1", help="name of the logging csv file")
parser.add_argument('--no-run', action="store_true", help="To not run the training procedure")
args = parser.parse_args()

torch.autograd.set_detect_anomaly(True)


class anode_initial_velocity(nn.Module):

    def __init__(self, in_channels, aug, gpu, dch=1):
        super(anode_initial_velocity, self).__init__()
        self.aug = aug
        self.in_channels = in_channels
        self.dch = dch
        self.gpu_id = gpu

    def forward(self, x0):
        outshape = list(x0.shape)
        outshape[1] = self.aug * self.dch
        out = torch.zeros(outshape).to(self.gpu_id)
        out[:, :1] += x0
        out = rearrange(out, 'b (d c) ... -> b d c ...', d=self.dch)
        return out


class hbnode_initial_velocity(nn.Module):

    def __init__(self, in_channels, out_channels, nhid):
        super(hbnode_initial_velocity, self).__init__()
        assert (3 * out_channels >= in_channels)
        self.actv = nn.LeakyReLU(0.3)
        self.fc1 = nn.Conv2d(in_channels, nhid, kernel_size=1, padding=0)
        self.fc2 = nn.Conv2d(nhid, nhid, kernel_size=3, padding=1)
        self.fc3 = nn.Conv2d(nhid, 2 * out_channels - in_channels, kernel_size=1, padding=0)
        self.out_channels = out_channels
        self.in_channels = in_channels

    def forward(self, x0):
        x0 = x0.float()
        out = self.fc1(x0)
        out = self.actv(out)
        out = self.fc2(out)
        out = self.actv(out)
        out = self.fc3(out)
        out = torch.cat([x0, out], dim=1)
        out = rearrange(out, 'b (d c) ... -> b d c ...', d=2)
        return out


class DF(nn.Module):

    def __init__(self, in_channels, nhid, gpu, out_channels=None):
        super(DF, self).__init__()
        if out_channels is None:
            out_channels = in_channels
        self.activation = nn.ReLU(inplace=True)
        self.fc1 = nn.Conv2d(in_channels + 1, nhid, kernel_size=1, padding=0)
        self.fc2 = nn.Conv2d(nhid + 1, nhid, kernel_size=3, padding=1)
        self.fc3 = nn.Conv2d(nhid + 1, out_channels, kernel_size=1, padding=0)
        self.gpu_id = gpu

    def forward(self, t, x0):
        x0 = rearrange(x0, 'b d c x y -> b (d c) x y')
        t_img = torch.ones_like(x0[:, :1, :, :]).to(device=self.gpu_id) * t
        out = torch.cat([x0, t_img], dim=1)
        out = self.fc1(out)
        out = self.activation(out)
        out = torch.cat([out, t_img], dim=1)
        out = self.fc2(out)
        out = self.activation(out)
        out = torch.cat([out, t_img], dim=1)
        out = self.fc3(out)
        out = rearrange(out, 'b c x y -> b 1 c x y')
        return out


class predictionlayer(nn.Module):
    def __init__(self, in_channels, truncate=False, dropout=0.0):
        super(predictionlayer, self).__init__()
        self.dense = nn.Linear(in_channels * 28 * 28, 10)
        self.truncate = truncate
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        if self.truncate:
            x = rearrange(x[:, 0], 'b ... -> b (...)')
        else:
            x = rearrange(x, 'b ... -> b (...)')
        x = self.dropout(x)
        x = self.dense(x)
        return x


class tvSequential(nn.Sequential):
    def __init__(self, ic, layer, predict):
        super(tvSequential, self).__init__(ic, layer, predict)
        self.ic = ic
        self.layer = layer
        self.predict = predict

    def forward(self, x):
        x = self.ic(x)
        x, rec = self.layer(x)
        out = self.predict(x)
        return out, rec


tanh = nn.Tanh()
hard_tanh_half = nn.Hardtanh(-0.5, 0.5)


def model_gen(name, gpu, **kwargs):
    if name == 'node':
        dim = 1
        nhid = 92
        evaluation_times = torch.Tensor([1, 2]).to(device=gpu)
        layer = NODElayer(NODE(DF(dim, nhid, gpu)), time_requires_grad=False, evaluation_times=evaluation_times,
                          **kwargs)
        model = nn.Sequential(anode_initial_velocity(1, dim, gpu),
                              layer, predictionlayer(dim))
    elif name == 'anode':
        dim = 6
        nhid = 64
        evaluation_times = torch.Tensor([1, 2]).to(device=gpu)
        layer = NODElayer(NODE(DF(dim, nhid, gpu)), time_requires_grad=False, evaluation_times=evaluation_times,
                          **kwargs)
        model = nn.Sequential(anode_initial_velocity(1, dim, gpu),
                              layer, predictionlayer(dim))
    elif name == 'sonode-':
        dim = 1
        nhid = 65
        evaluation_times = torch.Tensor([1, 2]).to(device=gpu)
        hblayer = NODElayer(SONODE(DF(2 * dim, nhid, gpu, dim)), time_requires_grad=False,
                            evaluation_times=evaluation_times,
                            **kwargs)
        model = nn.Sequential(hbnode_initial_velocity(1, dim, nhid),
                              hblayer, predictionlayer(dim, truncate=True))
    elif name == 'sonode':
        dim = 5
        nhid = 50
        evaluation_times = torch.Tensor([1, 2]).to(device=gpu)
        hblayer = NODElayer(SONODE(DF(2 * dim, nhid, gpu, dim)), time_requires_grad=False,
                            evaluation_times=evaluation_times,
                            **kwargs)
        model = nn.Sequential(hbnode_initial_velocity(1, dim, nhid),
                              hblayer, predictionlayer(dim, truncate=True))
    elif name == 'hbnode':
        dim = 5
        nhid = 50
        evaluation_times = torch.Tensor([1, 2]).to(device=gpu)
        layer = NODElayer(HeavyBallNODE(DF(dim, nhid, gpu), None), time_requires_grad=False,
                          evaluation_times=evaluation_times, **kwargs)
        model = nn.Sequential(hbnode_initial_velocity(1, dim, nhid),
                              layer, predictionlayer(dim, truncate=True))
    elif name == 'ghbnode':
        dim = 5
        nhid = 50
        evaluation_times = torch.Tensor([1, 2]).to(device=gpu)
        layer = NODElayer(HeavyBallNODE(DF(dim, nhid, gpu), actv_h=nn.Tanh(), corr=2.0, corrf=False),
                          time_requires_grad=False, evaluation_times=evaluation_times, **kwargs)
        model = nn.Sequential(hbnode_initial_velocity(1, dim, nhid),
                              layer, predictionlayer(dim, truncate=True))
    elif name == 'nesterovnode':
        dim = 5
        nhid = 50
        evaluation_times = torch.Tensor([1, 2]).to(device=gpu)
        layer = NODElayer(NesterovNODE(DF(dim, nhid, gpu), None, use_h=True, sign=-1), nesterov_algebraic=True,
                          time_requires_grad=False, evaluation_times=evaluation_times, **kwargs)
        model = nn.Sequential(hbnode_initial_velocity(1, dim, nhid),
                              layer, predictionlayer(dim, truncate=True))
    elif name == 'gnesterovnode':
        dim = 5
        nhid = 50
        evaluation_times = torch.Tensor([1, 2]).to(device=gpu)
        layer = NODElayer(
            NesterovNODE(DF(dim, nhid, gpu), actv_h=hard_tanh_half, actv_df=hard_tanh_half, corr=2.0, corrf=False,
                         use_h=True, sign=-1), nesterov_algebraic=True, activation_h=hard_tanh_half,
            time_requires_grad=False, evaluation_times=evaluation_times, **kwargs)
        model = nn.Sequential(hbnode_initial_velocity(1, dim, nhid),
                              layer, predictionlayer(dim, truncate=True))
    elif name == 'high_nesterovnode':
        dim = 5
        nhid = 50
        evaluation_times = torch.Tensor([1, 2]).to(device=gpu)
        layer = NODElayer(
            High_NesterovNODE(DF(dim, nhid, gpu), actv_h=hard_tanh_half, actv_df=hard_tanh_half, corr=2.0, corrf=False,
                         use_h=True, sign=-1, u = 0.1, s = 0.1), nesterov_algebraic=True, activation_h=hard_tanh_half,
            time_requires_grad=False, evaluation_times=evaluation_times,**kwargs)
        model = nn.Sequential(hbnode_initial_velocity(1, dim, nhid),
                              layer, predictionlayer(dim, truncate=True))
    elif name == 'ghigh_nesterovnode':
        dim = 5
        nhid = 50
        evaluation_times = torch.Tensor([1, 2]).to(device=gpu)
        layer = NODElayer(
            Generalized_High_NesterovNODE(DF(dim, nhid, gpu), actv_h=hard_tanh_half, actv_df=hard_tanh_half, corr=2.0, corrf=False,
                         use_h=True, sign=-1, u=0.1, s=0.1), nesterov_algebraic=True, activation_h=hard_tanh_half,
            time_requires_grad=False, evaluation_times=evaluation_times, **kwargs)
        model = nn.Sequential(hbnode_initial_velocity(1, dim, nhid),
                              layer, predictionlayer(dim, truncate=True))
    elif name == 'high_nesterovnode2':
        dim = 5
        nhid = 50
        evaluation_times = torch.Tensor([1, 2]).to(device=gpu)
        layer = NODElayer(HighNesterovNODE2(DF(dim, nhid, gpu), None, use_h=True, sign=-1), nesterov_algebraic=True,
                          time_requires_grad=False, evaluation_times=evaluation_times, **kwargs)
        model = nn.Sequential(hbnode_initial_velocity(1, dim, nhid),
                              layer, predictionlayer(dim, truncate=True))
    elif name == 'ghigh_nesterovnode2':
        dim = 5
        nhid = 50
        evaluation_times = torch.Tensor([1, 2]).to(device=gpu)
        layer = NODElayer(
            HighNesterovNODE2(DF(dim, nhid, gpu), actv_h=hard_tanh_half, actv_df=hard_tanh_half, corr=2.0, corrf=False,
                         use_h=True, sign=-1), nesterov_algebraic=True, activation_h=hard_tanh_half,
            time_requires_grad=False, evaluation_times=evaluation_times, **kwargs)
        model = nn.Sequential(hbnode_initial_velocity(1, dim, nhid),
                              layer, predictionlayer(dim, truncate=True))
    else:
        print('model {} not supported.'.format(name))
        model = None

    gpu = torch.device(f"cuda:{gpu}")
    print(gpu)
    # 将对应的模型加载到对应的GPU上
    return model.to(gpu)


def prepare_dataloader(train_set: Dataset, test_set: Dataset, batch_size: int):
    # 对于数据集进行封装，获取并行分发的数据集
    return DataLoader(
        train_set,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=False,
        sampler=DistributedSampler(train_set)
    ), DataLoader(test_set,
                  batch_size=batch_size,
                  shuffle=True,
                  num_workers=4
                  )


def ddp_setup(rank, world_size):
    """
    Args:
        rank: 进程的唯一标识，在 init_process_group 中用于指定当前进程标识，表示当前进程中GPU编号。
        world_size: 进程总数，也表示总共有多少个GPU。
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    # linux:init_process_group(backend="nccl", rank=rank, world_size=world_size)
    # windows:init_process_group(backend="nccl", rank=rank, world_size=world_size)
    init_process_group(backend="gloo", rank=rank, world_size=world_size)
    # 使GPU负载均衡
    torch.cuda.set_device(rank)
    torch.cuda.set_per_process_memory_fraction(0.85, rank)


def ddp_cleanup():
    """
    在所有任务完成之后消灭进程使用，位置和setup函数是对齐的。
    Returns
    -------

    """
    destroy_process_group()


def load_train_objs(name, gpu):
    """
    返回预加载训练的对象
    Parameters
    ----------
    name 模型名称
    gpu 需要加载到gpu的部分

    Returns
    -------
    train_set(训练集), model(模型), optimizer(优化器), lrscheduler(学习率调整器)
    """
    model = model_gen(name, tol=args.tol, gpu=gpu)
    # 打印当前构建模型的层级信息
    print(name, count_parameters(model), *[count_parameters(i) for i in model])
    model = DDP(model, device_ids=[gpu])
    # 配置优化器和学习率控制器
    optimizer = optim.Adam(model.parameters(), lr=args.lr / 2, weight_decay=0.000)
    lrscheduler = torch.optim.lr_scheduler.StepLR(optimizer, 200, 0.9)
    # 加载mnist数据集的dataset
    train_set = distributed_mnist_train(path_to_data='./mnist_data')
    return train_set, model, optimizer, lrscheduler


def load_test_objs(gpu):
    """
    返回预加载测试的对象
    Parameters
    ----------
    gpu 需要加载到gpu的部分

    Returns
    -------

    """
    test_set = distributed_mnist_test(path_to_data='./mnist_data')

    return test_set


def main(rank: int, name: str, world_size: int, total_epochs: int, batch_size: int, csvname: str):
    """
    mnist手写数字在每一个rank进程中的训练过程
    Parameters
    ----------
    rank          进程号
    world_size    总共的进程数目，这个和设备可用的总体GPU数目相同
    total_epochs  总共运行的轮次
    batch_size    数据批次大小

    Returns
    -------

    """
    # 1.开启多进程组并行计算
    ddp_setup(rank, world_size)
    # 2.获取mnist训练集、模型和优化器
    train_set, model, optimizer, lrscheduler = load_train_objs(name=name, gpu=rank)
    # 3.获取mnist测试集
    test_set = load_test_objs(gpu=rank)
    # 4.将数据集进行并行分发，形成新的batch_size
    train_data_loader, test_data_loader = prepare_dataloader(train_set, test_set, batch_size)
    # 5.开始并行训练操作
    trainer = Trainer(model=model, train_data=train_data_loader, test_data=test_data_loader, optimizer=optimizer,
                      gpu_id=rank,
                      lrscheduler=lrscheduler)
    trainer.train(max_epochs=total_epochs, modelname=name, testnumber=0, evalfreq=1, lrscheduler=True,
                  csvname=csvname)
    # 6.销毁并行计算的其余进程
    ddp_cleanup()


def delete_and_create_csv_file(*data_files):
    """
    删除原有的同名文件，并且创建新的文件，从而保证每次训练或者测试能够留存最新的数据样本
    Returns
    -------

    """
    for data_file in data_files:
        if os.path.exists(data_file):
            os.remove(data_file)
        # 按照指定文件路径以及文件名称创建新的文件
        f = open(data_file, 'w')
        f.close()


if __name__ == '__main__':
    # 设定运行记录的格式并打开对应的文件
    rec_names = ["model", "test#", "train/test", "iter", "loss", "acc", "forwardnfe", "backwardnfe", "time/iter",
                 "time_elapsed"]
    csvfile = open(f'./imgdat/outdat0.csv', 'w')
    evaluation_times_folder = "1_2"
    writer = csv.writer(csvfile)
    writer.writerow(rec_names)
    csvfile.close()
    dat = []

    # 按照不同的neural ode模型进行训练和测试工作
    world_size = torch.cuda.device_count()
    for name in args.names:
        runnum = name[:3]
        if not args.no_run:
            if not os.path.exists(os.getcwd() + f'\\imgdat\\{evaluation_times_folder}'):
                os.makedirs(os.getcwd() + f'\\imgdat\\{evaluation_times_folder}')
            csvname = os.getcwd() + f'\\imgdat\\{evaluation_times_folder}\\{name}_{args.tol}.csv'
            log_name = os.getcwd() + '\\output\\mnist\\log_{}.txt'.format(runnum)
            datfile_name = os.getcwd() + '\\output\\mnist\\mnist_dat_{}_{}.txt'.format(runnum, args.tol)
            delete_and_create_csv_file(csvname, log_name, datfile_name)
            log = open(log_name, 'w')
            datfile = open(datfile_name, 'wb')

            # 默认master节点的GPU的rank为0
            mp.spawn(main, args=(name, world_size, args.niters, args.batch_size, csvname), nprocs=world_size)
            # 运行信息记录
            log.writelines(['\n'] * 4)
            pickle.dump(dat, datfile)
            log.close()
            datfile.close()
