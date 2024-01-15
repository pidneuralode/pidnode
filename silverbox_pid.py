"""
Fig. 3
"""
import torch

from base import *
from sonode_data_loader import load_data
import os
import random

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

parser = ArgumentParser()
parser.add_argument('--tol', type=float, default=1e-3)
parser.add_argument('--adjoint', type=eval, default=False)
parser.add_argument('--visualise', type=eval, default=True)
parser.add_argument('--niters', type=int, default=1000)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--npoints', type=int, default=1000)
parser.add_argument('--experiment_no', type=int, default=1)
args = parser.parse_args()

randomSeed = 18
torch.manual_seed(randomSeed)
torch.cuda.manual_seed(randomSeed)
torch.cuda.manual_seed_all(randomSeed)  # if use multi-GPU
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(randomSeed)
random.seed(randomSeed)

v1_data, v2_data = load_data('./data/sb.csv', skiprows=1, usecols=(0, 1), rescaling=100)
time_rescale = 1.0
input_t = 1
forecast_t = 999
trsize = 10000
tssize = 4000
args.MODE = 0  # 0 for train and 1 for test


def preprocess(data):
    trdat = data[:trsize]
    tsdat = data[trsize:trsize + tssize]
    return trdat, tsdat


v1_data = preprocess(v1_data)
v2_data = preprocess(v2_data)
trdat = (v2_data[0][:input_t], v2_data[0])
tsdat = (v2_data[1][:input_t], v2_data[1])


class Vdiff(nn.Module):
    def __init__(self):
        super(Vdiff, self).__init__()
        self.osize = 1

    def forward(self, t, x, v):
        truev = v2_vfunc(t)
        return torch.norm(v[:, 0] - truev, 1)


def v1_func(time):
    t1 = torch.clamp(torch.floor(time), 0, len(v1_data) - 1).type(torch.long)
    delta = time - t1
    data = v1_data[args.MODE]
    return data[t1] + delta * (data[t1 + 1] - data[t1])


def v2_vfunc(time):
    t1 = torch.clamp(torch.floor(time), 0, len(v2_data) - 1).type(torch.long)
    data = v2_data[args.MODE]
    return data[t1 + 1] - data[t1]


class initial_velocity(nn.Module):

    def __init__(self, in_channels, out_channels, ddim, zpad=0):
        super(initial_velocity, self).__init__()
        self.fc1 = nn.Linear(in_channels, out_channels * ddim - in_channels - zpad, bias=False)
        self.ddim = ddim
        self.zpad = zpad

    def forward(self, x0):
        if self.zpad > 0:
            xpad = torch.cat([x0, torch.zeros(self.zpad)], dim=0)
        else:
            xpad = x0
        out = self.fc1(torch.ones_like(x0))
        out = torch.cat([xpad, out], dim=0).reshape(1, self.ddim, -1)
        return out


class initial_velocity_pid(nn.Module):

    def __init__(self, in_channels, out_channels, ddim, zpad=0):
        super(initial_velocity_pid, self).__init__()
        self.fc1 = nn.Linear(in_channels, out_channels * (ddim + 1) - in_channels - zpad, bias=False)
        self.ddim = ddim
        self.zpad = zpad

    def forward(self, x0):
        if self.zpad > 0:
            xpad = torch.cat([x0, torch.zeros(self.zpad)], dim=0)
        else:
            xpad = x0

        out = self.fc1(torch.ones_like(x0))

        out = torch.cat([xpad, out], dim=0).reshape(1, 3, -1)

        return out


class DF(nn.Module):

    def __init__(self, in_channels, out_channels=None):
        super(DF, self).__init__()
        out_channels = in_channels if out_channels is None else out_channels
        self.fc1 = nn.Linear(in_channels + 1, out_channels)
        # self.act = nn.ReLU(inplace=False)

    def forward(self, t, x):
        v1 = v1_func(t).reshape(-1, 1, 1)
        x = rearrange(x, 'b d c -> b 1 (d c)')
        # z_ = torch.cat((x, 0.01 * x ** 3, v1), dim=2)
        z_ = torch.cat((x, v1), dim=2)
        out = self.fc1(z_)
        # out = 0.4905 * x + 0.00618 * x ** 3 + 0.0613 * v1
        return out

modelnames = [
    'GNesterovNODE', 'GHBNODE',
    'PIDNODE(ki=1.5)', 'PIDNODE(ki=3)', 'PIDNODE(ki=5)',
    'PIDNODE(ki=10)', 'PIDNODE(ki=20)', 'PIDNODE(ki=1)',
    'PIDNODE(ki=0.5)', 'PIDNODE(ki=0)'
]


modelclass = [
    HeavyBallNODE,
    PIDNODE, PIDNODE, PIDNODE,
    PIDNODE, PIDNODE, PIDNODE,
    PIDNODE, PIDNODE
]
icparams = [
    (1, 2, 0),
    (1, 2, 0), (1, 2, 0), (1, 2, 0),
    (1, 2, 0), (1, 2, 0), (1, 2, 0),
    (1, 2, 0), (1, 2, 0)
]  # out_channels, ddim, zpad
dfparams = [
    (1,),
    (1,), (1,), (1,),
    (1,), (1,), (1,),
    (1,), (1,)
]
hard_tanh = nn.Hardtanh(-0.25, 0.25)
nintparams = [
    dict(),
    dict(), dict(), dict(),
    dict(), dict(), dict(),
    dict(), dict()
]

cellparams = [
    {'corr': 0, 'actv_h': hard_tanh},
    {'ki': 1.5}, {'ki': 3}, {'ki': 5},
    {'ki': 10}, {'ki': 20}, {'ki': 1},
    {'ki': 0.5}, {'ki': 0.1}
]

model_list = []
dim = 1

plt.figure(figsize=(20, 20))
axes = plt.gca()
axes.tick_params(axis='x', labelsize=50)
axes.tick_params(axis='y', labelsize=50)
colors = [
    "red",
    "deepskyblue",
    "royalblue",
    "navy",
    "green",
    "darkorange",
    'yellow',
    'purple',
    'pink'
]
line_styles = [
    ':',
    '-.',
    '-.',
    '-.',
    '--',
    '--',
    '-*',
    '-*',
    '--',
]
line_widths = [
    5,
    6,
    6,
    6,
    6,
    6,
    6,
    6,
    6
]
# '''
sizedata = []
num_epochs = 40
# plt.axis("scaled")
for i in range(2):
    print(i, modelnames[i])
    odesizelist = []
    for r in range(1):
        # for epoch_idx in range(num_epochs):
        cell = modelclass[i](DF(*dfparams[i]), **cellparams[i])
        ic = initial_velocity(input_t, *icparams[i])
        nint = NODEintegrate(cell, time_requires_grad=False, evaluation_times=torch.arange(1, 80.), tol=1e-7,
                             verbose=(r == 0), **nintparams[i])
        model = nn.Sequential(ic, nint)
        ode_states = model(trdat[0])
        ode_size = torch.norm(ode_states.reshape(ode_states.shape[0], -1), dim=1)
        odesizelist.append(ode_size.detach().numpy())
    dat = np.log10(np.mean(odesizelist, axis=0))
    plt.plot(dat, line_styles[i], linewidth=line_widths[i], label=modelnames[i], color=colors[i])
    sizedata.append(dat)

for i in range(2, 9):  # pidnode
    print(i, modelnames[i])
    odesizelist = []
    for r in range(10):
        cell = modelclass[i](DF(*dfparams[i]), **cellparams[i])
        ic = initial_velocity_pid(input_t, *icparams[i])
        nint = NODEintegrate(cell, time_requires_grad=False, evaluation_times=torch.arange(1, 80.), tol=1e-7,
                             verbose=(r == 0), **nintparams[i])
        model = nn.Sequential(ic, nint)
        ode_states = model(trdat[0])
        ode_size = torch.norm(ode_states.reshape(ode_states.shape[0], -1), dim=1)
        odesizelist.append(ode_size.detach().numpy())
    dat = np.log10(np.mean(odesizelist, axis=0))
    plt.plot(dat, label=modelnames[i], linewidth=5, color=colors[i])
    sizedata.append(dat)

plt.plot(np.log10(np.abs(trdat[1][:81])), label='Exact', linewidth=5, color='k')
tickrange = np.linspace(0, 18, 10)
plt.yticks(tickrange, ['$10^{{{}}}$'.format(int(i)) for i in tickrange])
plt.xlabel("$t$", fontsize=50)
plt.ylabel("||${\\mathbf{h}}(t)||_2$", fontsize=50)
plt.grid(visible=True, which='major', color='#666666', linestyle='-')
plt.minorticks_on()
plt.grid(visible=True, which='minor', color='#999999', linestyle='-', alpha=0.2)

handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys(), loc='upper left', fontsize=50)

plt.tight_layout()
plt.savefig('output/sb/blow_up.pdf')
plt.show()
np.savetxt('output/sb/sbinit.csv', np.array(sizedata), delimiter=',')
