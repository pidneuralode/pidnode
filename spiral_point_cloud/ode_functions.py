from time import time
import torch
from torch import nn


class NODEfunc(nn.Module):

    def __init__(self, dim, nhidden, augment_dim=0, time_dependent=True):
        super(NODEfunc, self).__init__()
        self.elu = nn.ELU(inplace=True)
        self.time_dependent = time_dependent
        dim = dim + augment_dim
        if self.time_dependent:
            self.fc1 = nn.Linear(dim + 1, nhidden)
        else:
            self.fc1 = nn.Linear(dim, nhidden)
        self.fc2 = nn.Linear(nhidden, nhidden)
        self.fc3 = nn.Linear(nhidden, dim)
        self.nfe = 0

    def forward(self, t, x):
        self.nfe += 1
        if self.time_dependent:
            # Shape (batch_size, 1)
            t_vec = torch.ones(x.shape[0], 1).to(x.get_device()) * t
            # Shape (batch_size, data_dim + 1)
            t_and_x = torch.cat([t_vec, x], 1)
            # Shape (batch_size, hidden_dim)
            out = self.fc1(t_and_x)
        else:
            out = self.fc1(x)
        out = self.elu(out)
        out = self.fc2(out)
        out = self.elu(out)
        out = self.fc3(out)
        return out


class SONODEfunc(nn.Module):
    def __init__(self, dim, nhidden, time_dependent=True, modelname=None, actv=nn.Tanh()):
        super(SONODEfunc, self).__init__()
        self.modelname = modelname
        indim = 2 * dim if self.modelname == 'SONODE' else dim
        self.time_dependent = time_dependent
        if self.time_dependent:
            indim += 1
        # only have residual for generalized model
        self.res = 2.0 if self.modelname == "GHBNODE" else 0.0
        self.elu = nn.ELU(inplace=False)
        self.fc1 = nn.Linear(indim, nhidden)
        self.fc2 = nn.Linear(nhidden, nhidden)
        self.fc3 = nn.Linear(nhidden, dim)
        self.gamma = nn.Parameter(torch.Tensor([-3.0]))
        self.actv = actv
        self.nfe = 0
        self.sigmoid = nn.Sigmoid()

    def forward(self, t, x):
        cutoff = int(len(x) / 2)
        z = x[:cutoff]
        v = x[cutoff:]
        if self.modelname == 'SONODE':
            z = torch.cat((z, v), dim=1)
        self.nfe += 1
        if self.time_dependent:
            # Shape (batch_size, 1)
            t_vec = torch.ones(z.shape[0], 1).to(x.get_device()) * t
            # Shape (batch_size, data_dim + 1)
            t_and_z = torch.cat([t_vec, z], 1)
            # Shape (batch_size, hidden_dim)
            out = self.fc1(t_and_z)
        else:
            out = self.fc1(z)
        out = self.elu(out)
        out = self.fc2(out)
        out = self.elu(out)
        if self.modelname == 'SONODE':
            out = self.fc3(out)
            return torch.cat((v, out))
        else:
            out = self.fc3(out) - self.sigmoid(self.gamma) * v - self.res * z
            if self.modelname == "GHBNODE":
                actv_v = self.actv
            else:
                actv_v = nn.Identity()
            return torch.cat((actv_v(v), out))


class NesterovNODEfunc(nn.Module):
    def __init__(self, dim, nhidden, time_dependent=True, modelname=None, xi=None, actv=nn.Tanh()):
        super(NesterovNODEfunc, self).__init__()
        self.modelname = modelname
        self.time_dependent = time_dependent
        # only have residual for generalized model
        self.res = 0.0 if xi is None else xi
        self.elu = nn.ELU(inplace=False)
        if self.time_dependent:
            self.fc1 = nn.Linear(dim + 1, nhidden)
        else:
            self.fc1 = nn.Linear(dim, nhidden)
        self.fc2 = nn.Linear(nhidden, nhidden)
        self.fc3 = nn.Linear(nhidden, dim)
        self.actv = actv
        self.nfe = 0
        self.verbose = False

    def forward(self, t, z):
        # verbose表示为了解更多的，可以在中间增加更多的模型运行信息
        if self.verbose:
            print("Inside ODE function")
            print("z:", z)
            print("t:", t)
        cutoff = int(len(z) / 2)
        h = z[:cutoff]
        dh = z[cutoff:]
        k_reciprocal = torch.pow(t, 3 / 2) * torch.exp(-t / 2)
        m = (3 / 2 * torch.pow(t, 1 / 2) * torch.exp(-t / 2) - 1 / 2 * k_reciprocal) * h \
            + k_reciprocal * dh
        x = h * k_reciprocal
        if z.is_cuda:
            k_reciprocal = k_reciprocal.to(z.get_device())
        self.nfe += 1
        if self.time_dependent:
            # Shape (batch_size, 1)
            t_vec = torch.ones(x.shape[0], 1).to(x.get_device()) * t
            # Shape (batch_size, data_dim + 1)
            t_and_z = torch.cat([t_vec, h], 1)
            # Shape (batch_size, hidden_dim)
            out = self.fc1(t_and_z)
        else:
            out = self.fc1(h)
        out = self.elu(out)
        out = self.fc2(out)
        out = self.elu(out)
        actv_df = self.actv if self.modelname == "GNesterovNODE" else nn.Identity()
        out = actv_df(self.fc3(out))
        dm = - m - out - self.res * h
        if self.verbose:
            print("out:", out)
            print("m:", m)
            print("x:", x)
            print("dm:", dm)
        if self.modelname in ("GNesterovNODE"):
            # actv_v = nn.Tanh()
            actv_v = self.actv
        else:
            actv_v = nn.Identity()
        return torch.cat((actv_v(m), dm))


# 高分辨率Nesterovnode模型
class HighNesterovNODEfunc(nn.Module):
    def __init__(self, dim, nhidden, time_dependent=True, modelname=None, xi=None, actv=nn.Tanh(), corr=-100.0,
                 corrf=True, gamma_guess=-3.0, gamma_act='sigmoid', siga=0.2, sign=-1.0, actv_m=None, actv_dm=None,
                 actv_df=None, args=None):
        super(HighNesterovNODEfunc, self).__init__()
        # 当前的代码暂时设置这几个超参数为固定参数，不进行学习操作
        self.gamma = torch.tensor([gamma_guess]).to(args.gpu)
        self.gammaact = nn.Sigmoid() if gamma_act == 'sigmoid' else gamma_act
        self.corr = torch.tensor([corr]).to(args.gpu)
        # 考虑是否需要学习
        self.siga = torch.tensor([siga]).to(args.gpu)
        self.sp = nn.Softplus()
        self.sign = sign  # Sign of df

        self.modelname = modelname
        self.time_dependent = time_dependent
        # only have residual for generalized model
        self.res = 0.0 if xi is None else xi
        self.elu = nn.ELU(inplace=False)
        if self.time_dependent:
            self.fc1 = nn.Linear(dim + 1, nhidden)
        else:
            self.fc1 = nn.Linear(dim, nhidden)
        self.fc2 = nn.Linear(nhidden, nhidden)
        self.fc3 = nn.Linear(nhidden, dim)
        self.actv = actv
        self.nfe = 0
        self.verbose = False

        # 通用的兜底激活函数
        self.sp = nn.Softplus()
        self.sign = sign  # Sign of df
        self.actv_h = nn.Identity() if actv is None else actv  # Activation for dh, GHBNODE only
        self.actv_m = nn.Identity() if actv_m is None else actv_m  # Activation for dh, GNNODE only
        self.actv_dm = nn.Identity() if actv_dm is None else actv_dm  # Activation for dh, GNNODE only
        self.actv_df = nn.Identity() if actv_df is None else actv_df  # Activation for df, GNNODE only

    def forward(self, t, z):
        self.nfe += 1
        if self.verbose:
            print("Inside ODE function")
            print("z:", z)
            print("t:", t)

        # 对于z进行切分为两个升维的变量 分别是h和dh
        cutoff = int(len(z) / 2)
        h = z[:cutoff]
        m = z[cutoff:]

        # 计算经过了神经网络拟合的f(h,t,sita)之后的数值df
        if self.time_dependent:
            # Shape (batch_size, 1)
            t_vec = torch.ones(h.shape[0], 1).to(z.get_device()) * t
            # Shape (batch_size, data_dim + 1)
            t_and_z = torch.cat([t_vec, h], 1)
            # Shape (batch_size, hidden_dim)
            out = self.fc1(t_and_z)
        else:
            out = self.fc1(h)
        out = self.elu(out)
        out = self.fc2(out)
        out = self.elu(out)
        df = self.fc3(out)
        # 对于经过神经网络传递运行之后的数据进行对应的激活函数泛化操作，对于其数值的上下界进行一定的限制
        df = self.actv_df(df)
        dh = self.actv_h(-m) + self.siga * df
        alfa = self.gammaact(self.gamma)
        dm = self.actv_dm(- alfa * m + (1 - alfa * self.siga) * df - self.sp(self.corr) * h)

        if self.verbose:
            print("out:", out)
            print("m:", m)
            print("dm:", dm)

        # 合并全量的二阶变量参数
        if self.modelname in ("GHighNesterovNODE"):
            # actv_v = nn.Tanh()
            actv_v = self.actv
        else:
            actv_v = nn.Identity()
        return torch.cat((actv_v(dh), dm))


# hbnode的pid梯度形式加速版本
class PIDHBNODEfunc(nn.Module):
    def __init__(self, dim, nhidden, time_dependent=True, modelname=None, xi=None, actv=nn.Tanh(), kp=2, ki=1.5, kd=2,
                 general_type=3, actv_h=None, gamma_guess=-3.0, gamma_act='sigmoid', corr=-100, corrf=True, sign=1,
                 actv_m=None, actv_dm=None, actv_df=None):
        super(PIDHBNODEfunc, self).__init__()

        # 在这里仍然使用通用的网络结构模型，但是对于输出的数值的数目需要做一定的扩展，从而方便进行h,m,v的拆分
        self.modelname = modelname
        self.time_dependent = time_dependent
        # only have residual for generalized model
        self.res = 0.0 if xi is None else xi
        self.elu = nn.ELU(inplace=False)
        if self.time_dependent:
            self.fc1 = nn.Linear(dim + 1, nhidden)
        else:
            self.fc1 = nn.Linear(dim, nhidden)
        self.fc2 = nn.Linear(nhidden, nhidden)
        self.fc3 = nn.Linear(nhidden, dim)
        self.actv = actv
        self.nfe = 0
        self.verbose = False

        # 对于pidhbnode需要的激活函数类型以及特定的pid控制器的参数进行初始化的设置
        self.gt = general_type
        self.gamma = nn.Parameter(torch.Tensor([gamma_guess]))
        self.gammaact = nn.Sigmoid() if gamma_act == 'sigmoid' else gamma_act
        self.kp = nn.Parameter(torch.Tensor([kp]))
        self.ki = nn.Parameter(torch.Tensor([ki]))
        self.kd = nn.Parameter(torch.Tensor([kd]))
        self.sp = nn.Softplus()
        self.sign = sign  # Sign of df
        self.actv_h = nn.Identity() if actv_h is None else actv_h  # Activation for dh, GHBNODE only
        self.actv_m = nn.Identity() if actv_m is None else actv_m  # Activation for dh, GNNODE only
        self.actv_dm = nn.Identity() if actv_dm is None else actv_dm  # Activation for dh, GNNODE only
        self.actv_df = nn.Identity() if actv_df is None else actv_df  # Activation for df, GNNODE only

    def forward(self, t, z):
        self.nfe += 1
        if self.verbose:
            print("Inside ODE function")
            print("z:", z)
            print("t:", t)

        # 对于z变量进行拆分，并且拆分成为三个变量，分别为h,m,v
        h, m, v = torch.tensor_split(z, 3, dim=0)
        dh = self.actv_h(m)

        # 对于经过了神经网络结构的df进行计算，这里都汇总到一个函数当中，可以考虑后续拆解出来
        if self.time_dependent:
            # Shape (batch_size, 1)
            t_vec = torch.ones(h.shape[0], 1).to(z.get_device()) * t
            # Shape (batch_size, data_dim + 1)
            t_and_z = torch.cat([t_vec, h], 1)
            # Shape (batch_size, hidden_dim)
            out = self.fc1(t_and_z)
        else:
            out = self.fc1(h)
        out = self.elu(out)
        out = self.fc2(out)
        out = self.elu(out)
        df = self.fc3(out)
        # 对于经过神经网络传递运行之后的数据进行对应的激活函数泛化操作，对于其数值的上下界进行一定的限制

        # 根据指定的不同的泛化类型选择对应的激活函数泛化形式
        if self.gt == 1:
            # 泛化1
            dm = self.actv_h(-self.kp * h - (self.gammaact(self.gamma) + self.kd) * m - self.ki * v) + df
            dv = h
        elif self.gt == 2:
            # 泛化 2
            dm = -self.kp * h - (self.gammaact(self.gamma) + self.kd) * m - self.ki * v + df
            dv = self.actv_h(v)
        elif self.gt == 3:
            # 泛化 3
            df = self.actv_df(df)
            dm = -self.kp * h - (self.gammaact(self.gamma) + self.kd) * m - self.ki * v + df
            dv = self.actv_h(v)
        elif self.gt == 4:
            # 泛化4
            dm = -self.kp * h - (self.gammaact(self.gamma) + self.kd) * m - self.ki * v + df
            dv = h
        elif self.gt == 5:
            # 泛化5
            dm = self.actv_h(-self.kp * h - (self.gammaact(self.gamma) + self.kd) * m - self.ki * v + df)
            dv = self.actv_h(v)

        if self.verbose:
            print("out:", out)
            print("m:", m)
            print("dm:", dm)
            print("dv", dv)

        # 合并全量的二阶变量参数
        if self.modelname in ("PIDGHBNODE"):
            # actv_v = nn.Tanh()
            actv_v = self.actv
        else:
            actv_v = nn.Identity()
        return torch.cat((actv_v(dh), dm, dv))