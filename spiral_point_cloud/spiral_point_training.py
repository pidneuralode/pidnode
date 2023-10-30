#!/usr/bin/env python
# coding: utf-8
import math
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
from torch.utils.data import DataLoader
from torch import nn
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import argparse

torch.set_default_tensor_type(torch.DoubleTensor)

from point_cloud.experiments.dataloaders import SpiralCircle
from visualization.plots import get_feature_history
from training import Trainer
from visualization.plots import multi_feature_plt
from models import initial_velocity, ODEBlock, Decoder, count_parameters, pidhbnode_initial_velocity
from ode_functions import NODEfunc, SONODEfunc, NesterovNODEfunc, HighNesterovNODEfunc, PIDHBNODEfunc

parser = argparse.ArgumentParser()
parser.add_argument('--tol', type=float, default=1e-7)
parser.add_argument('--num-epochs', type=int, default=200)
parser.add_argument('--visualize-results', default=1, type=int, choices=[0, 1])
parser.add_argument('--num-runs', help='Number of independent runs per model', default=50)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--output-directory', default='point_cloud')
parser.add_argument('--batch-size', default=50, type=int)
parser.add_argument('--names', nargs='+', help='List of models to run', default=["pidhbnode"])
parser.add_argument('--xi', help='Value of the "xi" term in generalized model', type=float, default=0.5)
parser.add_argument('--visualize-std', action="store_true", help='Whether to plot one std from the mean')
parser.add_argument('--visualize-features', action="store_true",
                    help='Whether to visualize features progression through training of a single run and not record anything else')
parser.add_argument('--loss', default="cross", help='loss for training')
# 支持对于相同的模型引入不同的参数
parser.add_argument('--extra-name', help='the extra name or parameter of model', default='')
# 支持对于PIDHBNODE引入不同的pid参数，方便找到最合适的参数类型
parser.add_argument('--kp', type=float, default=1.)
parser.add_argument('--ki', type=float, default=2.)
parser.add_argument('--kd', type=float, default=1.5)

args = parser.parse_args()

# 本次实验主要是针对于点云的多分类案例，其中外层为一个自定义区间的圆环，内层是按照螺线分布的几个曲线区域，通过这样更加复杂的点云分布，
# 可以进一步的测试当前模型对于其信息特征的提取表征能力
# 这里表示数据的维度，本次实验仍然采用的是二维平面的点进行区分的操作
data_dim = 2
# updated the range to match HeavyBall exp settings
# 构建同心圆结构的数据的数据集合，当前设置为2维度的
data_spiral = SpiralCircle(3, (3 * math.pi, 3.4 * math.pi), 150, 240)
dataloader = DataLoader(data_spiral, batch_size=args.batch_size, shuffle=True)


class ODENet(nn.Module):
    def __init__(self, device, node_layer, prediction_layer, augment_dim=0):
        super().__init__()
        self.odefunc = node_layer.to(device)
        self.prediction_layer = prediction_layer.to(device)
        self.augment_dim = augment_dim
        self.device = device

    def forward(self, x, return_features=False):
        if augment_dim > 0:
            # Add augmentation
            aug = torch.zeros(x.shape[0], augment_dim).to(self.device)
            # Shape (batch_size, data_dim + augment_dim)
            x_aug = torch.cat([x, aug], 1)
            x = x_aug
        features = self.odefunc(x)
        pred = self.prediction_layer(features)
        if return_features:
            return features, pred
        return pred


device = torch.device(f'cuda:{args.gpu}')
output_directory = f"output/{args.output_directory}"
full_names = ["node", "anode", "sonode", "hbnode", "ghbnode", "nesterovnode", "gnesterovnode", "pidhbnode",
              "pidghbnode",
              "high_nesterovnode", "ghigh_nesterov"]
names = full_names if args.names is None else args.names
alt_names = ["NODE", "ANODE", "SONODE", "HBNODE", "GHBNODE", "NesterovNODE", "GNesterovNODE", "PIDHBNODE", "PIDGHBNODE",
             "HighNesterovNODE", "GHighNesterovNODE"]
all_histories = []
num_epochs = args.num_epochs
tol = args.tol
t0 = 1
tN = 2
tanh = nn.Tanh()
hard_tanh_1 = nn.Hardtanh(min_val=-1, max_val=1, inplace=False)
hard_tanh_5 = nn.Hardtanh(min_val=-5, max_val=5, inplace=False)
hard_tanh_half = nn.Hardtanh(min_val=-0.5, max_val=0.5, inplace=False)
for i, name in enumerate(names):
    print("Model:", name)
    histories_file_path = f"{output_directory}/{name}_histories.pkl"
    if os.path.exists(histories_file_path) and not args.visualize_features:
        print("Histories log exists. Skipping run for this model!")
        with open(histories_file_path, "rb") as f:
            model_histories = pickle.load(f)
    else:
        num_runs = args.num_runs
        model_histories = {
            "epoch_loss_history": [],
            "epoch_nfe_history": [],
            "epoch_bnfe_history": []
        }
        for run_idx in range(num_runs):
            print(f"Run #: {run_idx + 1}/{num_runs}")
            augment_dim = 0
            if name == "node":
                nhid = 20
                feature_layers = [ODEBlock(
                    NODEfunc(data_dim, nhid), t0, tN, tol=tol), Decoder(data_dim, 3)]
            elif name == "anode":
                nhid = 20
                augment_dim = 1
                feature_layers = [ODEBlock(NODEfunc(
                    data_dim, nhid, augment_dim=augment_dim), t0, tN, tol=tol), Decoder(data_dim + augment_dim, 3)]
            elif name == "sonode":
                nhid = 13
                feature_layers = [initial_velocity(data_dim, nhid), ODEBlock(SONODEfunc(
                    data_dim, nhid, modelname="SONODE"), t0, tN, tol=tol, half=True), Decoder(data_dim, 3)]
            elif name == "hbnode":
                nhid = 14
                feature_layers = [initial_velocity(data_dim, nhid), ODEBlock(SONODEfunc(
                    data_dim, nhid, modelname="HBNODE"), t0, tN, tol=tol, half=True), Decoder(data_dim, 3)]
            elif name == "ghbnode":
                nhid = 14
                feature_layers = [initial_velocity(data_dim, nhid), ODEBlock(SONODEfunc(
                    data_dim, nhid, modelname="GHBNODE", actv=hard_tanh_half), t0, tN, tol=tol, half=True),
                                  Decoder(data_dim, 3)]
            elif name == "nesterovnode":
                nhid = 14
                feature_layers = [initial_velocity(data_dim, nhid), ODEBlock(NesterovNODEfunc(
                    data_dim, nhid, modelname="NesterovNODE"), t0, tN, tol=tol, half=True, nesterov_algebraic=True),
                                  Decoder(data_dim, 3)]
            elif name == "gnesterovnode":
                nhid = 14
                feature_layers = [initial_velocity(data_dim, nhid), ODEBlock(NesterovNODEfunc(
                    data_dim, nhid, modelname="GNesterovNODE", xi=args.xi, actv=hard_tanh_5), 1, 2, tol=tol, half=True,
                    nesterov_algebraic=True, actv_k=hard_tanh_5, use_momentum=False), Decoder(data_dim, 3)]
            elif name == 'high_nesterovnode':
                # 初步给出高分辨率模型
                nhid = 14
                feature_layers = [initial_velocity(data_dim, nhid), ODEBlock(HighNesterovNODEfunc(
                    data_dim, nhid, modelname="NesterovNODE", args=args), t0, tN, tol=tol, half=True,
                    nesterov_algebraic=False),
                                  Decoder(data_dim, 3)]
            elif name == 'ghigh_nesterovnode':
                # 初步给出高分辨率模型
                nhid = 14
                feature_layers = [initial_velocity(data_dim, nhid), ODEBlock(HighNesterovNODEfunc(
                    data_dim, nhid, modelname="GNesterovNODE", xi=args.xi, actv=hard_tanh_5, args=args), 1, 2, tol=tol,
                    half=True,
                    nesterov_algebraic=False, actv_k=hard_tanh_5, use_momentum=False), Decoder(data_dim, 3)]
            elif name == 'pidhbnode':
                # 初步给出pidnode模型
                nhid = 14
                feature_layers = [pidhbnode_initial_velocity(data_dim, nhid, args.gpu), ODEBlock(PIDHBNODEfunc(
                    data_dim, nhid, kp=args.kp, ki=args.ki, kd=args.kd, modelname="PIDHBNODE"),
                    t0, tN, tol=tol, one_third=True, nesterov_algebraic=False),
                                  Decoder(data_dim, 3)]
            elif name == 'pidghbnode':
                # 初步给出泛化的pidnode模型
                nhid = 14
                feature_layers = [pidhbnode_initial_velocity(data_dim, nhid, args.gpu), ODEBlock(PIDHBNODEfunc(
                    data_dim, nhid, kp=args.kp, ki=args.ki, kd=args.kd, modelname="PIDGHBNODE", xi=args.xi,
                    actv=hard_tanh_5, actv_h=hard_tanh_1, actv_df=hard_tanh_1),
                    t0, tN, tol=tol,
                    one_third=True, nesterov_algebraic=False, actv_k=hard_tanh_5, use_momentum=False),
                                  Decoder(data_dim, 3)]

            model = ODENet(device, nn.Sequential(*feature_layers[:-1]), feature_layers[-1],
                           augment_dim=augment_dim).to(args.gpu)
            if run_idx == 0:
                print("# Parameters:", count_parameters(model))
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

            trainer = Trainer(model, optimizer, device, loss=args.loss, verbose=True, print_freq=5)

            if args.visualize_features:
                # single_feature_plt(inputs, targets)
                viz_feature_file_path = f"{output_directory}/viz_features/{args.extra_name}{name}_feature_viz.pkl"
                if os.path.exists(viz_feature_file_path):
                    print("Feature viz log exists. Skipping run for this model!")
                    with open(viz_feature_file_path, "rb") as f:
                        log = pickle.load(f)
                        inputs = log["inputs"]
                        targets = log["targets"]
                        feature_history = log["feature_history"]
                else:
                    # Visualize a batch of data (use a large batch size for visualization)
                    dataloader_viz = DataLoader(data_spiral, batch_size=args.batch_size, shuffle=True)
                    for inputs, targets in dataloader_viz:
                        break
                    print("Feature viz log does not exist. Let's run this model!")
                    inputs = inputs.to(device)
                    targets = targets.to(device)
                    print("# Epochs:", num_epochs)
                    feature_history = get_feature_history(trainer, dataloader, inputs,
                                                          targets, num_epochs)
                    print("Logging feature histories to binary files!")
                    with open(viz_feature_file_path, "wb") as f:
                        log = {
                            "inputs": inputs,
                            "targets": targets,
                            "feature_history": feature_history
                        }
                        pickle.dump(log, f)

                number_visualization = 1
                step = (num_epochs - 1) // number_visualization
                print("Number of features step:", len(feature_history))
                print("Step:", step)
                print("Feature dimensions:", feature_history[0].shape[-1])
                if feature_history[0].shape[-1] == 2:
                    feature_history = [feature.detach().cpu() for feature in feature_history]
                elif feature_history[0].shape[-1] > 2:
                    feature_history = [feature[:, :2].detach().cpu() for feature in feature_history]
                targets = targets.detach().cpu()
                alt_name = alt_names[full_names.index(name)]
                multi_feature_plt(feature_history[::step], targets,
                                  f"{output_directory}/viz_features/{args.extra_name}{name}.pdf",
                                  name=alt_name)
                break
            else:
                # If we don't record feature evolution, simply train model
                trainer.train(dataloader, num_epochs)
                run_histories = trainer.histories
                for attr in ["loss", "nfe", "bnfe"]:
                    attr_key = f"epoch_{attr}_history"
                    model_histories[attr_key].append(run_histories[attr_key])

        if not args.visualize_features:
            # save history to binary file
            print("Logging histories to binary files!")
            with open(histories_file_path, "wb") as f:
                pickle.dump(model_histories, f)

    if not args.visualize_features:
        all_histories.append(model_histories)

if args.visualize_results == 1 and not args.visualize_features:
    histories_attr = [
        "epoch_nfe_history",
        "epoch_bnfe_history",
        "epoch_loss_history",
    ]
    display_attr = ["NFEs", "NFEs (backward)", "Loss"]
    save_names = ["nfe", "bnfe", "loss"]
    colors = [
        "mediumvioletred",
        "red",
        "deepskyblue",
        "royalblue",
        "navy",
        "green",
        "darkorange",
        'blue',
        'purple',
        'yellow',
        'orange',
    ]
    line_styles = [
        ':',
        '--',
        '-.',
        '-.',
        '-.',
        '-',
        '-',
        '-*',
        '-*',
        '*',
        '*',
    ]
    line_widths = [
        5,
        5,
        5,
        5,
        5,
        5,
        5,
        7,
        7,
        7,
        7,
    ]
    plot_std = args.visualize_std
    font = {'size': 40}
    plt.rc('font', **font)
    fig = plt.figure(figsize=(25, 7))
    gs = gridspec.GridSpec(1, 3, wspace=0.4)
    ax1 = plt.subplot(gs[0, 0])
    ax2 = plt.subplot(gs[0, 1])
    ax3 = plt.subplot(gs[0, 2])
    axes = (ax1, ax2, ax3)
    for i, save_name in enumerate(save_names):
        for j, name in enumerate(names):
            model_histories = all_histories[j]
            model_attrs = np.array(model_histories[histories_attr[i]])
            epochs = np.arange(model_attrs.shape[1])
            mean_attrs = model_attrs.mean(axis=0)
            # plot the mean line
            axes[i].plot(epochs, mean_attrs, line_styles[j], linewidth=line_widths[j], label=alt_names[j],
                         color=colors[j])
        axes[i].grid()
        axes[i].set(xlabel="Epoch", ylabel=display_attr[i])

    axbox = ax3.get_position()
    _ = plt.legend(bbox_to_anchor=(0.5, axbox.y0 - 0.50), loc="lower center",
                   bbox_transform=fig.transFigure, ncol=4, handletextpad=0.5, columnspacing=0.6, borderpad=0.3)
    plt.savefig(f"{output_directory}/viz_train/point_cloud.pdf", transparent=True, bbox_inches='tight', pad_inches=0)
