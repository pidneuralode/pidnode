import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

font = {'size': 30}

# 双星号传入map数值
plt.rc('font', **font)

# 开始预设node算法名称和作图名称
tolerance = "1e-05"
names = [
    "node",
    "anode",
    "sonode",
    "hbnode",
    "ghbnode",
    "nesterovnode",
    "gnesterovnode",
    "high_nesterovnode2",
    "ghigh_nesterovnode2"

]
alt_names = [
    "NODE",
    "ANODE",
    "SONODE",
    "HBNODE",
    "GHBNODE",
    "NesterovNODE",
    "GNesterovNODE",
    "HighNesterovNODE",
    "GHighNesterovNODE"
]
df_names = {}
for name in names:
    # filepath = f"../imgdat/1_2/{name}_{tolerance}.csv"
    filepath = f"C:/Users/29373/Desktop/NesterovNODE-main/NesterovNODE-main/mnist/imgdat/1_2/{name}_{tolerance}.csv"

    temp_df = pd.read_csv(filepath, header=None,
                          names=["model", "test#", "train/test", "iter", "loss", "acc", "forwardnfe", "backwardnfe",
                                 "time/iter", "time_elapsed"])
    df_names[name] = temp_df
df_names[names[-1]].head()

# 预设画图的颜色，线条等
colors = [
    "mediumvioletred",
    "red",
    "deepskyblue",
    "royalblue",
    "navy",
    "green",
    "darkorange",
    "yellow",
    "purple"
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
    '-*'
]
line_widths = [
    7,
    7,
    7,
    7,
    7,
    7,
    7,
    8,
    8
]

# 读取历史数据对于每一个模型进行nfe,acc,test loss等指标的构造 注意切片都是左闭右开的
fig = plt.figure(figsize=(25, 15))
gs = fig.add_gridspec(2, 6, hspace=0.25, wspace=1.2)
ax1 = fig.add_subplot(gs[0, :2])
ax2 = fig.add_subplot(gs[0, 2:4])
ax3 = fig.add_subplot(gs[0, 4:])
ax4 = fig.add_subplot(gs[1, 1:3])
ax5 = fig.add_subplot(gs[1, 3:5])
axes = (ax1, ax2, ax4)
height_width_ratio = "auto"
alt_attr_names = ["NFEs (forward)", "NFEs (backward)", "Loss"]
for j, attribute in enumerate(["forwardnfe", "backwardnfe", "loss"]):
    axes[j].set_aspect(height_width_ratio)
    for i, name in enumerate(names):
        # print(i, name)
        df_name = df_names[name]
        df_name_train = df_name.loc[df_name["train/test"] == "train"]
        attr_arr = df_name_train[attribute]
        iteration_arr = df_name_train["iter"]
        assert attr_arr.shape[0] <= 60  # max number of iterations
        axes[j].plot(iteration_arr, attr_arr, line_styles[i], linewidth=line_widths[i], color=colors[i],
                     label=alt_names[i])
    axes[j].set(xlabel="Epoch", ylabel=f"Train {alt_attr_names[j]}")
    # 对于指标展示图进行一定的坐标轴范围裁剪
    if attribute == "backwardnfe":
        axes[j].set_ylim([35, 5000])
    if attribute == "forwardnfe":
        axes[j].set_ylim([20, 5000])
    if attribute == "loss":
        axes[j].set_ylim(0.0, 4.0)
    axes[j].grid()
alt_attr_names = ["Accuracy", "NFEs (forward)"]
offset = 2
axes = (ax5, ax3)
for j, attribute in enumerate(["acc", "forwardnfe"]):
    axes[j].set_aspect(height_width_ratio)
    for i, name in enumerate(names):
        df_name = df_names[name]
        df_name_train = df_name.loc[df_name["train/test"] == "test"]
        attr_arr = df_name_train[attribute]
        if attribute == "acc":
            print(f"Accuracy of {name}: {np.max(attr_arr)}")
        iteration_arr = df_name_train["iter"]
        assert attr_arr.shape[0] <= 60  # max number of iterations
        axes[j].plot(iteration_arr, attr_arr, line_styles[i], color=colors[i], linewidth=line_widths[i],
                     label=alt_names[i])
    axes[j].set(xlabel="Epoch", ylabel=f"Test {alt_attr_names[j]}")
    if attribute == "acc":
        axes[j].set_xlim(5, 40)
        axes[j].set_ylim(0.96, 0.99)
    if attribute == "forwardnfe":
        axes[j].set_ylim(30, 90)
    # plt.legend()
    axes[j].grid()
axbox = axes[0].get_position()
l5 = plt.legend(bbox_to_anchor=(0.5, axbox.y0 - 0.22), loc="lower center",
                bbox_transform=fig.transFigure, ncol=3)
plt.savefig(f"mnist.pdf", transparent=True, bbox_inches='tight', pad_inches=0)
plt.show()

for i, name in enumerate(names):
    df_name = df_names[name]
    df_name_train = df_name.loc[df_name["train/test"] == "test"]
    attr_arr = df_name_train["acc"]
    print(f"Accuracy of {name}: {np.max(attr_arr)}")
