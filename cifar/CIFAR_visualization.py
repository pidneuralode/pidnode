import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

font = {'size': 40}
plt.rc('font', **font)

tolerance = "1e-05"
names = ["node", "anode", "sonode", "hbnode", "ghbnode", "pidnode", "gpidnode"]
alt_names = ["NODE", "ANODE", "SONODE", "HBNODE", "GHBNODE", "PIDNODE", "GPIDNODE"]
df_names = {}
for name in names:
    filepath = f"output\\cifar\\{name}\\{name}_{tolerance}_.csv"
    print("filepath:", filepath)
    # if name == "ghbnode":
    # 	filepath = f"../imgdat/1_2/backup/{name}_{tolerance}.csv"
    df = pd.read_csv(filepath, header=None,
                     names=["iter", "loss", "acc", "totalnfe", "forwardnfe", "time/iter", "time_elapsed"])
    df["train/test"] = np.where(pd.isnull(df["forwardnfe"]), "test", "train")
    df["backwardnfe"] = np.where(df["train/test"] == "test", 0, df["totalnfe"] - df["forwardnfe"])
    df["forwardnfe"] = np.where(df["train/test"] == "test", df["totalnfe"], df["forwardnfe"])
    df_names[name] = df

print(df_names[names[-1]].head(20))

colors = [
    "mediumvioletred",
    "red",
    "deepskyblue",
    "royalblue",
    "navy",
    "green",
    "darkorange",
    "purple",
    "lightgreen"
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
    9,
    9,
    9,
    9
]

fig = plt.figure(figsize=(26, 17))
gs = fig.add_gridspec(2, 6, hspace=0.25, wspace=1.2)
ax1 = fig.add_subplot(gs[0, :2])
ax2 = fig.add_subplot(gs[0, 2:4])
ax3 = fig.add_subplot(gs[0, 4:])
ax5 = fig.add_subplot(gs[1, 1:3])
ax6 = fig.add_subplot(gs[1, 3:5])
# ax4 = fig.add_subplot(gs[1, 0])
axes = (ax1, ax2, ax5)
# alt_attr_names = ["Train Forward NFEs", "Train Backward NFEs", "Train Time / Epoch (s)", "Train Loss"]
alt_attr_names = ["Train Forward NFEs", "Train Backward NFEs", "Train Loss"]
# for j, attribute in enumerate(["forwardnfe", "backwardnfe", "time/iter", "loss"]):
for j, attribute in enumerate(["forwardnfe", "backwardnfe", "loss"]):
    for i, name in enumerate(names):
        df_name = df_names[name]
        df_name_train = df_name.loc[df_name["train/test"] == "train"]
        attr_arr = df_name_train[attribute]
        iteration_arr = df_name_train["iter"]
        assert attr_arr.shape[0] <= 40  # max number of iterations
        axes[j].plot(iteration_arr, attr_arr, line_styles[i], linewidth=line_widths[i], color=colors[i],
                     label=alt_names[i])
    if attribute == "backwardnfe":
        axes[j].set_ylim((20, 300))
    if attribute == "time/iter":
        axes[j].set_ylim((200, 1200))
    axes[j].set(xlabel="Epoch", ylabel=f"{alt_attr_names[j]}")
    axes[j].grid()

axes = (ax3, ax6)
alt_attr_names = ["Test Forward NFEs", "Test Accuracy"]
for j, attribute in enumerate(["forwardnfe", "acc"]):
    for i, name in enumerate(names):
        df_name = df_names[name]
        df_name_train = df_name.loc[df_name["train/test"] == "test"]
        attr_arr = df_name_train[attribute]
        if attribute == "acc":
            print(f"Accuracy of {name}: {np.max(attr_arr)}")
        iteration_arr = df_name_train["iter"]
        assert attr_arr.shape[0] <= 40  # max number of iterations
        axes[j].plot(iteration_arr, attr_arr, line_styles[i], linewidth=line_widths[i], color=colors[i],
                     label=alt_names[i])
    if attribute == "acc":
        axes[j].set_xlim((0, 40))
        axes[j].set_ylim((0.45, 0.65))
    axes[j].set(xlabel="Epoch", ylabel=f"{alt_attr_names[j]}")
    axes[j].grid()

axbox = axes[-1].get_position()
_ = plt.legend(bbox_to_anchor=(0.5, axbox.y0 - 0.225), loc="lower center",
               bbox_transform=fig.transFigure, ncol=4, handletextpad=0.5, columnspacing=0.6, borderpad=0.3)
# plt.savefig(f"output/cifar.pdf", transparent=True, bbox_inches='tight', pad_inches=0)
plt.savefig(f"output/cifar.jpg", transparent=True, bbox_inches='tight', pad_inches=0, dpi=800)
plt.show()
