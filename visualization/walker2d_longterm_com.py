import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 加载长时间运行的walk2d数据
names = ["NODE", "ANODE",
         "SONODE",
         "HBNODE", "GHBNODE",
         "NesterovNODE", "GNesterovNODE",
         "PIDGHBNODE", "PIDHBNODE"]
file_names = ["../output/walker2d/walker_NODE_rnn_9929.csv", "../output/walker2d/walker_ANODE_rnn_10019.csv",
              "../output/walker2d/walker_SONODE_rnn_9861.csv",
              "../output/walker2d/walker_HBNODE_rnn_10099_12.csv", "../output/walker2d/walker_GHBNODE_rnn_10099_12.csv",
              "../output/walker2d/walker_NesterovNODE_rnn_10098.csv",
              "../output/walker2d/walker_GNesterovNODE_rnn_10098_12.csv",
              "../output/walker2d/walker_PIDGHBNODE_rnn_10055.csv",
              "../output/walker2d/walker_PIDHBNODE_rnn_10055.csv"]
df_names = {}
for i in range(len(names)):
    temp_df = pd.read_csv(file_names[i])
    attr_names = [col for col in temp_df.columns if 'grad' in col]
    df_names[names[i]] = temp_df[attr_names]

df_names_temp = {}
names = ["NODE", "PIDGHBNODE", "GHBNODE", "PIDHBNODE"]
for i, name in enumerate(names):
    df_names_temp[name] = df_names[name].drop(columns=['grad_64'])
    df_names_temp[name] = df_names_temp[name].rename(columns=lambda x: int(x[5:]))
    df_names_temp[name] = df_names_temp[name].reindex(sorted(df_names_temp[name].columns), axis=1)
    df_names_temp[name] = df_names_temp[name].iloc[:, ::-1]

# 对于采集的数据进行记录并且按照数值的大小输出水位图
font = {'size': 50}
plt.rc('font', **font)

fig = plt.figure(figsize=(30, 25))

ax1 = fig.add_subplot(221)
# ax1.set_title("NODE-RNN", fontsize=70)
# ax1.set_xlabel("Epoch", fontsize=70)
ax1.set_title("NODE-RNN", fontsize=70, fontproperties=font)
ax1.set_ylabel("T - t", fontsize=70)

# pcm = ax.pcolormesh(x, y, Z, vmin=-1., vmax=1., cmap='RdBu_r')
# 其中x,y表示的对应的二维坐标，z则是对应的坐标点上的数值
# ax1.pcolormesh就是在vmin到vmax之间的数值进行颜色的输出对比 可视化变化
# mesh1 = ax1.pcolormesh(df_names_temp["NODE"].values.T, cmap='viridis', vmin=0, vmax=0.03)
mesh1 = ax1.pcolormesh(df_names_temp["NODE"].values.T, cmap='viridis_r', vmin=0.01, vmax=0.06, alpha=0.8, )

ax2 = fig.add_subplot(222)
ax2.set_title("GHBNODE-RNN", fontsize=70)
# ax2.set_xlabel("Epoch", fontsize=70)
# ax2.set_ylabel("T - t", fontsize=70)
mesh2 = ax2.pcolormesh(df_names_temp["GHBNODE"].values.T, cmap='viridis_r', vmin=0.01, vmax=0.06, alpha=0.8, )

ax3 = fig.add_subplot(223)
ax3.set_title("PIDNODE-RNN", fontsize=70)
ax3.set_xlabel("Epoch", fontsize=70)
ax3.set_ylabel("T - t", fontsize=70)
mesh3 = ax3.pcolormesh(df_names_temp["PIDHBNODE"].values.T, cmap='viridis_r', vmin=0.01, vmax=0.06, alpha=0.8, )

ax4 = fig.add_subplot(224)
ax4.set_title("GPIDNODE-RNN", fontsize=70)
ax4.set_xlabel("Epoch", fontsize=70)
# ax4.set_ylabel("T - t", fontsize=70)
mesh4 = ax4.pcolormesh(df_names_temp["PIDGHBNODE"].values.T, cmap='viridis_r', vmin=0.01, vmax=0.06, alpha=0.8, )

# 需要补充新的比较 PIDNODE GPIDNODE HighNesterovNODE GHighNesterovNODE

fig.colorbar(mesh1, ax=ax1)
fig.colorbar(mesh2, ax=ax2)
fig.colorbar(mesh3, ax=ax3)
fig.colorbar(mesh4, ax=ax4)
fig.tight_layout()

# plt.savefig(f"walker2d_longterm.pdf", bbox_inches='tight', pad_inches=0)
plt.savefig(f"walker2d_longterm1.jpg", bbox_inches='tight', dpi=500)
plt.show()
