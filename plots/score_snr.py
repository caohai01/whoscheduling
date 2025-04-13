
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 读取数据
df = pd.read_csv(r'F:\etc\data2\20250123-7836\RL1400\stat_RL_202501231400_score-snr.csv')

# 准备参数
n = len(df)
x_index = np.arange(n)  # 数据点索引
bar_width = 0.5        # 单柱宽度
offset = bar_width/2    # 位置偏移量

# 创建画布和双坐标轴
fig, ax1 = plt.subplots(figsize=(15, 8))
ax2 = ax1.twinx()  # 创建右侧坐标轴

# 绘制左侧柱状图（Score）
bars1 = ax1.bar(x_index + offset,
               df['Score'],
               width=bar_width,
               color='#1f77b4',
               alpha=0.8,
               label='Score',
               edgecolor='white')

# 绘制右侧柱状图（SNR）
bars2 = ax2.bar(x_index + bar_width,
               df['SNR'],
               width=bar_width,
               color='#ff7f0e',
               alpha=0.8,
               label='SNR',
               edgecolor='white')

# 自动调整Y轴范围
def auto_axis_lim(axis, data, margin=0.1):
    """自动设置坐标轴范围并保持底部为0"""
    data_min = min(0, data.min())  # 考虑可能有负值
    data_max = data.max()
    axis_range = data_max - data_min
    axis.set_ylim(0, data_max + margin*axis_range)

auto_axis_lim(ax1, df['Score'])
auto_axis_lim(ax2, df['SNR'])
ax1.set_xlim(0, n)
# 坐标轴样式设置
ax1.set_xlabel('Target Index', fontsize=12, labelpad=10)
ax1.set_ylabel(f'Score', fontsize=12, labelpad=10)
ax2.set_ylabel(f'SNR', fontsize=12,  labelpad=10)

# 同步刻度颜色
#ax1.tick_params(axis='y', colors='#1f77b4')
#ax2.tick_params(axis='y', colors='#ff7f0e')

# X轴设置
plt.xticks(x_index[::5],          # 每5个显示一个标签
          labels=x_index[::5],  # 标签从1开始
          rotation=45,
          ha='right')

# 合并图例
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2,
         loc='upper right',
         fontsize=10,
         frameon=True)

# 添加网格线
ax1.grid(axis='y', alpha=0.5, linestyle='--')

# 图表装饰
plt.title(f'Score({df["Score"].sum():.3f}) : SNR({df["SNR"].sum():.3f})', fontsize=14, pad=20)
fig.tight_layout()

plt.show()