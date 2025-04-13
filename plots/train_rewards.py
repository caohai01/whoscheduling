import matplotlib.pyplot as plt
import numpy as np
# 从文train_his_0601140000_5s_100000.csv读取avg_reward_train, max_reward_train和min_reward_tran列
data = np.loadtxt(r'F:\etc\data\train_his_0601140000_5s_30000.csv', delimiter=',', skiprows=1)

# 生成示例数据
x = data[:, 0] # X轴数据（1到10）
mean = data[:, 1]  # 均值数据
min_val = data[:,2]  # 最小值数组
max_val = data[:,3]     # 最大值数组

# 创建画布和坐标轴
plt.figure(figsize=(12, 10), dpi=100)
plt.grid(True, linestyle='--', alpha=0.4)

# 绘制三条主曲线
plt.plot(x, max_val, color='#FF6B6B', linewidth=1.5, linestyle='--', label='Max Reward')
plt.plot(x, mean,   color='#4D96FF', linewidth=2.5,  markersize=6, label='Mean Reward')
plt.plot(x, min_val, color='#6BCB77', linewidth=1.5, linestyle='--', label='Min Reward')

# 填充颜色区域
plt.fill_between(x, mean, max_val, color='#FFE08E', alpha=0.3)
plt.fill_between(x, mean, min_val, color='#96D3A7', alpha=0.3)

# 配置坐标轴
# plt.title("Reward Trend During Reinforcement Training", fontsize=14, pad=15)
plt.xlabel("Epoch", fontsize=12)
plt.ylabel("Total Score", fontsize=12)

# 设置X轴刻度（每15个数据显示主刻度）
#plt.xticks(np.arange(min(x), max(x)+1, 15))  # 显示0,15,30等主要刻度
plt.xlim(0, 30000)  # 扩展显示范围

# 添加图例和样式优化
plt.legend(loc='lower right', frameon=True, framealpha=0.9)
plt.tight_layout()

# 显示图表
plt.show()