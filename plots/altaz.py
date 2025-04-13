import json

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
plt.rcParams['font.sans-serif'] = ['Times New Roman']  # 设置中文显示
# plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文显示
plt.rcParams['axes.unicode_minus'] = False    # 正负号显示
fig, axes = plt.subplots(2, 2, figsize=(10, 10), subplot_kw={'projection': 'polar'})
data_files = {
    'Random': '../plans/plan_Random_202406011400_2.0_31_14.879_bl.csv',
    'Greedy': '../plans/plan_Greedy_202406011400_2.0_39_21.409_bl.csv',
    'RL-GR': '../plans/plan_RL_202406011400_2.0_48_25.319_bl.csv',
    'RL-BS': '../plans/plan_RL_202406011400_2.0_48_25.656_bl.csv',
}

ai = 0
for ad in data_files:
    with open(data_files[ad]) as f:
        json_text = '[' + f.read()[:-2] + ']'
        items = json.loads(json_text)
    azimuths = [0]
    altitudes = [90]
    scores = [-float('nan')]
    for item in items:
        if 'constraints_value' not in item:
            continue
        azimuths.append(item['azimuth'])
        altitudes.append(item['altitude'])
        scores.append(item['constraints_value'])
    zenith = 90 - np.array(altitudes)
    theta = np.deg2rad(azimuths)
    score = np.array(scores)

    # 创建极坐标图
    ax = axes[ai // 2, ai % 2]
    ai += 1
    # 绘制散点图
    scatter = ax.scatter(theta, zenith, c=scores, cmap='viridis', alpha=0.8, s=40, zorder=0)

    # # 按方位角排序后连接散点
    # sorted_indices = np.argsort(azimuth)  # 获取方位角排序索引
    # theta_sorted = theta[sorted_indices]  # 排序后的角度
    # r_sorted = zenith[sorted_indices]          # 对应排序后的半径

    # # 绘制连接线（红色半透明折线）
    # ax.plot(theta, zenith,
    #         color='red',
    #         linestyle='-',
    #         linewidth=2,
    #         alpha=0.5,
    #         zorder=1)

    # 循环绘制带箭头的线段
    for i in range(len(theta) - 1):
        # 获取相邻点坐标
        start_theta = theta[i]
        start_r = zenith[i]
        end_theta = theta[i + 1]
        end_r = zenith[i + 1]

        # 绘制基础线段
        # ax.plot([start_theta, end_theta], [start_r, end_r], color='red', alpha=0.5, linewidth=1, zorder=1)

        # 添加箭头装饰
        dist = np.sqrt(start_r ** 2 + end_r ** 2 - 2 * start_r * end_r * np.cos(start_theta - end_theta))
        if dist < 10.0:  # 距离过近的箭头不绘制
            arrow = FancyArrowPatch(
                posA=(start_theta, start_r),  # 起点笛卡尔坐标
                posB=(end_theta, end_r),  # 终点笛卡尔坐标
                arrowstyle='-',
                color='red',
                connectionstyle='arc3,rad=0.5',  # 弧度
                mutation_scale=0,  # 箭头大小
                shrinkA=0, shrinkB=3,  # 不缩短线段
                alpha=0.5,
                transform=ax.transData  # 使用数据坐标系
            )
            ax.add_patch(arrow)
            # ax.plot([start_theta, end_theta], [start_r, end_r], color='red', alpha=0.5, linewidth=1, zorder=1)
        else:
            arrow = FancyArrowPatch(
                posA=(start_theta, start_r),  # 起点笛卡尔坐标
                posB=(end_theta, end_r),  # 终点笛卡尔坐标
                arrowstyle='->',
                color='red',
                connectionstyle='arc3,rad=0.2',  # 弧度
                mutation_scale=10,  # 箭头大小
                shrinkA=0, shrinkB=3,  # 不缩短线段
                alpha=0.8,
                transform=ax.transData  # 使用数据坐标系
            )
            ax.add_patch(arrow)

    # 坐标系设置
    ax.set_rlim(0, 75)  # 径向范围对应高度角 90°（中心）到 10°（边缘）
    ax.set_theta_zero_location('E')  # 0度方位角指向正东（右侧）
    ax.set_theta_direction(1)  # 逆时针方向增加角度
    score[0] = 0
    ax.set_xlabel(f"{ad}({score.sum():.3f})", fontsize=12, labelpad=-40)
    # 自定义径向刻度标签
    ax.set_rgrids(
        radii=[0, 15, 30, 45, 60, 75],
        labels=['90°', '75°', '60°', '45°', '30°', '15°'],
        fontsize=10
    )
    ax.grid(True, linestyle='--', alpha=0.7)

    # 添加方向标注
    # ax.text(np.deg2rad(0), 85, '东', ha='center', va='center')
    # ax.text(np.deg2rad(90), 85, '北', ha='center', va='center')
    # ax.text(np.deg2rad(180), 85, '西', ha='center', va='center')
    # ax.text(np.deg2rad(270), 85, '南', ha='center', va='center')
#添加颜色条
# 把颜色条放在最右侧，占整个图的0.8，四个子图共享一个颜色条
#cbar_ax = fig.add_axes((0.92, 0.03, 0.03, 0.94))
#cbar = fig.colorbar(scatter, cax=cbar_ax, pad=0.1, orientation='vertical')
cbar_ax = fig.add_axes((0.02, 0.04, 0.96, 0.015))
cbar = fig.colorbar(scatter, cax=cbar_ax, pad=0.1, orientation='horizontal')
cbar.set_label('Score', fontsize=12, labelpad=-6)

# 装饰图形
# plt.title("", fontsize=14, pad=20)
# plt.tight_layout()
plt.show()