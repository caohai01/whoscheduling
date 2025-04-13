import matplotlib.dates as mdates
from datetime import datetime, timedelta
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
plt.rcParams['agg.path.chunksize'] = 10000
plt.rcParams['font.family'] = ['Times New Roman']
# 读取CSV文件
interval_seconds = 5
df = pd.read_csv(r'F:\etc\score_dist_bl2025012510-24.csv')
headers = df.columns.values
data = df.values
# 转换为时间
from_time = datetime(2025, 1, 25, 10, 0, 0)
# x_timestamps = [from_time + timedelta(seconds=5*x_val) for x_val in data[:, 0]]
# 优化时间转换（避免生成完整时间对象列表）
def x_to_timestamp(x_values):
    """将x值转换为matplotlib时间数值"""
    # 计算总秒数（向量化操作）
    total_seconds = x_values * interval_seconds
    # 转换为时间数值（matplotlib内部表示）
    return mdates.date2num(from_time) + total_seconds / (24*3600.0)

# 转换为时间数值数组



# 提取数据
fit_col_index = 13
x = data[:, 0]          # 第一列作为横轴
y_columns = data[:, 1:] # 其他所有列作为纵轴
sel_col = y_columns[:, fit_col_index]    # 第15列数据（索引14）
x_time = x_to_timestamp(x)

# 创建画布
plt.figure(figsize=(16, 9), dpi=100)
ax = plt.gca()

# 绘制所有其他列的曲线
for i in range(y_columns.shape[1]):
    lw = 2
    ap = 0.7
    if i == fit_col_index:
        lw=4
        ap=1
    valid_mask = ~np.isnan(y_columns[:, i])
    plt.plot(x_time[valid_mask], y_columns[valid_mask, i],
            linewidth=lw,
            alpha=ap,
            label=headers[i+1])  # 只标注前5个图例

# 二次曲线拟合第15列数据
valid_mask = ~np.isnan(sel_col)
valid_indices = np.where(valid_mask)[0]
# 定位有效数据范围
first_valid = valid_indices[0]
last_valid = valid_indices[-1]

coefficients = np.polyfit(x_time[first_valid:last_valid+1], sel_col[first_valid:last_valid + 1], 2)
poly_func = np.poly1d(coefficients)
x_fit = np.linspace(x_time[first_valid], x_time[last_valid], 100)
y_fit = poly_func(x_fit)

# 绘制拟合曲线
plt.plot(x_fit, y_fit,
         color='#88EE66',
         linewidth=3,
         linestyle='dashed',
         label=f'Quadratic Fit of ({headers[fit_col_index + 1]})')

# 显示方程文本
equation_text = (f'$S = ({coefficients[0]:.3})t^2 + ({coefficients[1]:.3e})t + ({coefficients[2]:.3})$\n'
                f'$R^2 = {np.corrcoef(sel_col[first_valid:last_valid + 1], poly_func(x_time[first_valid:last_valid + 1]))[0,1] ** 2:.3f}$')
plt.text(0.53, 0.93, equation_text,
        transform=ax.transAxes,
        fontsize=12)

# 增加竖线，标识月落时间
moon_set_time = from_time + timedelta(hours=10.534)
special_x = mdates.date2num(moon_set_time)  #
# 添加垂直参考线
plt.axvline(x=special_x,
           color='#FF6B6B',
           linestyle='--',
           linewidth=1.5,
           alpha=0.6,
           zorder=3)
# 添加旋转文本标注
plt.text(special_x + 0.002,  # 微调X位置防止重叠
        0.45,
        f'Moon Set({moon_set_time.strftime("%H:%M")})',
        rotation=90,
        va='top',
        ha='right',
        color='#6B6B6B',
        fontsize=10,
        bbox=dict(facecolor='white', alpha=0.8, edgecolor='#FF6B6B', boxstyle='round,pad=0.2'))
# 图表装饰
#plt.title(f"Score Curves over Time", fontsize=14, pad=15)
plt.xlabel('Time from 2025-01-25 10:00 to 22:30 (UTC)', fontsize=12)
plt.ylabel("Score", fontsize=12)
plt.grid(True, alpha=0.3)
plt.legend(loc='upper left', ncol=4)
# # 配置时间轴格式
ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=60))  # 每60分钟一个主刻度
ax.xaxis.set_minor_locator(mdates.MinuteLocator(interval=30))   # 每30分钟一个次刻度

# 智能坐标轴范围设置
padding = 0.02 * (x_time.max() - x_time.min())
plt.xlim(x_time.min(), x_time.max())
plt.ylim(0, 1)
# 优化显示
plt.tight_layout()
plt.show()