import torch
from astroplan import Observer
from astropy.coordinates import EarthLocation
from astropy.time import Time, TimeDelta
from astropy import units as u
from enum import Enum
from ETC import exposure_time_dicts

class StationName(Enum):
    """
    站点表示
    """
    WHOE = 1
    WHOW = 2
    MSGT = 3



station = StationName.WHOW

if station == StationName.WHOW:  # 威海西测站信息
    location = EarthLocation.from_geodetic(lon=122.04961 * u.deg, lat=37.53592 * u.deg, height=100 * u.m)
    who = Observer(location=location, name="who", timezone="Asia/Shanghai")
    # 滤光片安装位置和消光
    filters_config = {
        # "U": {"position": 0, "exp_rate": 5.21, "ext_coef": 0.778, "zero_point": 4.427},
        "B": {"position": 1, "exp_rate": 6.0, "ext_coef": 0.537, "zero_point": 2.644},
        "V": {"position": 2, "exp_rate": 3.0, "ext_coef": 0.331, "zero_point": 2.670},
        "R": {"position": 3, "exp_rate": 1.5, "ext_coef": 0.241, "zero_point": 2.392},
        "I": {"position": 5, "exp_rate": 1.0, "ext_coef": 0.185, "zero_point": 2.890},
    }

    # 速度配置
    tel_slew_rate = 3 * u.deg / u.second  # 望远镜指向，赤经赤纬相同
    dome_slew_rate = 2 * u.deg / u.second  # 圆顶转动速度，小于0表示无需考虑
    filterwheel_switch_rate = 0.25  # 滤光片转轮转动速度，一秒多少圈 count per second，小于0表示不考虑

    # 硬件参数
    tel_diam = 100  # 直径 cm
    tel_f = 800  # 焦距 cm
    tel_aperture = tel_diam / tel_f * 100  # 孔径，焦比倒数
    tel_efficiency = 0.6  # 光学效率

    ccd_pixel_size = 13.5  # um
    ccd_noise = 15.0
    ccd_dark = 0.002
    ccd_readout = 3 * u.second  # CCD读出时间

    # 环境值
    avg_sky_brightness = 19  # 平均天光，星等
    default_seeing = 1.75 * u.arcsec  # 当地视宁度
    get_exposure_time = exposure_time_dicts.get('whow')

elif station == StationName.MSGT:  # mstg 测站信息
    location = EarthLocation.from_geodetic(lon=74.893245 * u.deg, lat=38.335449 * u.deg, height=4490 * u.m)
    who = Observer(location=location, name="mstg", timezone="Asia/Shanghai")

    # 滤光片安装位置和消光
    filters_config = {
        "W": {"position": 5, "exp_rate": 1.0, "ext_coef": 0.24, "zero_point": 2.890},
    }

    # 速度配置
    tel_slew_rate = 5 * u.deg / u.second  # 望远镜指向，赤经赤纬相同
    dome_slew_rate = -5 * u.deg / u.second  # 圆顶转动速度，小于0表示无需考虑
    filterwheel_switch_rate = -0.25  # 滤光片转轮转动速度，一秒多少圈 count per second，小于0表示不考虑

    # 硬件参数
    tel_diam = 15  # 直径 cm
    tel_f = 21  # 焦距 cm
    tel_aperture = tel_diam / tel_f  # 孔径，焦比倒数
    tel_efficiency = 0.195  # 光学效率

    ccd_pixel_size = 9.0  # um
    ccd_noise = 15.0
    ccd_dark = 0.002
    ccd_readout = 1 * u.second  # CCD读出时间

    # 环境值
    avg_sky_brightness = 20  # 平均天光，星等
    default_seeing = 0.88 * u.arcsec  # 当地视宁度

    get_exposure_time = exposure_time_dicts.get('mstg')

else:
    raise ValueError('station must be either "wh" or "mstg"')
#

# 目标文件
target_file = 'targets/bl.csv'
# 观测时间设置，UTC
start_time = Time("2024-06-01 14:00:00")  # 起始时间UTC
time_duration = 2 * u.hour  # 持续时长
end_time = start_time + TimeDelta(val=time_duration)  # 结束时间
time_resolution = 5 * u.second  # 时间分辨率
time_gap = time_resolution * 10  # 时间调整间隔

# 随机种子
rnd_seed = 1

#  for ReinforcementScheduler
X_COORDINATE_IDX = 0
Y_COORDINATE_IDX = 1
# FILTER_IDX = 2
VIS_DURATION_TIME_IDX = 2
SCORE_2_IDX = 3
SCORE_1_IDX = 4
SCORE_0_IDX = 5
COEF_TIMES = 1e5  # 为了在输入文件中保持精度，计算时除以该值
RISE_TIME_WINDOW_IDX = 6
SET_TIME_WINDOW_IDX = 7
ARRIVAL_TIME_IDX = 8

feature_count = 9  # 特征数量
dynamic_feature_count = 9  # 动态特征数量
rnn_hidden = 128  # 隐藏层神经元数量
n_layers = 2
n_heads = 8  # 多头注意力
ff_dim = 256  # feed forward
chooser_type = 'greedy'  # beam = beam search chooser, greedy = greedy search chooser
max_beam_width = 9
use_checkpoint = False

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
map_location = {'cpu': 'cuda' if torch.cuda.is_available() else 'cpu'}
trained_model_path = 'trained_models/trained_model_40k.pkl'
# data_path = 'data_0601140000_5s.csv'
# data_cols = (2, 3, 5, 6, 7, 8, 9, 10, 10)  # 文件中分别对应前面的IDXs
# dist_matrix_path = 'dist_0601140000_5s.csv'
