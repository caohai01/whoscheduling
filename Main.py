import json
import os.path

from astroplan.constraints import AtNightConstraint, AirmassConstraint, MoonIlluminationConstraint, \
    MoonSeparationConstraint, AltitudeConstraint
from astroplan import ObservingBlock, Observer, FixedTarget, time_grid_from_range
from astroplan.constraints import TimeConstraint
from astropy import units as u
from astroplan.scheduling import Schedule

from astropy.coordinates import EarthLocation, SkyCoord, AltAz
from astropy.time import Time, TimeDelta
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from MySchedulers import MaxScheduler, GreedyScheduler, RandomScheduler, VNSScheduler, ReinforcementScheduler
from CustomConstraints import SkyBrightnessConstraint, CombinedConstraints, ExtinctionConstraint
from MyTransitioner import MyTransitioner
from MyPlots import plot_schedule_scores, schedule_table
#from Config import *
import Config as cnf
# 安装位置

target_file = cnf.target_file

df = pd.read_csv(target_file)

df.set_index('target_id', inplace=True)


# target = FixedTarget(coord=SkyCoord(ra=101.28715533*u.deg, dec=16.71611586*u.deg), name="372542")
# targets = [FixedTarget.from_name('Deneb'), FixedTarget.from_name('M13')]
def get_exposure_time_I0(snr, mag):
    """
    按照天顶角45度，大气消光0.2系数，天光背景恒定
    :param snr:
    :param mag:
    :param band:
    :return:
    """
    ccd_pixel_size = 13.5  # um
    ccd_gain = 1.8
    ccd_bin = 1
    ccd_noise = 15.0
    ccd_dark = 0.002

    tel_diam = 100  # 直径 cm
    tel_f = 800  # 焦距 cm
    tel_aperture = tel_diam / tel_f * 100  # 孔径，焦比倒数
    tel_efficiency = 0.6   # 光学效率
    tel_seeing = 1.75  # 当地视宁度

    zero_point = 3780  # 仪器零点1 https://irsa.ipac.caltech.edu/data/SPITZER/docs/dataanalysistools/tools/pet/magtojy/
    # U B V R

    ext_coefficient = 0.185
    std_flux = 679836
    airmass = 1.7  # 大气质量，高度在45度左右，计算出1.4
    avg_sky_brightness = 19.11  # mag/arcsec^2

    m = mag + ext_coefficient * airmass
    ms = avg_sky_brightness
    # 1 sbu = 10^-9 erg s^-1 cm^-2 Å^-1 sr^-1
    # (ccd_pixel_size * 10^-4)^2 * 10^9
    # ms = ms * tel_efficiency * tel_aperture * tel_aperture * (ccd_pixel_size * ccd_pixel_size * 10) * np.pi / 4.0
    scale = 206265.0 / tel_f * (ccd_pixel_size / 10000) / 2
    pixel_count = np.pi * (tel_seeing / scale) ** 2

    Ns = np.pi * tel_diam * tel_diam / 4 * std_flux * tel_efficiency * pow(10, -0.4 * m)
    Nb = np.pi * tel_diam * tel_diam / 4 * std_flux * tel_efficiency * pow(10, -0.4 * ms) * pixel_count
    Nd = ccd_dark * pixel_count

    a = Ns * Ns
    b = -snr * snr * (Ns + Nb + Nd)
    c = -snr * snr * pixel_count * ccd_noise * ccd_noise
    exp = (-b + np.sqrt(b * b - 4 * a * c)) / (2 * a)
    if exp < 0.01:  # 限制在0.01和600秒之间
        exp = 0.01
    if exp > 120 - 1:
        exp = 120 - 1
    return exp

# def get_exposure_time_I(snr, mag):
#     """
#     按照天顶角45度，大气消光0.2系数，天光背景恒定
#     :param snr:
#     :param mag:
#     :param band:
#     :return:
#     """
#     ccd_pixel_size = 13.5  # um
#     ccd_gain = 1.8
#     ccd_bin = 1
#     ccd_noise = 15.0
#     ccd_dark = 0.002
#
#     tel_diam = 100  # 直径 cm
#     tel_f = 800  # 焦距 cm
#     tel_aperture = tel_diam / tel_f * 100  # 孔径，焦比倒数
#     tel_efficiency = 0.6  # 光学效率
#     tel_seeing = 1.75  # 当地视宁度
#
#     zero_point = 3780  # 仪器零点1 https://irsa.ipac.caltech.edu/data/SPITZER/docs/dataanalysistools/tools/pet/magtojy/
#     # U B V R
#
#     ext_coefficient = 0.185
#     std_flux = 679836
#     airmass = 1.4  # 大气质量，高度在45度左右，计算出1.4
#     avg_sky_brightness = 18.5
#
#     m = mag + ext_coefficient * airmass
#     ms = avg_sky_brightness
#     # 1 sbu = 10^-9 erg s^-1 cm^-2 Å^-1 sr^-1
#     # (ccd_pixel_size * 10^-4)^2 * 10^9
#     # ms = ms * tel_efficiency * tel_aperture * tel_aperture * (ccd_pixel_size * ccd_pixel_size * 10) * np.pi / 4.0
#     scale = 206265.0 / tel_f * (ccd_pixel_size / 10000) / 2
#     pixel_count = np.pi * (tel_seeing / scale) * (tel_seeing / scale)
#
#     Ns = np.pi * tel_diam * tel_diam / 4 * std_flux * tel_efficiency * pow(10, -0.4 * m)
#     Nb = np.pi * tel_diam * tel_diam / 4 * std_flux * tel_efficiency * pow(10, -0.4 * ms) * pixel_count
#     Nd = ccd_dark * pixel_count
#
#     a = Ns * Ns
#     b = -snr * snr * (Ns + Nb + Nd)
#     c = -snr * snr * pixel_count * ccd_noise * ccd_noise
#     exp = (-b + np.sqrt(b * b - 4 * a * c)) / (2 * a)
#     if exp < 0.01:  # 限制在0.01和120秒之间
#         exp = 0.01
#     if exp > 120 - 1:
#         exp = 120 - 1
#     return exp



#
global_constraints = [
    AltitudeConstraint(min=10 * u.deg),
    AtNightConstraint.twilight_astronomical(),
    AirmassConstraint(max=3, boolean_constraint=False),
    # MoonSeparationConstraint(min=30 * u.deg),
    SkyBrightnessConstraint(),
    # MoonIlluminationConstraint.dark(max=0.5),
    # MagnitudeConstraint(boolean_constraint=False),
]
# alt = global_constraints[0]
# 天顶位置地平坐标
altaz_coord = SkyCoord(alt=90*u.deg, az=0*u.deg, frame='altaz', obstime=cnf.start_time, location=cnf.who.location)
# 转换为赤道坐标
park_position = FixedTarget(coord=altaz_coord.icrs, name="Park_Position")
park_block = ObservingBlock.from_exposures(target=park_position, priority=1, time_per_exposure=0 * u.second,
                                           number_exposures=0, readout_time=0,
                                           constraints=[],
                                           configuration={"filters": "None", "filters_position": [0]})

blocks = []
# df = df.sample(n=20, random_state=1)

filters_config = cnf.filters_config
mid_time = cnf.start_time + cnf.time_duration / 2
for index, dr in df.iterrows():
    ra = dr['RA']
    dec = dr['Dec']
    target_name = f'{index}'
    snr = dr['snr']
    mag = dr['mag1'] * 0.8 + dr['mag2'] * 0.2
    coord = SkyCoord(ra=ra, dec=dec, unit='deg')
    exp_time_t = cnf.get_exposure_time(snr, mag, coord, mid_time)
    if exp_time_t < 0:  # 无法观测
        continue
    n = dr["count"]
    t = FixedTarget(coord=coord, name=target_name)
    # visible = alt(observer=who, targets=t, time_range=(start_time, end_time), time_grid_resolution=15 * u.minute)
    # if not visible.all():  # 前后高度角不满足要求 -> 忽略
    #     continue

    exp_time = 0
    filters = ""
    filter_pos = []

    for f in filters_config:
        position = filters_config[f]["position"]
        # filter_pos.append(position)
        exp_time = filters_config[f]["exp_rate"] * exp_time_t
        ext_coef = filters_config[f]["ext_coef"]
        block_constraints = [ExtinctionConstraint(val=ext_coef)]
        # exp_time += 4.0 # 转轮转换时间
        if exp_time > 900.0:
            exp_time = 900.0
        elif exp_time < 0:  # 无法观测
            continue
        # filters = filters + "," + f
        # t.name = f"{index}-{f}"

        ob = ObservingBlock.from_exposures(target=t, priority=1, time_per_exposure=exp_time * u.second,
                                           number_exposures=n, readout_time=1 * u.second,
                                           constraints=block_constraints,
                                           configuration={"filters": f, "filters_position": [position]})
        blocks.append(ob)

    # ob = ObservingBlock.from_exposures(target=t, priority=1, time_per_exposure=exp_time* u.second,
    #                                    number_exposures=n, readout_time=1 * u.second,
    #                                    constraints=block_constraints,
    #                                    configuration={"filters": filters, "filters_position": filter_pos})
    # ob = ObservingBlock(t, exp_time, 0, {"filter": f}, constraints=block_constrain, name=t.name)
    # blocks.append(ob)

    # if len(blocks) >= 60: # limit 60
    #     break


transitioner = MyTransitioner(cnf.tel_slew_rate, cnf.dome_slew_rate, cnf.filterwheel_switch_rate)

schedulers = {
    "Greedy": GreedyScheduler,
    #"Max": MaxScheduler,
    #"VNS": VNSScheduler,
    "Random": RandomScheduler,
    "RL": ReinforcementScheduler,
}

for name in schedulers:
    print("scheduler:", name, "...")
    scheduler_class = schedulers[name]
    obj_scheduler = scheduler_class(constraints=global_constraints, transitioner=transitioner,
                                    observer=cnf.who, time_resolution=cnf.time_resolution, gap_time=cnf.time_gap)
    schedule = Schedule(cnf.start_time, cnf.end_time)
    obj_scheduler(blocks, schedule)

    # total_score = np.sum([b.constraints_value for b in schedule.observing_blocks if hasattr(b, "constraints_value")])
    total_score = obj_scheduler.total_score
    # for b in schedule.observing_blocks:
    #     if not hasattr(b, "constraints_value"): # 忽略没有值的
    #         continue
    #     total_score += b.constraints_value #  * b.duration.value
    # total_score *= len(schedule.observing_blocks) / len(blocks)
    plt.figure(figsize=(14, 6))
    plot_schedule_scores(schedule)
    # plot_schedule_airmass(schedule)
    # plt.legend(bbox_to_anchor=(1, 0), loc='upper right', borderaxespad=0)
    # plt.title(f"{name}(count:{len(schedule.observing_blocks)}, total score:{total_score: .3f})")
    plt.show()

    # tb = schedule.to_table()
    tb = schedule_table(schedule, show_unused=True)
    print(f"\n\n{name} - count:{len(schedule.observing_blocks)} score:{total_score: .3f}")
    tb.pprint_all()

    # 将结果存为csv文件
    fn = os.path.basename(target_file)
    fdir = 'plans' # os.path.dirname(target_file)
    plan_file_name = (f"{fdir}/plan_{name}_{cnf.start_time.strftime('%Y%m%d%H%M')}_{cnf.time_duration.to(u.hour).value:.1f}"
                      f"_{len(schedule.observing_blocks)}_{total_score:.3f}_{fn}")

    with open(plan_file_name, "w") as f:
        for slot in schedule.slots:
            b = slot.block
            item = None
            if hasattr(b, 'target'):
                altaz_frame = AltAz(obstime=b.start_time, location=cnf.who.location)
                target_altaz = b.target.coord.transform_to(altaz_frame)
                item = {"target": b.target.name,
                        "RA": round(b.target.ra.value, 6),
                        "Dec": round(b.target.dec.value, 6),
                        "RA_text": b.target.ra.to_string(sep='hms'),
                        "Dec_text": b.target.dec.to_string(),
                        "start_time": b.start_time.strftime("%Y-%m-%d %H:%M:%S"),
                        "end_time": b.end_time.strftime("%Y-%m-%d %H:%M:%S"),
                        "exposure_duration": round(b.time_per_exposure.to(u.second).value, 3),
                        "exposure_interval": 0, "exposure_count": b.number_exposures,
                        "filter_name": b.configuration["filters"],
                        "filter_index": b.configuration["filters_position"],
                        "altitude": round(target_altaz.alt.value, 3),
                        "azimuth": round(target_altaz.az.value, 3),
                        "constraints_value": round(b.constraints_value, 4),
                        }
            elif b:  # transition block
                item = {"target": "transition",
                        "start_time": b.start_time.strftime("%Y-%m-%d %H:%M:%S"),
                        "end_time": b.end_time.strftime("%Y-%m-%d %H:%M:%S"),
                        "duration": round(slot.duration.to(u.second).value, 3),
                        }
            if item is not None:
                j = json.dumps(item)
                f.write(j + ",\n")

    print(plan_file_name)
# exit()
