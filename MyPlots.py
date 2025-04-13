import copy
import warnings
from typing import Sequence

from astroplan.plots import *
from astropy.table import Table
import matplotlib
from matplotlib import pyplot as plt
from matplotlib import dates

import numpy as np
import astropy.units as u
from astropy.time import Time
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
from matplotlib.ticker import MultipleLocator


def plot_scores(targets, schedule, ax=None, style_kwargs=None,
                style_sheet=None, min_score=0.0,
                max_score=1.0, use_local_tz=False):
    r"""
    Plots airmass as a function of time for a given target.

    If a `~matplotlib.axes.Axes` object already exists, an additional
    airmass plot will be "stacked" on it.  Otherwise, creates a new
    `~matplotlib.axes.Axes` object and plots airmass on top of that.

    When a scalar `~astropy.time.Time` object is passed in (e.g.,
    ``Time('2000-1-1')``), the resulting plot will use a 24-hour window
    centered on the time indicated, with airmass sampled at regular
    intervals throughout.
    However, the user can control the exact number and frequency of airmass
    calculations used by passing in a non-scalar `~astropy.time.Time`
    object. For instance, ``Time(['2000-1-1 23:00:00', '2000-1-1
    23:30:00'])`` will result in a plot with only two airmass measurements.

    For examples with plots, visit the documentation of
    :ref:`plots_time_dependent`.

    Parameters
    ----------
    targets : list of `~astroplan.FixedTarget` objects
        The celestial bodies of interest.
        If a single object is passed it will be converted to a list.

    observer : `~astroplan.Observer`
        The person, telescope, observatory, etc. doing the observing.

    time : `~astropy.time.Time`
        If scalar (e.g., ``Time('2000-1-1')``), will result in plotting target
        airmasses once an hour over a 24-hour window.
        If non-scalar (e.g., ``Time(['2000-1-1'])``, ``[Time('2000-1-1')]``,
        ``Time(['2000-1-1', '2000-1-2'])``),
        will result in plotting data at the exact times specified.

    ax : `~matplotlib.axes.Axes` or None, optional.
        The `~matplotlib.axes.Axes` object to be drawn on.
        If None, uses the current ``Axes``.

    style_kwargs : dict or None, optional.
        A dictionary of keywords passed into `~matplotlib.pyplot.plot_date`
        to set plotting styles.

    style_sheet : dict or `None` (optional)
        matplotlib style sheet to use. To see available style sheets in
        astroplan, print *astroplan.plots.available_style_sheets*. Defaults
        to the light theme.

    brightness_shading : bool
        Shade background of plot to scale roughly with sky brightness. Dark
        shading signifies times when the sun is below the horizon. Default
        is `False`.

    altitude_yaxis : bool
        Add alternative y-axis on the right side of the figure with target
        altitude. Default is `False`.

    min_airmass : float
        Lower limit of y-axis airmass range in the plot. Default is ``1.0``.

    max_airmass : float
        Upper limit of y-axis airmass range in the plot. Default is ``3.0``.

    min_region : float
        If set, defines an interval between ``min_airmass`` and ``min_region``
        that will be shaded. Default is `None`.

    max_region : float
        If set, defines an interval between ``max_airmass`` and ``max_region``
        that will be shaded. Default is `None`.

    use_local_tz : bool
        If the time is specified in a local timezone, the time will be plotted
        in that timezone.

    Returns
    -------
    ax : `~matplotlib.axes.Axes`
        An ``Axes`` object with added airmass vs. time plot.

    Notes
    -----
    y-axis is inverted and shows airmasses between 1.0 and 3.0 by default.
    If user wishes to change these, use ``ax.<set attribute>`` before drawing
    or saving plot:

    """
    # Import matplotlib, set style sheet
    if style_sheet is not None:
        matplotlib.rcdefaults()
        matplotlib.rcParams.update(style_sheet)

    # Set up plot axes and style if needed.
    if ax is None:
        ax = plt.gca()
    if style_kwargs is None:
        style_kwargs = {}
    style_kwargs = dict(style_kwargs)

    if 'lw' not in style_kwargs:
        style_kwargs.setdefault('linewidth', 1.5)
    if 'ls' not in style_kwargs and 'linestyle' not in style_kwargs:
        style_kwargs.setdefault('fmt', '-')
    time = (schedule.start_time + np.linspace(0, (schedule.end_time - schedule.start_time).value, 100) * u.day)
    if hasattr(time, 'utcoffset') and use_local_tz:
        tzoffset = time.utcoffset()
        tzname = time.tzname()
        tzinfo = time.tzinfo
    else:
        tzoffset = 0
        tzname = 'UTC'
        tzinfo = None
    # Populate time window if needed.
    # (plot against local time if that's requested)
    time_ut = Time(time)
    timetoplot = time_ut + tzoffset

    if not isinstance(targets, Sequence):
        targets = [targets]

    for target in targets:
        # Calculate airmass
        #airmass = observer.altaz(time_ut, target).secz
        scores = np.zeros(100, dtype='float')
        for block in schedule.observing_blocks:
            if block.target.name == target.name:
                avg_scores = block.constraint_avg_scores
                score_index = np.linspace(0, avg_scores.shape[0] - 1, 100).astype(int)
                scores = avg_scores[score_index]
                break

        # Plot data (against timezone-offset time)
        ax.plot_date(timetoplot.plot_date, scores, label=target.name, **style_kwargs)

    # Format the time axis
    xlo, xhi = (timetoplot[0]), (timetoplot[-1])
    ax.set_xlim([xlo.plot_date, xhi.plot_date])
    date_formatter = dates.DateFormatter('%H:%M')
    ax.xaxis.set_major_formatter(date_formatter)
    plt.setp(ax.get_xticklabels(), rotation=0, ha='center')

    # Shade background during night time

    # Invert y-axis and set limits.
    # y_lim = ax.get_ylim()
    # if y_lim[1] > y_lim[0]:
    #     ax.invert_yaxis()

    ax.set_ylim([min_score, max_score])

    # Draw lo/hi limit regions, if present
    ymax, ymin = ax.get_ylim()  # should be (hi_limit, lo_limit)

    # Set labels.
    ax.set_ylabel("Score")
    ax.set_xlabel(f"Time from {min(timetoplot).datetime.date()} [{tzname}]")

    # if altitude_yaxis and not _has_twin(ax):
    #     altitude_ticks = np.array([90, 60, 50, 40, 30, 20])
    #     airmass_ticks = 1./np.cos(np.radians(90 - altitude_ticks))
    #
    #     ax2 = ax.twinx()
    #     ax2.invert_yaxis()
    #     ax2.set_yticks(airmass_ticks)
    #     ax2.set_yticklabels(altitude_ticks)
    #     ax2.set_ylim(ax.get_ylim())
    #     ax2.set_ylabel('Altitude [degrees]')

    # Redraw figure for interactive sessions.
    ax.figure.canvas.draw()

    # Output.
    return ax


def get_target_color_value(target):
    # Sum the ASCII values of all characters in the word
    count = len(target.name)
    ascii_sum = sum(ord(char) for char in target.name)

    # Normalize the sum to the range [0, 1]
    max_ascii_sum = count * 255.0  # Since max ASCII value is 255
    normalized_value = ascii_sum / max_ascii_sum

    return normalized_value + count / 3.0


def plot_schedule_scores(schedule, show_night=False):
    """
    Plots when observations of targets are scheduled to occur superimposed
    upon plots of the scores of the targets.

    Parameters
    ----------
    schedule : `~astroplan.Schedule`
        a schedule object output by a scheduler
    show_night : bool
        Shades the night-time on the plot

    Returns
    -------
    ax :  `~matplotlib.axes.Axes`
        An ``Axes`` object with added airmass and schedule vs. time plot.
    """
    blocks = copy.copy(schedule.scheduled_blocks)
    sorted_blocks = sorted(schedule.observing_blocks, key=lambda x: x.target.name)
    targets = [block.target for block in sorted_blocks]
    ts = (schedule.start_time + np.linspace(0, (schedule.end_time - schedule.start_time).value, 100) * u.day)
    # plt.plot_date(ns.plot_date, masked_airmass, label=target_name, **style_kwargs)
    target_to_color = {}
    target_set = set(targets)  # 去重名
    names = np.sort([target.name for target in target_set])
    count = len(target_set)
    color_idx = np.linspace(0, 1, count)

    # lin_color = np.linspace(0, 1, count)
    # sorted_index = np.argsort(x)
    # color_idx = lin_color[sorted_index]

    # lighter, bluer colors indicate higher priority
    ax = plt.gca()
    time_ut = Time(ts)
    color_map = plt.cm.tab20
    for target_name, ci in zip(names, color_idx):
        # plot_airmass(target, schedule.observer, ts, style_kwargs=dict(color=color_map(ci)))
        #plot_altitude(target, schedule.observer, ts, style_kwargs=dict(color=color_map(ci)))
        # plot_scores(target, schedule, style_kwargs=dict(color=color_map(ci)), max_score=0.5, min_score=0.0)
        target_to_color[target_name] = color_map(ci)

    if show_night:
        # I'm pretty sure this overlaps a lot, creating darker bands
        for test_time in ts:
            midnight = schedule.observer.midnight(test_time)
            previous_sunset = schedule.observer.sun_set_time(
                midnight, which='previous')
            next_sunrise = schedule.observer.sun_rise_time(
                midnight, which='next')

            previous_twilight = schedule.observer.twilight_evening_astronomical(
                midnight, which='previous')
            next_twilight = schedule.observer.twilight_morning_astronomical(
                midnight, which='next')

            plt.axvspan(previous_sunset.plot_date, next_sunrise.plot_date,
                        facecolor='lightgrey', alpha=0.05)
            plt.axvspan(previous_twilight.plot_date, next_twilight.plot_date,
                        facecolor='lightgrey', alpha=0.05)

    score = 0.5
    for i, block in enumerate(blocks):
        if hasattr(block, 'target'):
            if hasattr(block, 'constraints_value'):
                score = block.constraints_value
                if score > 1.0:
                    score = 1.0
            # v = np.linspace(0, 1.0, 100)
            # ax.plot_date(time_ut.plot_date, v, label="test")
            ax.add_patch(Rectangle((block.start_time.plot_date, 0),
                                   block.end_time.plot_date - block.start_time.plot_date, score,
                                   label=block.target.name, fc=target_to_color[block.target.name], alpha=0.7))
            # # 方块
            # plt.axvspan(block.start_time.plot_date, block.end_time.plot_date,
            #             fc=target_to_color[block.target.name], alpha=1.0, ymax=score)
        else:  # transit block
            if i < len(blocks) - 1:
                score = blocks[i + 1].constraints_value
            ax.add_patch(Rectangle((block.start_time.plot_date, 0),
                                   block.end_time.plot_date - block.start_time.plot_date, score, alpha=0.8,
                                   label='Transitions', color='k'))
            #plt.axvspan(block.start_time.plot_date, block.end_time.plot_date, color='k', ymax=score, alpha=0.7)
    # plt.axhline(3, color='k', label='Transitions')

    xlo, xhi = (time_ut[0]), (time_ut[-1])
    ax.set_xlim([xlo.plot_date, xhi.plot_date])
    date_formatter = dates.DateFormatter('%H:%M')
    ax.xaxis.set_major_formatter(date_formatter)
    x_major_locator = MultipleLocator(0.01042)  # 15 minute # dates.MinuteLocator(byminute=5, interval=4)
    ax.xaxis.set_major_locator(x_major_locator)
    #plt.setp(ax.get_xticklabels(), rotation=0, ha='center')

    ax.set_ylim([0.00, 1.0])
    # plt.setp(ax.get_xticklabels(), rotation=0, ha='center')
    ax.set_ylabel('Score')
    ax.set_xlabel(f"Time from {time_ut[0].datetime.strftime('%Y-%m-%d %H:%M')} to {time_ut[-1].datetime.strftime('%H:%M')} (UTC)")

    handles, labels = ax.get_legend_handles_labels()
    new_handles, new_labels = [], []
    for p, t in zip(handles, labels):
        exists = t in new_labels
        if not exists:
            new_labels.append(t)
            new_handles.append(p)
    # new_labels.append(new_labels.pop(1))
    # new_handles.append(new_handles.pop(1))
    plt.legend(handles=new_handles, labels=new_labels,loc='upper right', ncol=4) #  bbox_to_anchor=(1.01, 1),
    # TODO: make this output a `axes` object


def schedule_table(schedule, show_transitions=True, show_unused=False):
    # TODO: allow different coordinate types
    target_names = []
    start_times = []
    end_times = []
    durations = []
    ra = []
    dec = []
    score = []
    config = []
    for slot in schedule.slots:
        if hasattr(slot.block, 'target'):
            start_times.append(slot.start.iso)
            end_times.append(slot.end.iso)
            durations.append(round(slot.duration.to(u.second).value, 3))
            target_names.append(slot.block.target.name)
            ra.append(round(slot.block.target.ra.value, 5))
            dec.append(round(slot.block.target.dec.value, 5))
            score.append(round(slot.block.constraints_value, 5))
            config.append(slot.block.configuration)
        elif show_transitions and slot.block:
            start_times.append(slot.start.iso)
            end_times.append(slot.end.iso)
            durations.append(round(slot.duration.to(u.second).value, 3))
            target_names.append('Transition')
            ra.append('')
            dec.append('')
            score.append(0)
            changes = {}
            for component in slot.block.components:
                changes[component] = round(slot.block.components[component].to(u.second).value, 3)
            # changes = list(slot.block.components.keys())
            # if 'slew_time' in changes:
            #     changes.remove('slew_time')
            config.append(changes)
        elif slot.block is None and show_unused:
            start_times.append(slot.start.iso)
            end_times.append(slot.end.iso)
            durations.append(round(slot.duration.to(u.second).value, 3))
            target_names.append('Unused')
            ra.append('')
            dec.append('')
            score.append(0)
            config.append('')
    return Table([target_names, start_times, end_times, durations, ra, dec, score, config],
                 names=('target', 'start time (UTC)', 'end time (UTC)',
                        'duration (seconds)', 'ra', 'dec', 'score', 'configuration'))


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import numpy as np
    filename = 'scores.csv'
    # 加载数据
    headers = np.loadtxt(filename, delimiter=',', skiprows=0, max_rows=1, dtype=str)
    headers = headers[1:] # 跳过Time
    data = np.loadtxt(filename, delimiter=',', skiprows=1)
    x_data = data[:, 0] # np.linspace(0, 1584, 1584)
    # y_data = [np.sin(x), np.cos(x), np.tan(x), np.sinh(x), np.cosh(x), np.tanh(x)]
    y_data = data[:, 1:]
    # y_data = []
    # 创建一个 2x3 的子图布局
    fig, axes = plt.subplots(2, 3, figsize=(16, 10), sharex=True)

    # 设置每个子图
    for i, ax in enumerate(axes.flat):

        # 截取大于0的部分
        y_index = np.where(y_data[:, i] > 0)[0]
        max_index = y_index.max()
        min_index = y_index.min()
        y = y_data[min_index:max_index+1, i]
        x = x_data[min_index:max_index+1]
        # 绘制散点图
        # ax.scatter(x, y, alpha=1.0, edgecolors='w', s=50, label='Score')
        ax.scatter(x, y, s=20, alpha=1, label='Score')
        # 拟合趋势线
        z = np.polyfit(x, y, 2)
        p = np.poly1d(z)
        y_pred = p(x)

        y_res = y - y_pred  # 残差
        RSS = np.sum(np.power(y_res, 2))  # 残差平方和
        ESS = len(y) * np.var(y)  # 总体平均方差
        r2 = 1 - RSS / ESS  # 拟合优度
        ax.plot(x, y_pred, "--", c='r', label=f'Fit ({r2:.6f})')
        ax.set_title(f'{headers[i]}')

        ax.set_xlim(0, 1600)
        # ax.set_xlabel('Time Step')
        # 确定纵轴范围
        y_max = np.max(y)
        y_min = np.min(y)
        offset = (y_max - y_min) * 0.1
        ax.set_ylim(max(y_min-offset, 0.0), min(y_max+offset, 1.0))
        # ax.autoscale(enable=True, axis='y', tight=False)
        # # ax.set_ylabel('Score')
        # 显示图例
        ax.legend()
        # coef = np.array2string(z, formatter={'float_kind': lambda x: f"{x:+.12f}\n"})[1:-1]
        # ax.text(800, y_min + offset * 3, f'coef:\n {coef}', ha='center')

    # 调整布局
    plt.tight_layout()
    fig.text(0.05, 0.5, 'Scores', va='center', rotation='vertical', fontsize='x-large')
    fig.text(0.5, 0.05, 'Time Steps', va='center', ha='center', fontsize='x-large')
    fig.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.1)
    plt.show()
