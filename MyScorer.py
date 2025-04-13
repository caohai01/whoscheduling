import hashlib
import os

import numpy as np
from astroplan import Schedule, time_grid_from_range
from astroplan.target import get_skycoord
from astropy import units as u


class AvgScorer(object):
    """
    Returns scores and score arrays from the evaluation of constraints on
    observing blocks
    """

    def __init__(self, blocks, observer, schedule, global_constraints=None):
        """
        Parameters
        ----------
        blocks : list of `~astroplan.scheduling.ObservingBlock` objects
            list of blocks that need to be scored
        observer : `~astroplan.Observer`
            the observer
        schedule : `~astroplan.scheduling.Schedule`
            The schedule inside which the blocks should fit
        global_constraints : list of `~astroplan.Constraint` objects
            any ``Constraint`` that applies to all the blocks
        """
        self.blocks = blocks
        self.observer = observer
        self.schedule = schedule
        self.global_constraints = global_constraints
        self.targets = get_skycoord([block.target for block in self.blocks])

    def compute_revisit_priority(self, block, current_time, min_visit_interval=0 * u.second):

        names = [b.name for b in self.schedule.scheduled_blocks]
        visited_count = names.count(block.name)
        exposure_count = block.number_exposure

        if visited_count < 1:
            rw = 1 / exposure_count
        else:
            names.reverse()
            index = len(names) - names.index(block.name) - 1
            last_visit_time = self.schedule.scheduled_blocks[index].start_time
            if min_visit_interval > 0 * u.second and last_visit_time + min_visit_interval <= current_time:
                rw = 0
            else:
                rw = visited_count * (1 - np.exp(visited_count - exposure_count))
        return rw

    def create_score_array(self, time_resolution=1 * u.minute):
        """
        this makes a score array over the entire schedule for all of the
        blocks and each `~astroplan.Constraint` in the .constraints of
        each block and in self.global_constraints.

        Parameters
        ----------
        time_resolution : `~astropy.units.Quantity`
            the time between each scored time

        Returns
        -------
        score_array : `~numpy.ndarray`
            array with dimensions (# of blocks, schedule length/ ``time_resolution``
        """

        constraint_score = self._load_scores(time_resolution)
        if constraint_score is not None:
            return constraint_score

        start = self.schedule.start_time
        end = self.schedule.end_time

        times = time_grid_from_range((start, end), time_resolution)

        constraint_score = np.ones((len(self.blocks), len(times)), dtype=np.float32)
        grid_count = constraint_score.shape[0]
        sum_score = np.zeros(constraint_score.shape)
        for i, block in enumerate(self.blocks):
            # TODO: 根据已观测列表调整重访优先级，保证优先级在[0, 1]
            for constraint in block.constraints:
                constraint_score[i] *= constraint(self.observer, block.target, times=times)

            if hasattr(block, 'priority'):
                constraint_score[i] *= block.priority  # 优先级

        for constraint in self.global_constraints:
            constraint_score *= constraint(self.observer, self.targets, times, grid_times_targets=True)

        self._save_scores(time_resolution, constraint_score)

        return constraint_score

    def _get_temp_file_name(self, time_resolution):

        def get_fixed_hash(string):
            # 创建一个 散列对象
            hash_object = hashlib.md5()  # hashlib.sha256()

            # 更新散列对象，注意需要将字符串编码成字节
            hash_object.update(string.encode('utf-8'))

            # 获取散列值的十六进制表示
            hash_value = hash_object.hexdigest()

            return hash_value

        start = self.schedule.start_time
        end = self.schedule.end_time

        b0 = (self.blocks[0].target.name + start.strftime('%Y%m%d%H%M%S') + end.strftime('%Y%m%d%H%M%S')
               + str(time_resolution) + self.blocks[-1].target.name + str(len(self.blocks)))

        temp_file = f"./caches/scores_cache_{get_fixed_hash(b0)}.npy"
        return temp_file

    def _load_scores(self, time_resolution):
        temp_file = self._get_temp_file_name(time_resolution)
        if os.path.exists(temp_file):
            return np.load(temp_file)
        else:
            return None

    def _save_scores(self, time_resolution, scores):
        temp_file = self._get_temp_file_name(time_resolution)
        np.save(temp_file, scores)

    def _get_average_score(self, score_array, time_resolution, poly_degree=2):
        avg_score = np.zeros(score_array.shape)
        grid_count = len(score_array[0])
        for i, b in enumerate(self.blocks):
            b_slots = np.ceil(b.duration / time_resolution)
            # b_slots_sq = b_slots ** 2
            start_j = -1
            end_j = grid_count
            for j in range(grid_count - b_slots):
                if start_j < 0 and score_array[i][j] > 0.0:  # first non zero
                    start_j = j
                if end_j == grid_count and start_j >= 0 and score_array[i][j] <= 0.0:  # first zero
                    end_j = j

                if all(score_array[i][j:j + b_slots]) and b_slots > 0:  # 全部有数值
                    avg_score[i][j] = np.sum(score_array[i][j:j + b_slots]) / b_slots
                else:  # 部分有数值，或者全0
                    avg_score[i][j] = 0
            b.constraint_avg_scores = np.copy(avg_score[i][:-b_slots])
            b.constraints_value = 0  # 初始值全部置为0

            if start_j < end_j - b_slots:
                start_j = max(start_j, 0)
                fit_grids = range(start_j, end_j)
                fit_scores = score_array[i][fit_grids]
                b.poly_parameters = np.polyfit(fit_grids, fit_scores, poly_degree)  # 曲线参数拟合

        # # 归一化到 [0,1] ==> 由于按照数值大小确定，归一化必要性不大?
        # max_score = np.max(prod_score)
        # min_score = np.min(prod_score)
        # if max_score > min_score:
        #     prod_score = (prod_score - min_score) / (max_score - min_score)
