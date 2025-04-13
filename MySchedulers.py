import copy
import os

import torch
from astroplan import Scheduler, AltitudeConstraint, Scorer, ObservingBlock, TransitionBlock, FixedTarget
from astropy import units as u
import numpy as np
from math import ceil, floor, fabs
from abc import abstractmethod

from astropy.coordinates import SkyCoord

from MyScorer import AvgScorer

from PN.Nets import RecPointerNetwork
from PN.Solution import BeamSearch, RunEpisode
from PN.Utility import scale_data


class MyScheduler(Scheduler):
    """
    自定义规划器基类
    """

    def __init__(self, *args, **kwargs):
        super(MyScheduler, self).__init__(*args, **kwargs)

    @abstractmethod
    def _make_schedule(self, blocks):
        raise NotImplementedError

    def _compute_scores(self, blocks):
        scorer = Scorer(blocks, self.observer, self.schedule, self.constraints)
        score_array = scorer.create_score_array(self.time_resolution)
        return score_array

    def _compute_average_scores(self, blocks):
        scorer = AvgScorer(blocks, self.observer, self.schedule, self.constraints)
        score_array = scorer.create_score_array(self.time_resolution)
        prod_score = np.zeros(score_array.shape)
        grid_count = len(score_array[0])

        for i, b in enumerate(blocks):
            b_slots = ceil(b.duration / self.time_resolution)
            # b_slots_sq = b_slots ** 2
            start_j = -1
            end_j = grid_count
            for j in range(grid_count - b_slots):

                if start_j < 0 and score_array[i][j] > 0.0: # first non zero
                    start_j = j
                if end_j == grid_count and start_j >= 0 and score_array[i][j] <= 0.0: # first zero
                    end_j = j

                if all(score_array[i][j:j + b_slots]) and b_slots > 0:  # 全部有数值
                    prod_score[i][j] = np.sum(score_array[i][j:j + b_slots]) / b_slots
                else:  # 部分有数值，或者全0
                    prod_score[i][j] = 0
            b.constraint_avg_scores = np.copy(prod_score[i][:-b_slots])
            b.constraints_value = 0  # 初始值全部置为0

            if start_j < end_j - b_slots:
                start_j = max(start_j, 0)
                fit_grids = range(start_j, end_j)
                fit_scores = prod_score[i][fit_grids]
                # b.poly_parameters = np.polyfit(fit_grids, fit_scores, 1)  # 一次曲线
                b.poly_parameters = np.polyfit(fit_grids, fit_scores, 2)  # 二次曲线参数
        # # 归一化到 [0,1] ==> 由于按照数值大小确定，归一化必要性不大?
        # max_score = np.max(prod_score)
        # min_score = np.min(prod_score)
        # if max_score > min_score:
        #     prod_score = (prod_score - min_score) / (max_score - min_score)
        return prod_score


    def _compute_slew_slots(self, blocks):
        s_count = 86400 * (self.schedule.end_time - self.schedule.start_time) / self.time_resolution * u.second / u.day
        slots_count = ceil(s_count.value)
        blocks_count = len(blocks)
        slew_array = np.zeros((blocks_count, blocks_count, slots_count), dtype="int")
        for i in range(blocks_count):
            bf = blocks[i]
            for j in range(i):
                bt = blocks[j]
                for k in range(slots_count):
                    current_time = self.schedule.start_time + k * self.time_resolution
                    trans = self.transitioner(bf, bt, current_time, self.observer)
                    if trans:
                        slew_array[j, i, k] = slew_array[i, j, k] = ceil(trans.duration / self.time_resolution)

        return slew_array

    def _compute_slew_slots_at_start(self, blocks):
        blocks_count = len(blocks)
        current_time = self.schedule.start_time
        slew_array = np.zeros((blocks_count, blocks_count), dtype="int")
        for i in range(blocks_count):
            bf = blocks[i]
            for j in range(i):
                bt = blocks[j]
                trans = self.transitioner(bf, bt, current_time, self.observer)
                if trans:
                    slew_array[j, i] = slew_array[i, j] = ceil(trans.duration / self.time_resolution)

        return slew_array

    def attempt_insert_block(self, new_block, new_start_time, start_time_idx):
        # set duration to be exact multiple of time resolution
        duration_indices = np.ceil(float(new_block.duration / self.time_resolution))
        new_block.duration = duration_indices * self.time_resolution

        # add 1 second to the start time to allow for scheduling at the start of a slot
        slot_index = [q for q, slot in enumerate(self.schedule.slots)
                      if slot.start < new_start_time + 1 * u.second < slot.end][0]
        slots_before = self.schedule.slots[:slot_index]
        slots_after = self.schedule.slots[slot_index + 1:]

        # now check if there's a transition block where we want to go
        # if so, we delete it. A new one will be added if needed
        delete_this_block_first = False
        if self.schedule.slots[slot_index].block:
            if isinstance(self.schedule.slots[slot_index].block, ObservingBlock):
                raise ValueError('block already occupied')
            else:
                delete_this_block_first = True

        # no slots yet, so we should be fine to just shove this in
        if not (slots_before or slots_after):
            new_block.end_idx = start_time_idx + duration_indices
            new_block.start_idx = start_time_idx
            if new_block.constraints is None:
                new_block.constraints = self.constraints
            elif self.constraints is not None:
                new_block.constraints = new_block.constraints + self.constraints
            try:
                self.schedule.insert_slot(new_start_time, new_block)
                return True
            except ValueError as error:
                # this shouldn't ever happen
                print('Failed to insert {} into schedule.\n{}'.format(
                    new_block.target.name, str(error)
                ))
                return False

        # Other slots exist, so now we have to see if it will fit
        # if slots before or after, we need `TransitionBlock`s
        tb_before = None
        tb_before_already_exists = False
        tb_after = None
        if slots_before:
            if isinstance(
                    self.schedule.slots[slot_index - 1].block, ObservingBlock):
                # make a transitionblock
                tb_before = self.transitioner(
                    self.schedule.slots[slot_index - 1].block, new_block,
                    self.schedule.slots[slot_index - 1].end, self.observer)
            elif isinstance(self.schedule.slots[slot_index - 1].block, TransitionBlock):
                tb_before = self.transitioner(
                    self.schedule.slots[slot_index - 2].block, new_block,
                    self.schedule.slots[slot_index - 2].end, self.observer)
                tb_before_already_exists = True

        if slots_after:
            slot_offset = 2 if delete_this_block_first else 1
            if isinstance(
                    self.schedule.slots[slot_index + slot_offset].block, ObservingBlock):
                # make a transition object after the new ObservingBlock
                tb_after = self.transitioner(
                    new_block, self.schedule.slots[slot_index + slot_offset].block,
                    new_start_time + new_block.duration, self.observer)

        # tweak durations to exact multiple of time resolution
        for block in (tb_before, tb_after):
            if block is not None:
                block.duration = self.time_resolution * ceil(float(block.duration / self.time_resolution))

        # if we want to shift the OBs to minimise gaps, here is
        # where we should do it.
        # Find the smallest shift (forward or backward) to close gap
        # Check against tolerances (constraints must still be met)
        # Shift if OK and update new_start_time and start_time_idx

        # Now let's see if the block and transition can fit in the schedule
        if slots_before:
            # we're OK if the index at the end of the updated transition
            # is less than or equal to `start_time_idx`
            ob_offset = 2 if tb_before_already_exists else 1
            previous_ob = self.schedule.slots[slot_index - ob_offset]
            if tb_before:
                transition_indices = int(tb_before.duration / self.time_resolution)
            else:
                transition_indices = 0

            if start_time_idx < previous_ob.block.end_idx + transition_indices:
                # cannot schedule
                return False

        if slots_after:
            # we're OK if the index at end of OB (plus transition)
            # is smaller than the start_index of the slot after
            slot_offset = 2 if delete_this_block_first else 1
            next_ob = self.schedule.slots[slot_index + slot_offset].block
            end_idx = start_time_idx + duration_indices
            if tb_after:
                end_idx += int(tb_after.duration / self.time_resolution)
                if end_idx >= next_ob.start_idx:
                    # cannot schedule
                    return False

        # OK, we should be OK to schedule now!
        try:
            # delete this block if it's a TransitionBlock
            if delete_this_block_first:
                slot_index = self.schedule.change_slot_block(slot_index, new_block=None)
            if tb_before and tb_before_already_exists:
                self.schedule.change_slot_block(slot_index - 1, new_block=tb_before)
            elif tb_before:
                self.schedule.insert_slot(tb_before.start_time, tb_before)
            elif tb_before_already_exists and not tb_before:
                # we already have a TB here, but we no longer need it!
                self.schedule.change_slot_block(slot_index - 1, new_block=None)

            new_block.end_idx = start_time_idx + duration_indices
            new_block.start_idx = start_time_idx
            if new_block.constraints is None:
                new_block.constraints = self.constraints
            elif self.constraints is not None:
                new_block.constraints = new_block.constraints + self.constraints
            self.schedule.insert_slot(new_start_time, new_block)

            if tb_after:
                self.schedule.insert_slot(tb_after.start_time, tb_after)

        except ValueError as error:
            # this shouldn't ever happen
            # print('Failed to insert {} (dur: {}) into schedule.\n{}\n{}'.format(
            #     new_block.target.name, new_block.duration, new_start_time.iso, str(error)
            # ))
            return False

        return True
    @property
    def total_score(self):
        t_score = 0.0
        schedule = self.schedule
        for b in schedule.observing_blocks:
            if not hasattr(b, "constraints_value"):  # 忽略没有值的
                continue
            t_score += b.constraints_value  # * b.duration.value
        return t_score


class MaxScheduler(MyScheduler):
    """
    place blocks to their highest score slots
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _make_schedule(self, blocks):
        prod_score = self._compute_average_scores(blocks)

        while True:  # loop util all prod_score 0
            # 选择最大值，找出对应的行列，存储到bi
            bi = np.unravel_index(np.argmax(prod_score), prod_score.shape)
            if prod_score[bi] <= 0:  # 最大值已经为0 -> 所有的已经处理
                break
            row = bi[0]
            col = bi[1]
            sel_block = blocks[row]
            slots = ceil(sel_block.duration / self.time_resolution)
            sel_block.constraints_value = prod_score[row, col] #* slots

            # col 存储了slots数量，可用来计算开始时间
            start_time = self.schedule.start_time + col * self.time_resolution
            sel_block.start_time = start_time
            sel_block.end_time = start_time + sel_block.duration
            try:
                prod_score[:, col] = 0  # 标记当前时间不可用，如果插入失败，则避免该位置
                # 试着插入当前sel_block
                success = self.attempt_insert_block(sel_block, start_time, col)
                # self.schedule.insert_slot(slot_time, sel_block)
                if success:
                    prod_score[row] = 0  # 清空所在行，保证下一次不被选中
                    # 标记剩余占用的时间为0，保证下一次不被选中
                    prod_score[:, col + 1:col + slots] = 0
                else:
                    raise Exception("attempt insert block failed")
            except Exception as ex:
                # print("warning:", ex, row, sel_block.target.name, start_time)
                pass

        return self.schedule


class GreedyScheduler(MyScheduler):
    """
    schedule blocks to current best value
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _make_schedule(self, blocks):
        block_count = len(blocks)
        trans_slots_count = np.zeros(block_count, dtype="int")
        observation_slots_count = np.zeros(block_count, dtype="int")
        block_scheduled = np.zeros(block_count, dtype="bool")
        prod_score = self._compute_average_scores(blocks)
        grid_count = prod_score.shape[1]

        # for i, b in enumerate(blocks): # 计算实际观测时长
        #     observation_slots_count[i] = ceil(b.duration / self.time_resolution)

        current_time = self.schedule.start_time
        current_index = 0
        while (current_index < grid_count) and (current_time < self.schedule.end_time):  # 双重保险
            # 首先计算当前时间下所有过度时间块及slots长度transition_blocks
            transition_blocks = []
            trans_slots_count.fill(0)
            scheduled_count = len(self.schedule.observing_blocks)
            if scheduled_count > 0:  # 前面已经有block，所以需要增加TransitionBlock
                for i, b in enumerate(blocks):
                    if block_scheduled[i]:  # 已经处理过的block直接忽略
                        transition_blocks.append(None)
                        continue

                    trans = self.transitioner(self.schedule.observing_blocks[-1], b, current_time, self.observer)
                    if trans:
                        transition_blocks.append(trans)
                        trans_slots_count[i] = ceil(trans.duration / self.time_resolution)  # 取slots上限
                    else:
                        transition_blocks.append(None)

            # 选择当前时间下分值最高的行
            cols = (current_index + trans_slots_count).clip(0, grid_count - 1)  # 不同的block的起始索引
            rows = np.arange(block_count)
            # 根据trans_slots_count重新计算分值， +1是为了避免除数为0，不影响最大值选取
            #observation_slots_count.fill(1)
            temp_score = prod_score[rows, cols] / (trans_slots_count + 1)
            row = np.argmax(temp_score)  # 选择temp_score最大的行
            # row = np.argmax(prod_score[rows, cols])  # 选择值最大的行，列数根据计算确定，不同的trans_slots_count，值不同

            new_block_col = cols[row]  # new_block_col block的起始时刻, row确定new block
            if prod_score[row, new_block_col] <= 0:  # 在该时刻没有合适的block，时间跳过，要是time_resolution的整数倍
                skip_index_count = ceil(self.gap_time / self.time_resolution)
                skip_time = skip_index_count * self.time_resolution

                current_time += skip_time
                current_index += skip_index_count
                continue

            # 选出了新的block
            new_block = blocks[row]
            new_slots_count = ceil(new_block.duration / self.time_resolution)
            new_block.duration = new_slots_count * self.time_resolution

            transition_block = transition_blocks[row] if transition_blocks else None

            if transition_block:  # 先插入过度块
                transition_block.duration = trans_slots_count[row] * self.time_resolution
                self.schedule.insert_slot(current_time, transition_block)
                current_time += trans_slots_count[row] * self.time_resolution
                current_index += trans_slots_count[row]

            if current_index > grid_count or current_time > self.schedule.end_time:  # 插入过度块之后已经没有空间
                # 最后一个transition_block 其实也没必要了，暂时保留吧
                break

            new_block.start_time = current_time
            new_block.end_time = current_time + new_block.duration
            new_block.constraints_value = prod_score[row, new_block_col]

            self.schedule.insert_slot(new_block.start_time, new_block)
            prod_score[row] = 0  # 清空所在行，保证下一次不被选中
            block_scheduled[row] = True  # 标记已经处理
            current_time += new_block.duration
            current_index += new_slots_count
        return self.schedule


class RandomScheduler(MyScheduler):
    """
    按照时间顺序随机选择目标进行观测，具体步骤为：随着时间的推进，随机选择一个未观测目标，如果能观测则进行观测，如果不能，则再随机
    如果多次随机均无法进行观测，则当前时间增加gap，继续
    """

    def __init__(self, *args, **kwargs):
        self.rnd_seed = 2
        super().__init__(*args, **kwargs)

    def _make_schedule(self, blocks):

        prod_score = self._compute_average_scores(blocks)
        grid_count = prod_score.shape[1]
        block_count = len(blocks)
        start_time = self.schedule.start_time
        current_time = start_time
        idx = list(range(block_count))
        np.random.seed(seed=self.rnd_seed)
        # np.random.shuffle(idx)
        f_count = 0  # 失败次数
        revers_filters = True

        while current_time < self.schedule.end_time and len(idx) > 0:
            # 随机选择一个block，不用choice，是需要索引移除已安排的元素
            t = np.random.randint(0, len(idx))

            name = blocks[idx[t]].target.name
            sel_indices = []

            revers_filters = not revers_filters
            for i in idx:
                if blocks[i].target.name == name:
                    sel_indices.insert(0, i) if revers_filters else sel_indices.append(i)

            for row in sel_indices:
                sel_block = blocks[row]
                trans = None
                scheduled_count = len(self.schedule.observing_blocks)
                if scheduled_count > 0:
                    # a test transition between the last scheduled block and this one
                    trans = self.transitioner(self.schedule.observing_blocks[-1], sel_block, current_time,
                                              self.observer)

                transition_time = 0 * u.second if not trans else trans.duration
                trans_duration = ceil(transition_time / self.time_resolution) * self.time_resolution
                start_time = current_time + trans_duration
                col = ceil((start_time - self.schedule.start_time) / self.time_resolution)

                if col >= grid_count or prod_score[
                    row, col] <= 0 or start_time > self.schedule.end_time:  # 当前这个观测不完或者已经不可观测
                    f_count += 1  # 失败，再随机另一个
                    if f_count > block_count:  # 如果失败次数过多，则跳过gap时间
                        skip_time = ceil(self.gap_time / self.time_resolution) * self.time_resolution
                        current_time += skip_time
                        f_count = 0
                    continue

                if trans:
                    trans.duration = trans_duration
                    self.schedule.insert_slot(current_time, trans)
                    current_time += trans_duration

                f_count = 0
                sel_slots =  ceil(sel_block.duration / self.time_resolution)
                sel_duration = sel_slots * self.time_resolution  # block_slots[row]
                sel_block.start_time = current_time
                sel_block.end_time = current_time + sel_duration
                sel_block.duration = sel_duration
                sel_block.constraints_value = prod_score[row, col] #* sel_slots
                self.schedule.insert_slot(current_time, sel_block)

                idx.remove(row)
                current_time += sel_duration

        return self.schedule


class VNSScheduler(MyScheduler):
    """
    Variable Neighborhood Search

    """

    def __init__(self, *args, **kwargs):
        self.rnd_seed = 1
        self.iter_count = 50
        self.attempt_count = 20
        self.neighbour = [self._stochastic_2_opt, self._stochastic_swap, self._roll]
        # self.neighbour = []
        # self.best_solution = None
        # self.best_score = 0

        super().__init__(*args, **kwargs)

    @staticmethod
    def _stochastic_2_opt(solution):
        """
        随机将一部分元素原地逆序
        2-opt : revers sub solution
        :param solution:
        :return: new solution
        """
        i, j = np.random.choice(range(0, len(solution) - 1), 2, replace=False)
        if i > j:  # keep i at front fo j
            i, j = j, i
        new_solution = copy.deepcopy(solution)
        new_solution[i:j + 1] = list(reversed(solution[i:j + 1]))
        return new_solution

    @staticmethod
    def _stochastic_swap(solution, length=1):
        """
        随机交换length个元素， 内部顺序保持不变
         swap sub-solution of length
        :param solution:
        :param length:
        :return: new solution
        """
        i, j = np.random.choice(range(0, len(solution) - length), 2, replace=False)
        new_solution = copy.deepcopy(solution)
        new_solution[i:i + length] = solution[j:j + length]
        new_solution[j:j + length] = solution[i:i + length]
        return new_solution

    @staticmethod
    def _roll(solution, shift=2):
        """
        向前滚动shift
         roll array by shift poistion
        :param solution:
        :param shift:
        :return: new solution
        """
        count = len(solution)
        new_solution = []
        for i in range(count):
            new_solution.append(solution[(i + shift) % count])

        return new_solution

    def _shake(self, solution, k=0):

        k %= len(self.neighbour)
        solution = self.neighbour[k](solution)
        return solution

    def _compute_total_score(self, solution, score_array, slew_array, blocks):
        total_score = 0
        cur_index = 0
        last_row = -1
        scheduled_count = 0
        for row in solution:
            block = blocks[row]
            duration_slots = ceil(block.duration / self.time_resolution)
            cur_score = score_array[row][cur_index] * duration_slots
            total_score += cur_score

            if cur_index > 0 and last_row >= 0:  # 除了第一个，其他都应该有过渡时间
                trans_slots = slew_array[row][last_row]
                duration_slots += trans_slots  # 暂时统一安排一个值，计算比较复杂

            cur_index += duration_slots
            scheduled_count += 1
            if cur_index >= score_array.shape[1]:
                total_score -= cur_score  # 最后一个安排不了
                scheduled_count -= 1
                break
            last_row = row

        return total_score * total_score / score_array.shape[0]  # 增加完成度

    def _variable_neighborhood_descent(self, solution, score_array, slew_array, blocks):
        k = 0
        best_solution = solution
        best_score = self._compute_total_score(best_solution, score_array, slew_array, blocks)
        neighbour_size = len(self.neighbour)
        while k < neighbour_size:
            s = self._shake(solution, k)
            score = self._compute_total_score(s, score_array, slew_array, blocks)
            if score > best_score:
                best_score = score
                best_solution = s
                k = 0
            else:
                k += 1

        return best_solution, best_score

    def _variable_neighborhood_search(self, init_solution, init_score, score_array, slew_array, blocks):
        best_solution = init_solution
        best_score = init_score
        fail_count = 0
        for _ in range(self.iter_count):
            for k in range(self.attempt_count):
                solution, solution_score = self._variable_neighborhood_descent(best_solution, score_array, slew_array,
                                                                               blocks)
                if solution_score <= best_score:
                    fail_count += 1
                    if fail_count > self.attempt_count:  # so many failed cases -> break
                        break
                else:
                    best_score = solution_score
                    best_solution = solution
                    fail_count = 0

        return best_solution, best_score

    def _make_schedule(self, blocks):
        prod_score = self._compute_average_scores(blocks)
        slew_slots = self._compute_slew_slots_at_start(blocks)  # too slow, replace with average slew time

        np.random.seed(seed=self.rnd_seed)
        block_count = len(blocks)
        init_solution = list(range(block_count))
        # TODO: 移除那些无法观测的目标
        ti = block_count - 1
        while ti >= 0:
            if not any(prod_score[ti]):
                init_solution.remove(ti)
            ti -= 1

        np.random.shuffle(init_solution)  # 初始方案完全随机
        init_score = self._compute_total_score(init_solution, prod_score, slew_slots, blocks)
        best_solution, best_score = self._variable_neighborhood_search(init_solution, init_score, prod_score,
                                                                       slew_slots, blocks)

        current_time = self.schedule.start_time

        for row in best_solution:
            sel_block = blocks[row]

            trans = None
            scheduled_count = len(self.schedule.observing_blocks)
            if scheduled_count > 0:
                # a test transition between the last scheduled block and this one
                trans = self.transitioner(self.schedule.observing_blocks[-1], sel_block, current_time,
                                          self.observer)

            transition_time = 0 * u.second if not trans else trans.duration
            trans_duration = ceil(transition_time / self.time_resolution) * self.time_resolution
            start_time = current_time + trans_duration
            if start_time >= self.schedule.end_time:  # 当前这个观测不完或者已经不可观测
                break

            col = ceil((start_time - self.schedule.start_time) / self.time_resolution)
            if col >= prod_score.shape[1]:
                break

            if prod_score[row, col] <= 0:  # 当前目标已经不可观测 -> 尝试下一个， 破坏了原来的顺序，最好的办法是利用VNS重新规划剩下的方案
                # current_time += ceil(sel_block.duration / self.time_resolution) * self.time_resolution

                continue

            if trans:
                trans.duration = trans_duration
                self.schedule.insert_slot(current_time, trans)
                current_time += trans_duration

            sel_slots = ceil(sel_block.duration / self.time_resolution)
            sel_duration = sel_slots * self.time_resolution  # block_slots[row]
            end_time = current_time + sel_duration
            if end_time > self.schedule.end_time:  # 安排不下
                break
            sel_block.start_time = current_time
            sel_block.end_time = end_time
            sel_block.duration = sel_duration
            sel_block.constraints_value = prod_score[row, col] # * sel_slots
            self.schedule.insert_slot(current_time, sel_block)

            current_time = end_time

        return self.schedule


class ReinforcementScheduler(MyScheduler):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _bs_inference(self, inst_data, dist_mat, start_time, args, run_episode):
        data_scaled = scale_data(inst_data, args)
        binst_data, bdata_scaled = inst_data.unsqueeze(0), data_scaled.unsqueeze(0)

        # beam_size = args.max_beam_size
        with torch.no_grad():
            seq, _ = run_episode(binst_data, bdata_scaled, start_time, dist_mat, 'greedy')

        # 需要修改BeamSearch，处理seq去掉不必要的元素，否则计算长度会有问题
        seq_list = [seq[:, k] for k in range(seq.shape[1])]

        rewards = run_episode.mu.reward_fn(inst_data, seq_list, dist_mat)
        max_reward, idx_max = torch.max(rewards, 0)
        score = max_reward.item()

        route = [val.item() for val in seq[idx_max] if val.item() != 0]

        return route, score

    def _gr_inference(self, inst_data, dist_mat, start_time, args, run_episode):
        data_scaled = scale_data(inst_data, args)
        binst_data, bdata_scaled = inst_data.unsqueeze(0), data_scaled.unsqueeze(0)

        with torch.no_grad():
            seq, _, _, _ = run_episode(binst_data, bdata_scaled, start_time, dist_mat, 'greedy')

        rewards = run_episode.mu.reward_fn(inst_data, seq, dist_mat)
        max_reward, idx_max = torch.max(rewards, 0)
        score = max_reward.item()

        route = [act.item() for act in seq]

        return route, score

    def _get_slew_times(self, blocks, operate_time):
        block_count = len(blocks)
        slew_time_matrix = np.zeros((block_count, block_count), dtype="int")

        for bi in range(block_count):
            bf = blocks[bi]
            for j in range(bi):
                bt = blocks[j]
                trans = self.transitioner(bf, bt, operate_time, self.observer)
                if trans:
                    slew_time_matrix[j, bi] = slew_time_matrix[bi, j] = ceil(trans.duration / self.time_resolution)

        return slew_time_matrix

    def _get_features(self, blocks, scores):
        block_count = len(blocks)
        assert block_count == scores.shape[0]
        final_time_step = scores.shape[1]

        f_data = []
        for bi in range(block_count):
            block = blocks[bi]
            coord = block.target.coord
            coeff = block.poly_parameters * 1e5

            start_index = next((i for i, x in enumerate(scores[bi]) if x), None)
            if start_index is None:  # all zero
                rise_step = set_step = 0
            else:
                end_index = next((i for i, x in enumerate(scores[bi][start_index:]) if not x),
                                 len(scores[bi]))
                if end_index > start_index:  #
                    rise_step = start_index
                    set_step = end_index
                else:
                    rise_step = set_step = 0

            f_data.append([coord.ra.value, coord.dec.value, ceil(block.duration / self.time_resolution),
                           coeff[0], coeff[1], coeff[2], rise_step, set_step, final_time_step])

        return np.array(f_data)
    def _get_solution(self, data, dist, cnf):
        # import Config as cnf
        np.random.seed(cnf.rnd_seed)
        torch.manual_seed(cnf.rnd_seed)
        if str(cnf.device) in ['cuda', 'cuda:0', 'cuda:1']:
            torch.cuda.manual_seed(cnf.rnd_seed)

        # 转换成 张量并加载到设备
        t_data = torch.FloatTensor(data).to(cnf.device)
        t_dist = torch.FloatTensor(dist).to(cnf.device)
        start_time_step = 0

        model = RecPointerNetwork(cnf.feature_count, cnf.dynamic_feature_count, cnf.rnn_hidden, cnf)
        model = model.to(cnf.device).eval()

        model.load_state_dict(torch.load(cnf.trained_model_path, weights_only=True))  # 加载存储的模型'trained_model.pkl'
        agent = RunEpisode(model, cnf)  # default greedy
        chooser = self._gr_inference
        if cnf.chooser_type == 'beam':  # beam search
            agent = BeamSearch(model, cnf)
            chooser = self._bs_inference

        agent = agent.eval()

        scheduled_index_list, _ = chooser(t_data, t_dist, start_time_step, cnf, agent)

        return scheduled_index_list

    def _make_schedule(self, blocks):
        all_prod_score = self._compute_average_scores(blocks)  # 模拟模型推理过程时间
        # 移除全为0的那些，因为模型训练推理时也会移除这些
        visible_index = np.any(all_prod_score, axis=1)
        invisible_blocks_idx = np.nonzero(~visible_index)
        prod_score = all_prod_score[visible_index]
        filtered_blocks = []
        for i, b in enumerate(blocks):
            if visible_index[i]:
                filtered_blocks.append(b)
        visual_block_count = len(filtered_blocks)
        import Config as cnf_rl # 由于配置过于复杂，直接使用配置文件
        # 准备数据data, dist
        #data = np.loadtxt(cnf_rl.data_path, delimiter=',', skiprows=1, usecols=cnf_rl.data_cols)
        data = self._get_features(filtered_blocks, prod_score)
        # 计算指向时间
        start_time = self.schedule.start_time
        end_time = self.schedule.end_time
        mid_time = start_time + (end_time - start_time) / 2
        dist = self._get_slew_times(filtered_blocks, mid_time)
        # dist = np.loadtxt(cnf_rl.dist_matrix_path, delimiter=',')
        # 加载模型并获取规划列表
        output = self._get_solution(data, dist, cnf_rl)

        grid_count = prod_score.shape[1]
        valid_block_count = prod_score.shape[0]

        current_time = start_time
        ii = 0

        # failed_count = 0
        # park = FixedTarget(coord=SkyCoord(ra=127.5625, dec=37.503, unit='deg'), name='park_position')
        # park_block = ObservingBlock(target=park, duration=0 * u.second, priority=1,
        #                             configuration={"filters": ["B"], "filters_position": [0]})
        # park_block.constraints_value = 0.0
        # self.schedule.insert_slot(current_time, park_block) # 模拟望远镜的起始park位置

        failed_blocks_index = []
        while current_time < self.schedule.end_time and ii < len(output):
            # 使用模型选择block，替换下面的随机过程
            row = output[ii]  # 这里为啥要-1? 为了训练，在模型中分别增加了一个起始节点，结果中的索引比实际目标要大1
            sel_block = filtered_blocks[row]

            trans = None
            scheduled_count = len(self.schedule.observing_blocks)
            if scheduled_count > 0:
                # a test transition between the last scheduled block and this one
                trans = self.transitioner(self.schedule.observing_blocks[-1], sel_block, current_time,
                                          self.observer)

            transition_time = 0 * u.second if not trans else trans.duration
            trans_duration = ceil(transition_time / self.time_resolution) * self.time_resolution
            start_time = current_time + trans_duration
            col = ceil((start_time - self.schedule.start_time) / self.time_resolution)

            if col >= grid_count or prod_score[row, col] <= 0 or start_time > self.schedule.end_time:  # 当前这个观测不完或者已经不可观测
                failed_blocks_index.append(row)
                # if len(failed_blocks_index) > block_count:  # 如果失败次数过多，则跳过时间
                #     skip_time = ceil(self.gap_time / self.time_resolution) * self.time_resolution
                #     current_time += skip_time
                # sel_duration = ceil(sel_block.duration / self.time_resolution) * self.time_resolution
                # current_time += sel_duration
                ii += 1  # 直接跳过
                continue

            if trans:
                trans.duration = trans_duration
                self.schedule.insert_slot(current_time, trans)
                current_time += trans_duration

            failed_count = 0
            sel_slots = ceil(sel_block.duration / self.time_resolution)
            sel_duration = sel_slots * self.time_resolution  # block_slots[row]
            sel_block.start_time = current_time
            sel_block.end_time = current_time + sel_duration
            sel_block.duration = sel_duration
            sel_block.constraints_value = prod_score[row, col] #* sel_slots
            prod_score[row] = 0 # 避免后面使用分数决定时出错
            self.schedule.insert_slot(current_time, sel_block)

            current_time += sel_duration
            ii += 1

        # 仍有空余时间，按照贪心方法依次安排，每次均选择最大的分值
        unscheduled_blocks_index = np.setdiff1d(np.array(list(range(0, valid_block_count))), np.array(output)) # -1 # 这里为啥要-1? 为了训练，在模型中分别增加了一个起始节点，结果中的索引比实际目标要大1
        if failed_blocks_index:
            unscheduled_blocks_index = np.append(failed_blocks_index, unscheduled_blocks_index).astype(dtype=np.int64)# 安排失败的也加进来

        while current_time < self.schedule.end_time and unscheduled_blocks_index.shape[0] > 0:
            col = ceil((current_time - self.schedule.start_time) / self.time_resolution)
            if col >= prod_score.shape[1]:
                break
            row = np.argmax(prod_score[unscheduled_blocks_index, col])

            if prod_score[unscheduled_blocks_index[row], col] <= 0:  # all zero
                break

            new_block = filtered_blocks[unscheduled_blocks_index[row]]
            new_slots_count = ceil(new_block.duration / self.time_resolution)
            new_block.duration = new_slots_count * self.time_resolution

            # a test transition between the last scheduled block and this one
            trans = self.transitioner(self.schedule.observing_blocks[-1], new_block, current_time, self.observer)
            trans_duration_time = ceil(trans.duration / self.time_resolution) * self.time_resolution # 取整
            trans.duration = trans_duration_time
            if current_time + trans_duration_time + new_block.duration > self.schedule.end_time: # 超出计划时间
                break

            self.schedule.insert_slot(current_time, trans)
            current_time += trans_duration_time

            # 给新加入的赋值
            new_block.start_time = current_time
            new_block.end_time = new_block.start_time + new_block.duration
            col = ceil((current_time - self.schedule.start_time) / self.time_resolution)
            new_block.constraints_value = prod_score[unscheduled_blocks_index[row], col] # * new_slots_count

            self.schedule.insert_slot(new_block.start_time, new_block)
            prod_score[row] = 0  # 清空所在行，保证下一次不被选中
            current_time += new_block.duration
            unscheduled_blocks_index = np.delete(unscheduled_blocks_index, row)  # remove scheduled index

        # # 如果还有剩余，在已经安排的列表中，寻找最小的一个安排，并在剩余中挑选小于该安排的指向时间+曝光时间的block，直接替换
        # scheduled_blocks = self.schedule.scheduled_blocks
        # unscheduled_blocks = filtered_blocks[unscheduled_blocks_index]
        # scheduled_scores = [b.constraints_value if hasattr(b, 'constraints_value') else 1 for b in scheduled_blocks]
        # min_index = scheduled_scores.index(min(scheduled_scores))
        # min_block = scheduled_blocks[min_index]
        # previous_block = None
        # min_index_duration = min_block.duration
        #
        # min_index_start_col = ceil((min_block.start_time - self.schedule.start_time) / self.time_resolution)
        # if min_index > 1: # 增加指向时间
        #     trans_block = scheduled_blocks[min_index - 1]
        #     previous_block = scheduled_blocks[min_index - 2]
        #     min_index_duration += trans_block.duration
        #     min_index_start_col -= (trans_block.duration / self.time_resolution)
        #     if min_index_start_col < 0:
        #         min_index_start_col = 0
        #
        # # 找出可能的候选目标
        # candidate_blocks_index = []
        # candidate_blocks_scores = []
        # for bi in unscheduled_blocks_index:
        #     b = filtered_blocks[bi]
        #     if b.duration <= min_index_duration and prod_score[bi, min_index_start_col] > min_block.constraints_value:
        #         candidate_blocks_index.append(bi)
        #         candidate_blocks_scores.append(prod_score[bi, min_index_start_col])
        #
        # if not candidate_blocks_index:
        #     continue
        # # 按照分值顺序逐个尝试，直至有一个成功
        # sorted_candidate_index = np.argsort(candidate_blocks_scores)
        # replaced = False
        # for bi in sorted_candidate_index:
        #     b = filtered_blocks[bi]
        #     if previous_block is not None: # 有前导块
        #         trans_block = scheduled_blocks[bi]
        #         if trans_block.duration + b.duration <= min_index_duration: # 总时间小 -> 替换
        #             # 先替换 trans_block
        #             # 再替换 ob_block
        #             replaced = True
        #             break # 替换结束就退出
        #     else: # 无前导块 -> 直接替换
        #         # 替换ob_blcck
        #         replaced = True
        #         break # 替换结束就退出
        # # 成功后，替换待观测目标，在待规划集合中排除该候选目标，并把当前目标放回未规划集合
        # if replaced:
        #     pass
        # else:
        #     pass
        # # 如果都不能成功，则在寻找下一个时排除当前的min_index


        return self.schedule
