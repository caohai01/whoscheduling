import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from torch.distributions import Categorical


class DynamicFeatures():

    def __init__(self, args):
        super(DynamicFeatures, self).__init__()

        self.arrival_time_idx = args.ARRIVAL_TIME_IDX
        self.rise_time_window_idx = args.RISE_TIME_WINDOW_IDX
        self.set_time_window_idx = args.SET_TIME_WINDOW_IDX
        # self.reward_k_idx = args.REWARD_K_IDX
        # self.reward_b_idx = args.REWARD_B_IDX
        # self.reward_b_times = args.B_TIMES
        # self.reward_k_times = args.K_TIMES
        self.reward_idx = [args.SCORE_2_IDX, args.SCORE_1_IDX, args.SCORE_0_IDX]
        self.reward_times = args.COEF_TIMES
        self.device = args.device
        self.feature_count = args.dynamic_feature_count

    def make_dynamic_feat(self, data, current_time, current_poi_idx, dist_mat, batch_idx):
        num_dyn_feat = self.feature_count
        _, sequence_size, input_size = data.size()
        batch_size = batch_idx.shape[0]

        dyn_feat = torch.ones(batch_size, sequence_size, num_dyn_feat, device=self.device)

        # data[0, 0, :] stores the information of the start node
        tour_start_time = data[0, 0, self.rise_time_window_idx]
        max_tour_duration = data[0, 0, self.arrival_time_idx] - tour_start_time
        arrive_j_times = current_time + dist_mat[current_poi_idx]

        dyn_feat[:, :, 0] = (current_time - data[batch_idx, :, self.rise_time_window_idx]) / max_tour_duration
        dyn_feat[:, :, 1] = (data[batch_idx, :, self.set_time_window_idx] - current_time) / max_tour_duration
        dyn_feat[:, :, 2] = (data[batch_idx, :, self.arrival_time_idx] - current_time) / max_tour_duration
        dyn_feat[:, :, 3] = (current_time - tour_start_time) / max_tour_duration

        dyn_feat[:, :, 4] = (arrive_j_times - tour_start_time) / max_tour_duration
        dyn_feat[:, :, 5] = (arrive_j_times - data[batch_idx, :, self.rise_time_window_idx]) / max_tour_duration
        dyn_feat[:, :, 6] = (data[batch_idx, :, self.set_time_window_idx] - arrive_j_times) / max_tour_duration
        dyn_feat[:, :, 7] = (data[batch_idx, :, self.arrival_time_idx] - arrive_j_times) / max_tour_duration

        # 动态奖励
        # acts = current_poi_idx
        #if dyn_feat.shape[2] > 8:
        # dyn_feat[:, :, 8] = data[batch_idx, :, self.reward_k_idx] * arrive_j_times / self.reward_k_times + data[batch_idx, :, self.reward_b_idx] / self.reward_b_times
        dyn_feat[:, :, 8] = (data[batch_idx, :, self.reward_idx[0]] * arrive_j_times ** 2 +
                             data[batch_idx, :, self.reward_idx[1]] * arrive_j_times +
                             data[batch_idx, :, self.reward_idx[2]]) / self.reward_times

        return dyn_feat


class ModelUtils():
    def __init__(self, args):
        super(ModelUtils, self).__init__()

        self.device = args.device
        self.rise_time_window_idx = args.RISE_TIME_WINDOW_IDX
        self.set_time_window_idx = args.SET_TIME_WINDOW_IDX
        self.vis_duration_time_idx = args.VIS_DURATION_TIME_IDX
        self.arrival_time_idx = args.ARRIVAL_TIME_IDX
        # self.reward_k_idx = args.REWARD_K_IDX
        # self.k_times = args.K_TIMES
        # self.reward_b_idx = args.REWARD_B_IDX
        # self.b_times = args.B_TIMES
        self.coefs = [args.SCORE_2_IDX, args.SCORE_1_IDX, args.SCORE_0_IDX]
        self.coef_times = args.COEF_TIMES

    def feasibility_control(self, braw_inputs, mask, dist_mat, pres_act, present_time, batch_idx, first_step=False):

        done = False
        mask_c = mask.clone()  # 上次的mask
        # step_batch_size = batch_idx.shape[0]
        # float_zero = torch.FloatTensor([0.0]).to(self.device)
        end_time = braw_inputs[:, :, self.arrival_time_idx]

        arrivej = present_time + dist_mat[pres_act]  # 当前动作到所有其他动作的距离？
        # waitj = torch.max(float_zero, braw_inputs[:, :, self.opening_time_window_idx] - arrivej)
        # 不再考虑等待开门的问题
        # waitj = float_zero

        # c1 到达时间加上等待时间要大于开门时间才有效
        # c1 = arrivej >= braw_inputs[:, :, self.opening_time_window_idx]

        # c2 当前到达时间加上等待时间和停留时间要小于关门时间，才有效
        c2 = arrivej < braw_inputs[:, :, self.set_time_window_idx]
        #vc2 = arrivej <= braw_inputs[:, :, self.closing_time_window_idx]
        # c3 计算当前时间present_time下的收益值是否大于0，如果小于零，则mask 置为0， 否则置为1， 此处计算要排除最后一个节点
        # c3 = (braw_inputs[:, :, self.coefs[0]] * arrivej ** 2 + braw_inputs[:, :, self.coefs[1]] * arrivej +
        #       braw_inputs[:, :, self.coefs[2]]) > 0
        # c3[:, -1] = True  # c3条件需要排除最后一个节点，保证最后一个节点一直可以访问，否则会提前结束

        if not first_step:
            mask_c[batch_idx, pres_act] = 0

        mask_c[batch_idx] *= c2  # * c3

        # # 全部到了最后一个结点则认为结束，需要修改为：当前时间时间超出了最大时间则认为结束
        # if not mask_c[:, -1].any():
        #     done = True

        # 任何一个时间超出了最大时间则认为结束
        # print(batch_idx)
        # timeout = arrivej > end_time
        available_batch_idx = torch.nonzero(torch.any(mask_c, dim=1))
        if available_batch_idx.shape[0] < 1: # or timeout.all()
            done = True

        return done, mask_c

    def one_step_update(self, raw_inputs_b, dist_mat, pres_action, future_action, present_time, batch_idx, batch_size):

        present_time_b = torch.zeros(batch_size, 1, device=self.device)
        pres_actions_b = torch.zeros(batch_size, dtype=torch.int64, device=self.device)
        step_mask_b = torch.zeros(batch_size, 1, device=self.device, requires_grad=False, dtype=torch.bool)

        arrive_j = dist_mat[pres_action, future_action].unsqueeze(1) + present_time
        wait_j = torch.max(torch.FloatTensor([0.0]).to(self.device),
                           raw_inputs_b[batch_idx, future_action, self.rise_time_window_idx].unsqueeze(1) - arrive_j)
        present_time = arrive_j + wait_j + raw_inputs_b[batch_idx, future_action, self.vis_duration_time_idx].unsqueeze(
            1)

        present_time_b[batch_idx] = present_time

        pres_actions_b[batch_idx] = future_action
        step_mask_b[batch_idx] = 1

        return pres_actions_b, present_time_b, step_mask_b

    def adjacency_matrix(self, braw_inputs, mask, dist_mat, pres_act, present_time):
        mask_c = mask.clone()
        step_batch_size, points_count = mask_c.shape

        # # present_time not in [open, close]
        arrive_j = present_time + dist_mat[pres_act]
        # # 保证时间在升起之后
        c1 = arrive_j >= braw_inputs[:, :, self.rise_time_window_idx]
        # 保证时间在所有降落之前
        c2 = arrive_j + braw_inputs[:, :, self.vis_duration_time_idx] <= braw_inputs[:, :, self.arrival_time_idx]
        # 保证分数是大于0的
        c3 = (braw_inputs[:, :, self.coefs[0]] * (arrive_j ** 2) + braw_inputs[:, :, self.coefs[1]] * arrive_j +
              braw_inputs[:, :, self.coefs[2]]) > 0
        # c3[:, -1] = True  # c3条件需要排除最后一个节点，保证最后一个节点一直可以访问，虽然它的分数是0，否则会提前结束->改变后已经没必要
        mask_c *= c1 & c2 & c3

        adj_mask = mask_c.unsqueeze(1).repeat(1, points_count, 1)
        # masked index set to 0
        # 所有不可访问节点对饮的位置全为0
        z_idx = (mask_c == 0).nonzero()
        adj_mask[z_idx[:, 0], z_idx[:, 1], :] = 0
        adj_mask[z_idx[:, 0], :, z_idx[:, 1]] = 0

        # diagonal set to 0
        # 对角线置为0表示不能自己和自己连接
        dgn_idx = torch.arange(0, points_count, device=self.device)
        adj_mask[:, dgn_idx, dgn_idx] = 0
        return adj_mask

    def adjacency_matrix0(self, braw_inputs, mask, dist_mat, pres_act, present_time):
        # feasible neighborhood for each node
        maskk = mask.clone()
        step_batch_size, npoints = mask.shape

        #one step forward update
        arrivej = dist_mat[pres_act] + present_time
        farrivej = arrivej.view(step_batch_size, npoints)
        tw_start = braw_inputs[:, :, self.rise_time_window_idx]
        waitj = torch.max(torch.FloatTensor([0.0]).to(self.device), tw_start - farrivej)
        durat = braw_inputs[:, :, self.vis_duration_time_idx]

        fpresent_time = farrivej + waitj + durat
        fpres_act = torch.arange(0, npoints, device=self.device).expand(step_batch_size, -1)

        # feasible neighborhood for each node
        adj_mask = maskk.unsqueeze(1).repeat(1, npoints, 1)
        arrivej = dist_mat.expand(step_batch_size, -1, -1) + fpresent_time.unsqueeze(2)
        waitj = torch.max(torch.FloatTensor([0.0]).to(self.device), tw_start.unsqueeze(2) - arrivej)

        tw_end = braw_inputs[:, :, self.set_time_window_idx]
        ttime = braw_inputs[:, 0, self.arrival_time_idx]

        dlast = dist_mat[:, -1].unsqueeze(0).expand(step_batch_size, -1)

        c1 = arrivej + waitj + durat.unsqueeze(1) <= tw_end.unsqueeze(1)
        c2 = arrivej + waitj + durat.unsqueeze(1) + dlast.unsqueeze(1) <= ttime.unsqueeze(1).unsqueeze(1).expand(-1,
                                                                                                                  npoints,
                                                                                                                  npoints)
        adj_mask = adj_mask * c1 * c2

        # self-loop
        idx = torch.arange(0, npoints, device=self.device).expand(step_batch_size, -1)
        adj_mask[:, idx, idx] = 1

        return adj_mask

    def reward_fn(self, data, sample_solution, dist_mat):
        """
        Returns:
            Tensor of shape [batch_size] containing rewards
        """
        batch_size = torch.tensor(sample_solution[0].shape[0], device=self.device)
        total_count = torch.tensor([data.shape[0]] * batch_size, device=self.device)

        total_reward = torch.zeros(batch_size, device=self.device)
        arrival_j = torch.zeros(batch_size, device=self.device)
        stop_sign = torch.zeros(batch_size, device=self.device, dtype=torch.bool)
        completion_ratio = torch.ones(batch_size, device=self.device)

        valid_count = torch.zeros(batch_size, device=self.device)
        last_act_id = sample_solution[0]
        sample_solution.pop(0)  # 第一个总是起始位置
        for i, act_id in enumerate(sample_solution):
            # 逐个依次计算奖励，和时间呈线性关系
            stop_sign = stop_sign | (act_id == 0)  # 标记那些已经停掉的方案
            if stop_sign.all():  # all zero, 结束, 其实没必要，因为sample_solution不会到那一步
                break

            arrival_j += dist_mat[last_act_id, act_id]  # add distance
            # total_reward += (data[act_id, self.reward_k_idx] / self.k_times * arrival_j +
            #                 data[act_id, self.reward_b_idx] / self.b_times).squeeze(0)
            duration = data[act_id, self.vis_duration_time_idx]
            step_reward = ((data[act_id, self.coefs[0]] * (arrival_j ** 2) +
                            data[act_id, self.coefs[1]] * arrival_j +
                            data[act_id, self.coefs[2]]) / self.coef_times).squeeze(0)

            # step_reward = torch.nan_to_num(step_reward, nan=0.0)  # replace nan with 0
            step_reward = step_reward.clamp(min=0.0, max=1.0)  # keep reward in [0, 1]
            # only valid steps be considered
            valid = (~stop_sign).int()
            total_reward += step_reward * valid
            valid_count += valid
            # move on
            arrival_j += duration  # add duration
            last_act_id = act_id

        if data.shape[0] >= 1:
            completion_ratio = valid_count / total_count
        return total_reward * completion_ratio

    def reward_fn1(self, data, sample_solution, dist_mat):
        """
        Returns:
            Tensor of shape [batch_size] containing rewards
        """
        batch_size = sample_solution[0].shape[0]

        total_reward = torch.zeros(batch_size, device=self.device)
        arrival_j = torch.zeros(batch_size, device=self.device)
        stop_sign = torch.zeros(batch_size, device=self.device, dtype=torch.bool)
        completion_ratio = torch.ones(batch_size, device=self.device)

        valid_count = torch.zeros(batch_size, device=self.device)
        last_act_id = sample_solution[0]
        sample_solution.pop(0) # 第一个总是起始位置
        for i, act_id in enumerate(sample_solution):
            # 逐个依次计算奖励，和时间呈线性关系
            stop_sign = stop_sign | (act_id == 0)  # 标记那些已经停掉的方案
            if stop_sign.all():  # all zero, 结束
                break

            arrival_j += dist_mat[last_act_id, act_id]  # add distance
            # total_reward += (data[act_id, self.reward_k_idx] / self.k_times * arrival_j +
            #                 data[act_id, self.reward_b_idx] / self.b_times).squeeze(0)
            step_reward = ((data[act_id, self.coefs[0]] * arrival_j ** 2 +
                            data[act_id, self.coefs[1]] * arrival_j +
                            data[act_id, self.coefs[2]]) / self.coef_times).squeeze(0)

            step_reward = step_reward.clamp(min=0.0, max=1.0)  # keep reward in [0, 1]
            # only valid steps be considered
            valid = (~stop_sign).int()
            total_reward += step_reward * valid
            valid_count += valid
            # move on
            arrival_j += data[act_id, self.vis_duration_time_idx]  # add duration
            last_act_id = act_id

        if data.shape[0] >= 1:
            completion_ratio = valid_count / data.shape[0]
        return total_reward * completion_ratio

class RunEpisode(nn.Module):

    def __init__(self, neuralnet, args):
        super(RunEpisode, self).__init__()

        self.device = args.device
        if type(neuralnet) == torch.nn.DataParallel or type(neuralnet) == torch.nn.parallel.DistributedDataParallel:
            self.neuralnet = neuralnet.module  # add module for data parallel
        else:
            self.neuralnet = neuralnet
        self.dyn_feat = DynamicFeatures(args)
        # self.lookahead = Lookahead(args)
        self.mu = ModelUtils(args)

    def forward(self, binputs, bdata_scaled, start_time, dist_mat, infer_type):

        self.batch_size, sequence_size, input_size = binputs.size()

        h_0, c_0 = self.neuralnet.decoder.hidden_0

        # for lstm
        dec_hidden = (h_0.expand(self.batch_size, -1), c_0.expand(self.batch_size, -1))

        # for gru,
        # dec_hidden = h_0.expand(self.batch_size, -1)
        # 首先把分值为0的点mask值设为0，屏蔽那些不用的
        # 添加mask，屏蔽那些不能访问的数据

        # gap = torch.ones((self.batch_size, sequence_size), device=self.device)
        # 有足够的可观测窗口
        valid_idx = (binputs[:, :, self.mu.set_time_window_idx] -
                     binputs[:, :, self.mu.rise_time_window_idx] >= binputs[:, :, self.mu.vis_duration_time_idx])
        # valid_idx2 = binputs[:, :,self.mu.vis_duration_time_idx] >= gap
        # mask = torch.ones(self.batch_size, sequence_size, device=self.device, requires_grad=False, dtype=torch.uint8)
        mask = valid_idx.int()

        bpresent_time = start_time * torch.ones(self.batch_size, 1, device=self.device)
        bend_time = binputs[:, 0, self.mu.arrival_time_idx].unsqueeze(1)
        lst_log_probs, lst_actions, lst_step_mask, lst_entropy = [], [], [], []
        bpres_actions = torch.zeros(self.batch_size, dtype=torch.int32, device=self.device)

        batch_idx = torch.arange(0, self.batch_size, device=self.device)

        mask[:, 0] = 0  # 第一个为起始节点
        done, mask = self.mu.feasibility_control(binputs[batch_idx], mask, dist_mat, bpres_actions,
                                                 bpresent_time, batch_idx, first_step=True)

        adj_mask = self.mu.adjacency_matrix(binputs[batch_idx], mask, dist_mat, bpres_actions, bpresent_time)

        # encoder first forward pass
        bdyn_inputs = self.dyn_feat.make_dynamic_feat(binputs, bpresent_time, bpres_actions, dist_mat, batch_idx)
        emb1 = self.neuralnet.sta_emb(bdata_scaled)
        emb2 = self.neuralnet.dyn_emb(bdyn_inputs)
        enc_inputs = torch.cat((emb1, emb2), dim=2)

        _, _, enc_outputs = self.neuralnet(enc_inputs, enc_inputs, adj_mask, enc_inputs, dec_hidden, mask,
                                           first_step=True)

        decoder_input = enc_outputs[batch_idx, bpres_actions]

        done, mask = self.mu.feasibility_control(binputs[batch_idx], mask, dist_mat, bpres_actions, bpresent_time,
                                                 batch_idx)

        adj_mask = self.mu.adjacency_matrix(binputs[batch_idx], mask, dist_mat, bpres_actions, bpresent_time)

        # encoder/decoder forward pass
        bdyn_inputs = self.dyn_feat.make_dynamic_feat(binputs, bpresent_time,
                                                      bpres_actions, dist_mat, batch_idx)
        emb2 = self.neuralnet.dyn_emb(bdyn_inputs)
        enc_inputs = torch.cat((emb1, emb2), dim=2)

        if not done:  # 避免第一次测试就没有可行解造成训练异常
            policy, dec_hidden, enc_outputs = self.neuralnet(enc_inputs, enc_outputs, adj_mask, decoder_input, dec_hidden,
                                                             mask)

            lst_actions.append(bpres_actions)

        # Starting the trip
        while not done:

            future_actions, log_probs, entropy = self.select_actions(policy, infer_type)

            bpres_actions, bpresent_time, bstep_mask = self.mu.one_step_update(binputs, dist_mat,
                                                                               bpres_actions[batch_idx],
                                                                               future_actions, bpresent_time[batch_idx],
                                                                               batch_idx, self.batch_size)

            blog_probs = torch.zeros(self.batch_size, 1, dtype=torch.float32, device=self.device)
            blog_probs[batch_idx] = log_probs.unsqueeze(1)

            bentropy = torch.zeros(self.batch_size, 1, dtype=torch.float32, device=self.device)
            bentropy[batch_idx] = entropy.unsqueeze(1)

            lst_log_probs.append(blog_probs)
            lst_actions.append(bpres_actions)
            lst_step_mask.append(bstep_mask)
            lst_entropy.append(bentropy)

            done, mask = self.mu.feasibility_control(binputs[batch_idx], mask, dist_mat,
                                                     bpres_actions[batch_idx], bpresent_time[batch_idx],
                                                     batch_idx)

            if done:
                break

            # # mask[:, -1]为0表示结束
            # sub_batch_idx = torch.nonzero(mask[batch_idx][:, -1]).squeeze(1) # 原来的索引映射
            # batch_idx = torch.nonzero(mask[:, -1]).squeeze(1) # 仅保留未结束的batch

            # bpresent_time <= bend_time 为0表示结束，需压缩为1维数组

            sub_batch_idx = torch.nonzero(torch.any(mask[batch_idx], dim=1)).squeeze(1)
            # batch_idx = torch.nonzero((bpresent_time < bend_time).squeeze(1)).squeeze(1)
            # batch_index remove mask all 0
            batch_idx = torch.nonzero(torch.any(mask, dim=1)).squeeze(1)

            adj_mask = self.mu.adjacency_matrix(binputs[batch_idx], mask[batch_idx], dist_mat,
                                                bpres_actions[batch_idx], bpresent_time[batch_idx])

            #update decoder input and hidden
            decoder_input = enc_outputs[sub_batch_idx, bpres_actions[sub_batch_idx]]  # ???

            # for lstm
            dec_hidden = (dec_hidden[0][sub_batch_idx], dec_hidden[1][sub_batch_idx])
            # for gru
            # dec_hidden = dec_hidden[sub_batch_idx]
            # encoder/decoder forward pass
            bdyn_inputs = self.dyn_feat.make_dynamic_feat(binputs, bpresent_time[batch_idx], bpres_actions[batch_idx],
                                                          dist_mat, batch_idx)
            emb2 = self.neuralnet.dyn_emb(bdyn_inputs)
            enc_inputs = torch.cat((emb1[batch_idx], emb2), dim=2)

            policy, dec_hidden, enc_outputs = self.neuralnet(enc_inputs, enc_outputs[sub_batch_idx], adj_mask,
                                                             decoder_input, dec_hidden, mask[batch_idx])
        if not lst_log_probs:  # No solution was found.
            t_zero = torch.zeros((self.batch_size, 1), dtype=torch.float32, device=self.device)
            return lst_actions, t_zero, t_zero, t_zero < 0
        else:
            return lst_actions, torch.cat(lst_log_probs, dim=1), torch.cat(lst_entropy, dim=1), torch.cat(lst_step_mask,
                                                                                                          dim=1)

    def select_actions(self, policy, infer_type):

        if infer_type == 'stochastic':
            # if torch.isnan(policy).any():  # any nan -> random policy, 有可能出现全0的情况，采样会失败
            #     act_ind = torch.randint(0, policy.shape[1], (self.batch_size,), device=self.device)
            #     log_select = torch.zeros(self.batch_size, requires_grad=False, device=self.device)
            #     poli_entro = torch.zeros(self.batch_size, requires_grad=False, device=self.device)
            # else:
            m = Categorical(policy)
            act_ind = m.sample()
            log_select = m.log_prob(act_ind)
            poli_entro = m.entropy()

        elif infer_type == 'greedy':
            prob, act_ind = torch.max(policy, 1)
            log_select = prob.log()
            poli_entro = torch.zeros(self.batch_size, requires_grad=False, device=self.device)

        return act_ind, log_select, poli_entro


class BeamSearch(nn.Module):
    def __init__(self, neuralnet, args):
        super(BeamSearch, self).__init__()

        self.device = args.device
        self.neuralnet = neuralnet
        self.dyn_feat = DynamicFeatures(args)
        # self.lookahead = Lookahead(args)
        self.mu = ModelUtils(args)
        self.beam_size = args.max_beam_width

    def forward(self, inputs, data_scaled, start_time, dist_mat, infer_type):

        batch_size, sequence_size, input_size = inputs.size()

        if self.beam_size > sequence_size:  # 如果比输入还要大，那就按照输入大小
            self.beam_size = sequence_size

        # first step  - node 0
        bpresent_time = start_time * torch.ones(1, 1, device=self.device)

        # gap = torch.ones((batch_size, sequence_size), device=self.device)
        # valid_idx = (inputs[:, :, self.mu.closing_time_window_idx] - inputs[:, :, self.mu.opening_time_window_idx] >=
        #              inputs[:, :, self.mu.vis_duration_time_idx])
        # # valid_idx2 = inputs[:, :, self.mu.vis_duration_time_idx] >= gap
        # mask = valid_idx.int()
        # mask[:, 0] = 0
        mask = torch.ones(batch_size, sequence_size, device=self.device, requires_grad=False, dtype=torch.uint8)

        bpres_actions = torch.zeros(1, dtype=torch.int64, device=self.device)
        beam_idx = torch.arange(0, 1, device=self.device)
        batch_idx = torch.arange(0, mask.shape[0], device=self.device)
        done, mask = self.mu.feasibility_control(inputs.expand(beam_idx.shape[0], -1, -1),
                                                 mask, dist_mat, bpres_actions, bpresent_time, batch_idx,
                                                 first_step=True)
        adj_mask = self.mu.adjacency_matrix(inputs.expand(beam_idx.shape[0], -1, -1),
                                            mask, dist_mat, bpres_actions, bpresent_time)

        h_0, c_0 = self.neuralnet.decoder.hidden_0
        dec_hidden = (h_0.expand(1, -1), c_0.expand(1, -1))

        # encoder first forward pass
        bdata_scaled = data_scaled.expand(1, -1, -1)
        sum_log_probs = torch.zeros(1, device=self.device).float()

        bdyn_inputs = self.dyn_feat.make_dynamic_feat(inputs.expand(1, -1, -1), bpresent_time, bpres_actions, dist_mat,
                                                      beam_idx)
        emb1 = self.neuralnet.sta_emb(bdata_scaled)
        emb2 = self.neuralnet.dyn_emb(bdyn_inputs)
        enc_inputs = torch.cat((emb1, emb2), dim=2)

        _, _, enc_outputs = self.neuralnet(enc_inputs, enc_inputs, adj_mask, enc_inputs, dec_hidden, mask,
                                           first_step=True)

        decoder_input = enc_outputs[beam_idx, bpres_actions]

        done, mask = self.mu.feasibility_control(inputs.expand(beam_idx.shape[0], -1, -1),
                                                 mask, dist_mat, bpres_actions, bpresent_time,
                                                 torch.arange(0, mask.shape[0], device=self.device))
        adj_mask = self.mu.adjacency_matrix(inputs.expand(beam_idx.shape[0], -1, -1),
                                            mask, dist_mat, bpres_actions, bpresent_time)

        # encoder/decoder forward pass
        bdyn_inputs = self.dyn_feat.make_dynamic_feat(inputs.expand(beam_idx.shape[0], -1, -1), bpresent_time,
                                                      bpres_actions, dist_mat, beam_idx)
        emb2 = self.neuralnet.dyn_emb(bdyn_inputs)
        enc_inputs = torch.cat((emb1, emb2), dim=2)

        policy, dec_hidden, enc_outputs = self.neuralnet(enc_inputs, enc_outputs, adj_mask, decoder_input, dec_hidden,
                                                         mask)
        # 初始方案?第二个action?
        future_actions, log_probs, beam_idx = self.choose_actions(policy, sum_log_probs, mask, infer_type)
        # info update
        h_step = torch.index_select(dec_hidden[0], dim=0, index=beam_idx)
        c_step = torch.index_select(dec_hidden[1], dim=0, index=beam_idx)
        dec_hidden = (h_step, c_step)

        mask = torch.index_select(mask, dim=0, index=beam_idx)
        bpresent_time = torch.index_select(bpresent_time, dim=0, index=beam_idx)
        bpres_actions = torch.index_select(bpres_actions, dim=0, index=beam_idx)
        enc_outputs = torch.index_select(enc_outputs, dim=0, index=beam_idx)
        sum_log_probs = torch.index_select(sum_log_probs, dim=0, index=beam_idx)

        emb1 = torch.index_select(emb1, dim=0, index=beam_idx)

        # initialize buffers
        bllog_probs = torch.zeros(bpres_actions.shape[0], sequence_size, device=self.device).float()
        blactions = torch.zeros(bpres_actions.shape[0], sequence_size, device=self.device).long()

        sum_log_probs += log_probs.squeeze(0).detach()

        step = 0
        blactions[:, step] = bpres_actions

        final_log_probs, final_actions, last_step_mask = [], [], []

        # Starting the trip
        while not done and step < sequence_size - 1:

            future_actions = future_actions.squeeze(0)

            batch_size = bpres_actions.shape[0]
            batch_idx = torch.arange(0, batch_size, device=self.device)
            bpres_actions, bpresent_time, bstep_mask = self.mu.one_step_update(inputs.expand(batch_size, -1, -1),
                                                                               dist_mat,
                                                                               bpres_actions, future_actions,
                                                                               bpresent_time, batch_idx, batch_size)

            bllog_probs[:, step] = log_probs
            blactions[:, step + 1] = bpres_actions
            step += 1

            done, mask = self.mu.feasibility_control(inputs.expand(beam_idx.shape[0], -1, -1),
                                                     mask, dist_mat, bpres_actions, bpresent_time,
                                                     torch.arange(0, mask.shape[0], device=self.device))

            adj_mask = self.mu.adjacency_matrix(inputs.expand(beam_idx.shape[0], -1, -1),
                                                mask, dist_mat, bpres_actions, bpresent_time)
            # active_idx 表示还在候选搜索中的索引，标志是mask最后一列不为0
            # active_beam_idx = torch.nonzero(mask[:, -1]).squeeze(1)
            active_beam_idx = torch.nonzero(torch.any(mask, dim=1)).squeeze(1)
            # end_idx表示已经结束的索引，标志是最后一个掩码mask为0
            # end_beam_idx = torch.nonzero((mask[:, -1] == 0)).squeeze(1)
            end_beam_idx = torch.nonzero(~torch.any(mask, dim=1)).squeeze(1)

            if active_beam_idx.shape[0] > 0:
                final_log_probs.append(torch.index_select(bllog_probs, dim=0, index=end_beam_idx))
                final_actions.append(torch.index_select(blactions, dim=0, index=end_beam_idx))

                # ending seq info update
                h_step = torch.index_select(dec_hidden[0], dim=0, index=active_beam_idx)
                c_step = torch.index_select(dec_hidden[1], dim=0, index=active_beam_idx)
                dec_hidden = (h_step, c_step)

                mask = torch.index_select(mask, dim=0, index=active_beam_idx)
                adj_mask = torch.index_select(adj_mask, dim=0, index=active_beam_idx)

                bpresent_time = torch.index_select(bpresent_time, dim=0, index=active_beam_idx)
                bpres_actions = torch.index_select(bpres_actions, dim=0, index=active_beam_idx)
                enc_outputs = torch.index_select(enc_outputs, dim=0, index=active_beam_idx)

                emb1 = torch.index_select(emb1, dim=0, index=active_beam_idx)

                blactions = torch.index_select(blactions, dim=0, index=active_beam_idx)
                bllog_probs = torch.index_select(bllog_probs, dim=0, index=active_beam_idx)
                sum_log_probs = torch.index_select(sum_log_probs, dim=0, index=active_beam_idx)

            if done or not mask.any():
                break

            decoder_input = enc_outputs[torch.arange(0, bpres_actions.shape[0], device=self.device), bpres_actions]

            bdyn_inputs = self.dyn_feat.make_dynamic_feat(inputs.expand(beam_idx.shape[0], -1, -1), bpresent_time,
                                                          bpres_actions, dist_mat, active_beam_idx)
            emb2 = self.neuralnet.dyn_emb(bdyn_inputs)
            enc_inputs = torch.cat((emb1, emb2), dim=2)

            policy, dec_hidden, enc_outputs = self.neuralnet(enc_inputs, enc_outputs, adj_mask, decoder_input,
                                                             dec_hidden, mask)

            future_actions, log_probs, beam_idx = self.choose_actions(policy, sum_log_probs, mask, infer_type)

            # info update
            h_step = torch.index_select(dec_hidden[0], dim=0, index=beam_idx)
            c_step = torch.index_select(dec_hidden[1], dim=0, index=beam_idx)
            dec_hidden = (h_step, c_step)

            mask = torch.index_select(mask, dim=0, index=beam_idx)
            adj_mask = torch.index_select(adj_mask, dim=0, index=beam_idx)

            bpresent_time = torch.index_select(bpresent_time, dim=0, index=beam_idx)
            bpres_actions = torch.index_select(bpres_actions, dim=0, index=beam_idx)

            enc_outputs = torch.index_select(enc_outputs, dim=0, index=beam_idx)

            emb1 = torch.index_select(emb1, dim=0, index=beam_idx)

            blactions = torch.index_select(blactions, dim=0, index=beam_idx)
            bllog_probs = torch.index_select(bllog_probs, dim=0, index=beam_idx)
            sum_log_probs = torch.index_select(sum_log_probs, dim=0, index=beam_idx)

            sum_log_probs += log_probs.squeeze(0).detach()

        return torch.cat(final_actions, dim=0), torch.cat(final_log_probs, dim=0)

    def choose_actions(self, policy, sum_log_probs, mask, infer_type='stochastic'):

        _, seq_size = policy.size()  # replace beam_size with _
        nzn = torch.nonzero(mask, as_tuple=False).shape[0]
        sample_size = min(nzn, self.beam_size)

        logzero = 1.0  # torch.finfo().min
        lpolicy = policy.masked_fill(mask == 0, logzero).log()
        npolicy = sum_log_probs.unsqueeze(1) + lpolicy
        if infer_type == 'stochastic':
            nnpolicy = npolicy.masked_fill(mask == 0, -torch.inf).view(1, -1)

            m = Categorical(nnpolicy)
            gact_ind = torch.multinomial(nnpolicy, sample_size)
            log_select = m.log_prob(gact_ind)

        elif infer_type == 'greedy':
            nnpolicy = npolicy.masked_fill(mask == 0, -torch.inf).view(1, -1)

            _, gact_ind = nnpolicy.topk(sample_size, dim=1)  # 取前K个值
            prob = policy.view(-1)[gact_ind]

            log_select = prob.log()

        # beam_id为留下的实例
        beam_id = torch.floor_divide(gact_ind, seq_size).squeeze(0)
        # act_ind为动作索引
        act_ind = torch.fmod(gact_ind, seq_size)

        return act_ind, log_select, beam_id
