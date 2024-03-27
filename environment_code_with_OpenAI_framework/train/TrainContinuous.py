import copy
import random
from typing import Optional
import pygame
from pygame import gfxdraw
import gym
import math
from gym import spaces
import numpy as np
import pandas as pd
from gym.envs.train import tong_utils as utils
import pickle


class trainEnv(gym.Env):
    metadata = {
        "render_modes": ['human', 'rgb_array'],
        "render_fps": 60
    }

    def __init__(self):
        self.state = np.array([])
        self.easy_state = np.array([])
        self.info = np.array([])
        self.reward = 0
        self.energy_rate = 0
        self.station_i = 0
        self.p_time_i = 1
        root = '/Users/tongx2/opt/anaconda3/envs/ERL/lib/python3.8/site-packages/gym/envs/train/'
        df_slope = pd.read_csv(root+'data/parameter/slope.csv')
        df_speed_lim = pd.read_csv(root+'data/parameter/speed_limit.csv')
        df_train = pd.read_csv(root+'data/parameter/train.csv')
        df_plan_time = pd.read_csv(root+'data/parameter/time.csv')

        self.train_info = df_train.apply(lambda row: row.astype(float).to_dict(), axis=1).to_list()[0]
        self.station_num = len(df_slope.groupby('station'))
        self.station = [i for i in range(self.station_num)]

        self.n_seg_slope = dict(
            zip(self.station, (df_slope.groupby('station')['seg_slope'].apply(list).reset_index())['seg_slope'].tolist()))
        self.n_slope = dict(zip(self.station, (df_slope.groupby('station')['slope'].apply(list).reset_index())['slope'].tolist()))
        self.n_seg_v_lim = dict(zip(self.station, (df_speed_lim.groupby('station')['seg_v_lim'].apply(list).reset_index())[
            'seg_v_lim'].tolist()))
        self.n_v_lim = dict(
            zip(self.station, (df_speed_lim.groupby('station')['v_lim'].apply(list).reset_index())['v_lim'].tolist()))

        self.n_plan_time = dict(
            zip(self.station,
                (df_plan_time.groupby('station')['plan_time'].apply(list).reset_index())['plan_time'].tolist()))

        self.n_station_len = {s: seg[-1] for s, seg in self.n_seg_v_lim.items()}

        self.n_energy_eff_save = dict(zip(self.station, [[0.05 for _ in range(5)] for i in range(self.station_num)]))
        self.n_energy_thr = dict(zip(self.station, [[1500 for _ in range(5)] for i in range(self.station_num)]))

        with open(root+'data/mri/n_mri.pickle', 'rb') as f:
            self.n_mri = pickle.load(f)
        with open(root+'data/bound/n_speed_lb.pickle', 'rb') as f:
            self.n_speed_lb = pickle.load(f)
        with open(root+'data/bound/n_speed_ub.pickle', 'rb') as f:
            self.n_speed_ub = pickle.load(f)
        with open(root+'data/bound/n_speed_pmp.pickle', 'rb') as f:
            self.n_speed_pmp = pickle.load(f)
        with open(root+'data/reward_weight/n_energy_thr.pickle', 'rb') as f:
            self.n_energy_thr = pickle.load(f)
        with open(root+'data/reward_weight/n_energy_eff_save.pickle', 'rb') as f:
            self.n_energy_eff_save = pickle.load(f)

        self.n_energy_weight = dict(zip(self.station,
                                        [[3 * (thr / save) for save, thr in zip(self.n_energy_eff_save[i],
                                                                                self.n_energy_thr[i])] for i
                                         in range(self.station_num)]))

        self.n_psi_time = dict(zip(self.station, [[] for _ in range(self.station_num)]))
        for i in range(self.station_num):
            for j in range(len(self.n_speed_pmp[i])):
                self.n_psi_time[i].append(utils.get_line_step_time(self.n_speed_pmp[i][j]))

        self.n_speed_high = dict(zip(self.station, [[] for _ in range(self.station_num)]))
        self.n_speed_low = dict(zip(self.station, [[] for _ in range(self.station_num)]))
        for i in range(self.station_num):
            self.n_speed_low[i] = np.minimum(np.ones(len(self.n_mri[i][0]))*80/3.6, self.n_mri[i][0])
            self.n_speed_high[i] = self.n_mri[i][0]


        self.location = 0

        self.plan_time = self.n_plan_time[self.station_i][self.p_time_i]
        self.line_len = self.n_station_len[self.station_i]


        self.train_len = self.train_info['len']
        self.train_weight = self.train_info['mass']

        self.slope = self.n_slope[self.station_i]
        self.slope_seg = self.n_seg_slope[self.station_i]

        self.speed_lim = self.n_v_lim[self.station_i]
        self.speed_lim_seg = self.n_seg_v_lim[self.station_i]

        self.energy_thr = self.n_energy_thr[self.station_i][self.p_time_i]


        self.energy_weight = self.n_energy_weight[self.station_i][self.p_time_i]

        self.psi_ub = self.n_speed_ub[self.station_i][self.p_time_i]
        self.psi_lb = self.n_speed_lb[self.station_i][self.p_time_i]
        self.psi_me = self.n_speed_pmp[self.station_i][self.p_time_i]
        self.mri = self.n_mri[self.station_i][self.p_time_i]




        self.speed = np.zeros(self.line_len + 1)
        self.slope_step = utils.get_list(self.slope, self.slope_seg)
        self.next_slope = utils.get_next_info(self.slope_seg, self.slope, self.line_len)
        self.next_slope_remain_step = utils.get_remain_step(self.slope_seg)
        self.speed_limit = utils.get_list(self.speed_lim, self.speed_lim_seg)
        self.next_speed_limit = utils.get_next_info(self.speed_lim_seg, self.speed_lim, self.line_len)
        self.next_speed_limit_remain_step = utils.get_remain_step(self.speed_lim_seg)
        self.remain_times = np.zeros(self.line_len + 1)
        self.remain_times[0] = self.plan_time
        self.remain_distances = np.zeros(self.line_len + 1)
        for index in range(0, self.line_len + 1):
            self.remain_distances[index] = self.line_len - index
        self.remain_energy = np.zeros(self.line_len + 1)
        self.remain_energy[0] = self.energy_thr
        self.psi_time_shift = np.zeros(self.line_len + 1)
        self.psi_energy_shift = np.zeros(self.line_len + 1)
        self.psi_time = self.n_psi_time[self.station_i][self.p_time_i]



        self.speed_step = np.zeros(self.line_len + 1)
        self.km_speed_step = np.zeros(self.line_len + 1)
        self.gear_step = np.zeros(self.line_len + 1)
        self.time_step = np.zeros(self.line_len + 1)
        self.actual_times = np.zeros(self.line_len + 1)
        self.accelerated_step = np.zeros(self.line_len + 1)
        self.parking_accuracy = np.zeros(self.line_len + 1)
        self.energy = np.zeros((self.line_len + 1))
        self.acc_reward = np.zeros((self.line_len + 1))
        self.time_reward = np.zeros((self.line_len + 1))
        self.energy_reward = np.zeros((self.line_len + 1))
        self.over_steps = np.zeros((self.line_len + 1))
        self.low_steps = np.zeros((self.line_len + 1))
        self.gear_acc = np.zeros((self.line_len + 1))
        self.real_coasting = np.zeros((self.line_len + 1))
        self.time_error = np.zeros((self.line_len + 1))
        self.energy_save = np.zeros((self.line_len + 1))

        self.min_gear_action = 0
        self.max_gear_action = 1

        self.low_action = np.array(
            [self.min_gear_action])
        self.high_action = np.array(
            [self.max_gear_action])

        self.over_psi_steps = 0
        self.low_psi_steps = 0
        self.min_speed = 0
        self.max_speed = 400
        self.min_speed_limit = 0
        self.max_speed_limit = 400
        self.min_next_speed_limit = 0
        self.max_next_speed_limit = 400
        self.min_next_speed_limit_switch = 0
        self.max_next_speed_limit_switch = 20000
        self.min_current_slope = -100
        self.max_current_slope = 100
        self.min_next_slope = -100
        self.max_next_slope = 100
        self.min_next_slope_switch = 0
        self.max_next_slope_switch = 20000
        self.min_remaining_time = 0
        self.max_remaining_time = 4000
        self.min_remaining_distance = 0
        self._min_remaining_energy = -4000
        self._max_remaining_energy = 4000
        self.max_remaining_distance = 500000
        self.min_psi_ub = 0
        self.max_psi_ub = 400
        self.min_psi_lb = 0
        self.max_psi_lb = 400
        self.min_psi_time_shift = -2000
        self.max_psi_time_shift = 2000
        self.min_slope = -20
        self.max_slope = 20



        self.low_state = np.array(
            [self.min_speed, self.min_remaining_time, self.min_remaining_distance, self._min_remaining_energy,
             self.min_psi_lb, self.min_psi_ub, self.min_psi_time_shift, self.min_slope])
        self.high_state = np.array(
            [self.max_speed, self.max_remaining_time, self.max_remaining_distance, self._max_remaining_energy,
             self.max_psi_lb, self.max_psi_ub, self.max_psi_time_shift, self.max_slope])
        self.screen = None
        self.clock = None
        self.is_open = True
        self.surf = None
        self.height_y = utils.get_height(self.n_slope[self.station_i], self.n_seg_slope[self.station_i])

        self.action_space = spaces.Box(low=self.low_action, high=self.high_action, shape=(1,), dtype=np.float32)

        self.observation_space = spaces.Box(low=self.low_state, high=self.high_state,
                                            dtype=np.float32)

        self.train_step = 50
        self.seed()
        self.reset()

    def seed(self, seed=None):
        pass


    def step(self, action):
        gear_action = 0
        if self.info[0][self.location] == 0:
            max_gear_action = self.train_info['traction_140']
        else:
            max_gear_action = min(self.train_info['traction_140'], self.train_info['power'] / (
                        (1+self.train_info['mass_factor']) * self.train_info['mass'] * self.info[0][self.location]))

        gear_action = max_gear_action * action
        if self.info[0][self.location-1] < self.psi_lb[20000] and self.location > 20000:
            gear_action = 0



        origin_gear_action = gear_action

        is_done = False
        self.reward = 0

        for train_step in range(self.train_step):
            if is_done:
                break
            self.location += 1
            ub = self.psi_ub[self.location]
            lb = self.psi_lb[self.location]
            me = self.psi_me[self.location]
            if self.psi_ub[self.location - 1] == self.psi_lb[self.location - 1]:

                slope_accelerated = utils.get_slope_accelerated(self.location, self.slope_seg, self.slope,
                                                                self.train_info)
                last_speed = self.state[0][self.location - 1]

                speed = self.psi_ub[self.location]

                resist_acc = utils.get_accelerated(last_speed, 0, slope_accelerated, self.location, self.train_info)

                gear_action = max(min((speed ** 2 - last_speed ** 2) / 2 - resist_acc, max_gear_action), -1.2)

            else:

                if self.state[0][self.location - 1] >= self.psi_ub[self.location - 1]:

                    slope_accelerated = utils.get_slope_accelerated(self.location, self.slope_seg, self.slope,
                                                                    self.train_info)
                    last_speed = self.state[0][self.location - 1]

                    speed = self.psi_ub[self.location]

                    resist_acc = utils.get_accelerated(last_speed, 0, slope_accelerated, self.location, self.train_info)

                    # gear_action = max(min((speed ** 2 - last_speed ** 2) / (2 * pram.train_step) - resist_acc, max_gear_action), pram.min_action)
                    gear_action = max(min((speed ** 2 - last_speed ** 2) / 2 - resist_acc, max_gear_action), -1)
                    if gear_action > origin_gear_action and self.location != 1:
                        gear_action = origin_gear_action

                    if 8000 < self.location <= 17000:
                        gear_action = 0

                if self.state[0][self.location - 1] <= self.psi_lb[self.location - 1]:
                    slope_accelerated = utils.get_slope_accelerated(self.location, self.slope_seg, self.slope,
                                                                    self.train_info)
                    last_speed = self.state[0][self.location - 1]

                    speed = self.psi_lb[self.location]

                    resist_acc = utils.get_accelerated(last_speed, 0, slope_accelerated, self.location, self.train_info)

                    gear_action = max(min((speed ** 2 - last_speed ** 2) / 2 - resist_acc, max_gear_action), -1)

                    if gear_action < origin_gear_action:
                        gear_action = origin_gear_action

                    if 8000 < self.location <= 17000:
                        gear_action = max_gear_action


            gear_accelerated = gear_action
            slope_accelerated = utils.get_slope_accelerated(self.location, self.slope_seg, self.slope, self.train_info)
            last_speed = self.info[0][self.location - 1]
            accelerated = utils.get_accelerated(last_speed, gear_accelerated, slope_accelerated, self.location, self.train_info)
            speed = max(utils.get_speed(last_speed, accelerated, self.location), 0)

            move_time = utils.get_move_time(speed, last_speed, accelerated)
            accuracy = -1
            energy = utils.get_energy(gear_accelerated, self.train_info)
            remain_times = self.state[7][self.location - 1] - move_time
            actual_times = self.info[4][self.location - 1] + move_time
            actual_energy = self.info[7][self.location - 1] + energy



            remain_energy = self.easy_state[3][self.location - 1] - energy
            shift_time = self.psi_time[self.location] - actual_times

            done = bool(self.location == self.line_len)
            is_done = done


            self.state[0][self.location] = speed
            self.state[7][self.location] = remain_times


            self.easy_state[0][self.location] = speed
            self.easy_state[1][self.location] = remain_times
            self.easy_state[3][self.location] = remain_energy
            self.easy_state[6][self.location] = shift_time
            self.easy_state[7][self.location] = self.slope_step[self.location]




            self.info[0][self.location] = speed
            self.info[1][self.location] = speed * 3.6
            self.info[2][self.location] = action
            self.info[3][self.location] = move_time
            self.info[4][self.location] = actual_times
            self.info[5][self.location] = accelerated
            if done:
                if speed == 0:
                    accuracy = math.fabs(self.line_len - ((self.location - 1) + (last_speed ** 2) / (-2 * accelerated)))
                else:
                    accuracy = speed * speed / (-2 * accelerated)

            self.info[6][self.location] = accuracy
            self.info[7][self.location] = actual_energy


            self.reward = 0


            if done:

                if math.fabs(self.info[6][self.location]) <= 5:
                    acc_reward = 0
                else:
                    acc_reward = - self.info[6][self.location]
                self.reward += acc_reward
                self.info[10][self.location] = acc_reward

                time_error = math.fabs(self.plan_time - self.info[4][self.location])
                time_reward = 3 - (3*self.plan_time/4) * time_error / self.plan_time
                self.info[17][self.location] = self.plan_time - self.info[4][self.location]
                self.reward += time_reward
                self.info[11][self.location] = time_reward

                is_error = False
                if is_error:
                    acc_shift = min(self.train_info['traction_140'], self.train_info['power'] / (
                        (1 + self.train_info['mass_factor']) * self.train_info['mass'] * self.info[0][int(self.line_len/2)]))
                    energy_error = time_error*self.info[0][
                        int(self.line_len/2)]*utils.get_energy(acc_shift, self.train_info)
                    if time_error <= 4:
                        energy_error = 0
                    if self.info[7][self.location] >= self.energy_thr:
                        energy_reward = -3 + self.energy_weight * (
                                    self.energy_thr - self.info[7][self.location] - energy_error) / self.energy_thr
                    else:
                        energy_reward = min(
                            self.energy_weight * (self.energy_thr - self.info[7][self.location] - energy_error) / self.energy_thr, 5)
                else:
                    if self.info[7][self.location] >= self.energy_thr:
                        energy_reward = -3 + self.energy_weight * (self.energy_thr - self.info[7][self.location]) / self.energy_thr
                    else:
                        energy_reward = min(
                            self.energy_weight * (self.energy_thr - self.info[7][self.location]) / self.energy_thr, 5)
                self.info[16][self.location] = self.energy_thr - self.info[7][self.location]
                self.reward += energy_reward
                self.info[12][self.location] = energy_reward
                self.info[13][self.location] = self.over_psi_steps
                self.info[14][self.location] = self.low_psi_steps


            else:

                if speed * 3.6 == self.info[8][self.location] and last_speed != 0 and origin_gear_action > gear_action:
                    if self.location < self.line_len - 10:
                        self.over_psi_steps += 1

                if speed * 3.6 == self.info[9][self.location] and last_speed != 0 and origin_gear_action < gear_action:
                    if self.location < self.line_len - 10:
                        self.low_psi_steps += 1



        self.info[15][self.location] = gear_action


        reward = self.reward
        state = np.array([index[self.location] for index in self.easy_state], dtype=np.float32)
        info = np.array([index[self.location] for index in self.info], dtype=np.float32)

        return state, reward, is_done, info



    def reset(
            self,
            *,
            seed: Optional[int] = None,
            return_info: bool = False,
            options: Optional[dict] = None,
    ):
        super().reset(seed=seed)

        self.height_y = utils.get_height(self.n_slope[self.station_i], self.n_seg_slope[self.station_i])

        self.location = 0
        self.reward = 0

        self.plan_time = self.n_plan_time[self.station_i][self.p_time_i]
        self.line_len = self.n_station_len[self.station_i]

        self.slope = self.n_slope[self.station_i]
        self.slope_seg = self.n_seg_slope[self.station_i]

        self.speed_lim = self.n_v_lim[self.station_i]
        self.speed_lim_seg = self.n_seg_v_lim[self.station_i]

        self.energy_thr = self.n_energy_thr[self.station_i][self.p_time_i]
        self.energy_weight = self.n_energy_weight[self.station_i][self.p_time_i]

        self.psi_ub = self.n_speed_ub[self.station_i][self.p_time_i]
        self.psi_lb = self.n_speed_lb[self.station_i][self.p_time_i]
        self.psi_me = self.n_speed_pmp[self.station_i][self.p_time_i]
        self.mri = self.n_mri[self.station_i][self.p_time_i]


        self.over_psi_steps = 0
        self.low_psi_steps = 0

        self.speed = np.zeros(self.line_len + 1)
        self.slope_step = utils.get_list(self.slope, self.slope_seg)
        self.next_slope = utils.get_next_info(self.slope_seg, self.slope, self.line_len)
        self.next_slope_remain_step = utils.get_remain_step(self.slope_seg)
        self.speed_limit = utils.get_list(self.speed_lim, self.speed_lim_seg)
        self.next_speed_limit = utils.get_next_info(self.speed_lim_seg, self.speed_lim, self.line_len)
        self.next_speed_limit_remain_step = utils.get_remain_step(self.speed_lim_seg)
        self.remain_times = np.zeros(self.line_len + 1)
        self.remain_times[0] = self.plan_time
        self.remain_distances = np.zeros(self.line_len + 1)
        for index in range(0, self.line_len + 1):
            self.remain_distances[index] = self.line_len - index
        self.remain_energy = np.zeros(self.line_len + 1)
        self.remain_energy[0] = self.energy_thr
        self.psi_time_shift = np.zeros(self.line_len + 1)
        self.psi_energy_shift = np.zeros(self.line_len + 1)
        self.psi_time = self.n_psi_time[self.station_i][self.p_time_i]

        self.state = np.array([
            np.array(self.speed),
            np.array(self.speed_limit),
            np.array(self.next_speed_limit),
            np.array(self.next_speed_limit_remain_step),
            np.array(self.slope_step),
            np.array(self.next_slope),
            np.array(self.next_slope_remain_step),
            np.array(self.remain_times),
            np.array(self.remain_distances)
        ])
        self.easy_state = np.array([
            np.array(self.speed),
            np.array(self.remain_times),
            np.array(np.array(self.remain_distances)),
            np.array(self.remain_energy),
            np.array(self.psi_lb),
            np.array(self.psi_ub),
            np.array(self.psi_time_shift),
            np.array(np.zeros(len(self.speed))),

        ])
        self.speed_step = np.zeros(self.line_len + 1)
        self.km_speed_step = np.zeros(self.line_len + 1)
        self.gear_step = np.zeros(self.line_len + 1)
        self.time_step = np.zeros(self.line_len + 1)
        self.actual_times = np.zeros(self.line_len + 1)
        self.accelerated_step = np.zeros(self.line_len + 1)
        self.parking_accuracy = np.zeros(self.line_len + 1)
        self.energy = np.zeros((self.line_len + 1))
        self.acc_reward = np.zeros((self.line_len + 1))
        self.time_reward = np.zeros((self.line_len + 1))
        self.energy_reward = np.zeros((self.line_len + 1))
        self.over_steps = np.zeros((self.line_len + 1))
        self.low_steps = np.zeros((self.line_len + 1))
        self.gear_acc = np.zeros((self.line_len + 1))
        self.time_error = np.zeros((self.line_len + 1))
        self.energy_save = np.zeros((self.line_len + 1))

        self.info = np.array([
            np.array(self.speed_step),
            np.array(self.km_speed_step),
            np.array(self.gear_step),
            np.array(self.time_step),
            np.array(self.actual_times),
            np.array(self.accelerated_step),
            np.array(self.parking_accuracy),
            np.array(self.energy),
            np.array(self.psi_ub),
            np.array(self.psi_lb),
            np.array(self.acc_reward),
            np.array(self.time_reward),
            np.array(self.energy_reward),
            np.array(self.over_steps),
            np.array(self.low_steps),
            np.array(self.gear_acc),
            self.energy_save,
            self.time_error

        ])
        return np.array([index[0] for index in self.easy_state], dtype=np.float32)


    def render(self, mode='human'):
        screen_w = 700
        screen_h = 650
        x_zoom = screen_w / (self.line_len + 1)
        y_zoom = screen_h / 310

        shift_scale_y = screen_h * 0.25
        multiple_scale_y = 1
        shift_scale_x = 12
        multiple_scale_x = 0.98

        if self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode((screen_w, screen_h))
        if self.clock is None:
            self.clock = pygame.time.Clock()

        self.surf = pygame.Surface((screen_w, screen_h))
        self.surf.fill((255, 255, 255))

        v_limit = utils.get_py_draw_line(self.state[8][self.line_len::-1], self.state[1]
                                         , x_rate=0.98, y_rate=0.5, x_shift=0.01, y_shift=0.25, line_len=self.line_len)
        pygame.draw.aalines(self.surf, points=v_limit, closed=False, color=(100, 0, 0))

        mri = utils.get_py_draw_line(self.state[8][self.line_len::-1], self.mri*3.6
                                     , x_rate=0.98, y_rate=0.5, x_shift=0.01, y_shift=0.25, line_len=self.line_len)
        pygame.draw.aalines(self.surf, points=mri, closed=False, color=(255, 0, 0))

        psi_ub = utils.get_py_draw_line(self.state[8][self.line_len::-1], self.psi_ub*3.6
                                        , x_rate=0.98, y_rate=0.5, x_shift=0.01, y_shift=0.25, line_len=self.line_len)
        pygame.draw.aalines(self.surf, points=psi_ub, closed=False, color='orange')

        psi_lb = utils.get_py_draw_line(self.state[8][self.line_len::-1], self.psi_lb*3.6
                                        , x_rate=0.98, y_rate=0.5, x_shift=0.01, y_shift=0.25, line_len=self.line_len)
        pygame.draw.aalines(self.surf, points=psi_lb, closed=False, color=(88, 142, 85))

        psi_me = utils.get_py_draw_line(self.state[8][self.line_len::-1], self.psi_me*3.6
                                        , x_rate=0.98, y_rate=0.5, x_shift=0.01, y_shift=0.25, line_len=self.line_len)
        pygame.draw.aalines(self.surf, points=psi_me, closed=False, color=(104, 197, 255))

        x_axis = utils.get_py_draw_line(self.state[8][self.line_len::-1], np.repeat(0.25, self.line_len)
                                        , x_rate=0.98, y_rate=0.5, x_shift=0.01, y_shift=0.25, line_len=self.line_len)
        pygame.draw.aalines(self.surf, points=x_axis, closed=False, color=(0, 0, 0))

        y_axis = utils.get_py_draw_line(np.array([0, 0]), np.array([0, max(self.speed_limit)])
                                        , x_rate=0.98, y_rate=0.5, x_shift=0.01, y_shift=0.25, line_len=self.line_len)
        pygame.draw.aalines(self.surf, points=y_axis, closed=False, color=(0, 0, 0))


        if max(self.speed_limit) > 100:
            scale_num = max(self.speed_limit) // 25
            scale_rate = 25
        else:
            scale_num = (max(self.speed_limit) // 10)+1
            scale_rate = 10
        for scale_i in range(1, int(scale_num)):
            y_axis_scale = utils.get_py_draw_line(np.array([0, 20]), np.array([scale_rate*scale_i, scale_rate*scale_i])
                                                  , x_rate=0.98, y_rate=0.5, x_shift=0.01, y_shift=0.25, line_len=self.line_len)
            pygame.draw.aalines(self.surf, points=y_axis_scale, closed=False, color=(0, 0, 0))

        slope = utils.get_py_draw_line(self.state[8][self.line_len::-1], self.height_y
                                       , x_rate=0.98, y_rate=0.13, x_shift=0.01, y_shift=0.05, line_len=self.line_len)
        pygame.draw.aalines(self.surf, points=slope, closed=False, color=(0, 0, 255))


        loc = self.location

        gfxdraw.filled_circle(self.surf, int(((loc * 0.98) * x_zoom + 0.01*screen_w)),
                              int(((self.height_y[loc] * 0.13) * y_zoom + 0.05*screen_h)), 2, (255, 0, 0))

        if self.line_len != loc:
            loc_xs = self.state[8][self.line_len:self.line_len - loc - 1:-1]
            gear_xs = self.state[8][self.line_len:self.line_len - loc - 1:-1]
        else:
            loc_xs = self.state[8][self.line_len::-1]
            gear_xs = self.state[8][self.line_len::-1]

        loc_ = utils.get_py_draw_line(loc_xs, self.state[0][0:loc + 1] * 3.6
                                      , x_rate=0.98, y_rate=0.5, x_shift=0.01, y_shift=0.25, line_len=self.line_len)
        pygame.draw.aalines(self.surf, points=loc_, closed=False, color=(0, 0, 0))

        gear_zero = utils.get_py_draw_line(np.array([0, self.line_len]), np.array([0, 0])
                                           , x_rate=0.98, y_rate=1, x_shift=0.01, y_shift=0.15, line_len=self.line_len)
        pygame.draw.aalines(self.surf, points=gear_zero, closed=False, color=(0, 0, 0))



        gear = utils.get_py_draw_line(gear_xs, self.info[2][0:loc + 1]
                                      , x_rate=0.98, y_rate=7, x_shift=0.01, y_shift=0.15, line_len=self.line_len)
        pygame.draw.aalines(self.surf, points=gear, closed=False, color=(0, 0, 255))

        self.surf = pygame.transform.flip(self.surf, False, True)
        self.screen.blit(self.surf, (0, 0))

        my_font = pygame.font.SysFont("pingfang", 16)

        state_name = ["v km/h", "vlim", "n_vlim", "rmn_vlim", "g ‰", "n_g", "rmn_g", "rmn_t s",
                      "剩余距离m"]
        for state_name_i in range(len(state_name)):
            text = state_name[state_name_i]
            textImage = my_font.render(text, True, (0, 0, 0))
            self.screen.blit(textImage, (0 + 77 * state_name_i , 5))

        for state_i in range(len(self.state)):
            text = str(round(self.state[state_i][loc], 2))
            textImage = my_font.render(text, True, (0, 0, 0))
            self.screen.blit(textImage, (0 + 77 * state_i , 30))

        info_name = ["v m/s", "v km/s", "a m/s^2", "t s", "At s", "a", "m", "E "]
        for info_name_i in range(len(info_name)):
            text = info_name[info_name_i]
            textImage = my_font.render(text, True, (0, 0, 0))
            self.screen.blit(textImage, (77 * info_name_i, 55))

        for info_i in range(len(info_name)):
            text = str(round(self.info[info_i][loc], 2))
            textImage = my_font.render(text, True, (0, 0, 0))
            self.screen.blit(textImage, (0 + 77 * info_i, 80))

        text = "R"
        textImage = my_font.render(text, True, (0, 0, 0))
        self.screen.blit(textImage, (0 + 77 * 6, 105))

        all_reward = str(round(self.info[10][loc] + self.info[11][loc] + self.info[12][loc], 4))
        textImage = my_font.render(all_reward, True, (0, 0, 0))
        self.screen.blit(textImage, (77 * 6, 130))

        info_name = ["_", "tR", "eR", "_", "tC", "eC"]
        for info_name_i in range(len(info_name)):
            text = info_name[info_name_i]
            textImage = my_font.render(text, True, (0, 0, 0))
            self.screen.blit(textImage, (0 + 77 * info_name_i, 105))

        info_list = [0, 0, 0]
        for info_i in range(3):
            text = str(round(self.info[10 + info_i][loc], 2))
            info_list[info_i] = math.fabs(self.info[10 + info_i][loc])
            textImage = my_font.render(text, True, (0, 0, 0))
            self.screen.blit(textImage, (77 * info_i, 130))

        for info_i in range(3, 6):
            if info_list[info_i - 3] != 0:
                text = str(round((info_list[info_i - 3] * 100) / (info_list[0] + info_list[1] + info_list[2]), 2)) + '%'
            else:
                text = str(0)
            textImage = my_font.render(text, True, (0, 0, 0))
            self.screen.blit(textImage, (77 * info_i, 130))

        psi_info_name = ["Fast", "Lower"]
        for psi_info_name_i in range(len(psi_info_name)):
            text = psi_info_name[psi_info_name_i]
            textImage = my_font.render(text, True, (0, 0, 0))
            self.screen.blit(textImage, (77 * (psi_info_name_i + 7), 105))

        over_psi_steps = str(round(self.over_psi_steps, 0))
        textImage = my_font.render(over_psi_steps, True, (0, 0, 0))
        self.screen.blit(textImage, (77 * 7, 130))

        low_psi_steps = str(round(self.low_psi_steps, 0))
        textImage = my_font.render(low_psi_steps, True, (0, 0, 0))
        self.screen.blit(textImage, (77 * 8, 130))

        if mode == "human":
            self.clock.tick(self.metadata["render_fps"])
            pygame.display.flip()

        done = bool(loc == self.line_len)
        if done:
            pygame.time.wait(1000)

        if mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
            )
        else:
            return self.is_open

    def close(self):
        if self.screen is not None:
            pygame.quit()
            self.is_open = False


if __name__ == '__main__':
    train_env = trainEnv()


    pd.set_option('expand_frame_repr', False)
    pd.set_option('colheader_justify', 'left')
    state1, reward1, done1, info1 = 0, 0, 0, 0
    v_step = []
    loc_step = []
    t_step = []
    e_step = []

    '''
    for i in range(0, len(psi_action)):
        state1, reward1, done1, info1 = train_env.step([psi_action[i], -1])
        loc_step.append(i+1)
        v_step.append(info1[1])
        t_step.append(info1[4])
        e_step.append(info1[7])

    df_data = {
        '位置': loc_step,
        '速度': v_step,
        '时间': t_step,
        '能耗': e_step
    }
    df_data_1 = {
        'msi': msi
    }
    df = pd.DataFrame(df_data_1)
    df.to_excel('./astpsi_output_step.xlsx', index=True)

    '''

    for i in range(100):
        done2 = False

        steps = int(input("Step"))
        action = float(input("Power"))

        i = 0
        for step in range(steps):
            i += 1
            state1, reward1, done1, info1 = train_env.step(action)
            if i%10 == 0:
                train_env.render(mode="human")
            if done1:
                done2 = True
                break

        # title = ["speed", "limit", "n-limit", "limit-step", "slope", "n-slope", "slope-step", "re-time", "re-distance"]
        # a = pd.DataFrame([state1], columns=title)
        # print(a, "\n")
        # title1 = ["m/s", "km/h", "gear", "time", "global-time", "a", "acc", "energy", "fsb_max", "fsb_min"]
        # b = pd.DataFrame([info1], columns=title1)
        # print(b)
        # print(train_env.location)
        # print(done1)
        # print(reward1)
        if done2:
            break
