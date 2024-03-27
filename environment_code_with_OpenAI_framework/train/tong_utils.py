import numpy as np
import copy
import math
from collections import defaultdict
import matplotlib.pyplot as plt
# import time as get_time
from scipy.interpolate import interp1d


def get_render_list(seg, info):
    list = [[], []]
    for i in range(len(seg)-1):
        list[0].append(seg[i])
        list[1].append(info[i])
        list[0].append(seg[i+1])
        list[1].append(info[i])
    return list


def get_train_list(list_step, n_station):

    n_list = []
    list_step_ = [sublist[1:] for sublist in list_step]
    list_step_rev_ = [list(reversed(row)) for row in list_step_]

    for stops in n_station:
        merged_data = []
        for i in range(len(stops) - 1):
            start_station = stops[i]
            end_station = stops[i + 1]
            if end_station > start_station:
                merged_data.append(sum(list_step_[start_station: end_station], []))
                merged_data[-1].insert(0, 0)
            else:
                merged_data.append(sum(list_step_rev_[end_station: start_station], []))
                merged_data[-1].insert(0, 0)
        n_list.append(merged_data)
    return n_list


def get_train_list_(fast_profile, min_E_profile, n_station):

    n_list_0 = []
    n_list_1 = []
    fast_profile_ = [sublist[:-1] for sublist in fast_profile]
    fast_profile_rev_ = [list(reversed(row)) for row in fast_profile]
    min_E_profile_ = [sublist[:-1] for sublist in min_E_profile]
    min_E_profile_rev_ = [list(reversed(row)) for row in min_E_profile]

    for stops in n_station:
        merged_data_0 = []
        merged_data_1 = []
        for i in range(len(stops) - 1):
            start_station = stops[i]
            end_station = stops[i + 1]
            if end_station > start_station:
                fast = sum(fast_profile_[start_station: end_station], [])
                min_E = sum(min_E_profile_[start_station: end_station], [])
                for j in range(1000, len(fast) - 1000):
                    if fast[j] <= 80/3.6:
                        fast[j] = 80/3.6
                    if min_E[j] <= 80/3.6:
                        min_E[j] = 80/3.6
                merged_data_0.append(fast)
                merged_data_0[-1].append(0)
                merged_data_1.append(min_E)
                merged_data_1[-1].append(0)
            else:
                fast = sum(fast_profile_rev_[end_station: start_station], [])
                min_E = sum(min_E_profile_rev_[end_station: start_station], [])
                for j in range(1000, len(fast) - 1000):
                    if fast[j] <= 80/3.6:
                        fast[j] = 80/3.6
                    if min_E[j] <= 80/3.6:
                        min_E[j] = 80/3.6
                merged_data_0.append(fast)
                merged_data_0[-1].append(0)
                merged_data_1.append(min_E)
                merged_data_1[-1].append(0)
        n_list_0.append(merged_data_0)
        n_list_1.append(merged_data_1)
    return n_list_0, n_list_1


def get_n_list(info_step, n_station_loc):
    n_info_step = []
    for i in range(len(n_station_loc)):
        n_info_step.append([])
        for j in range(len(n_station_loc[i])-1):
            n_info_step[i].append([])
            n_info_step[i][j].append(0)
            if n_station_loc[i][0] < n_station_loc[i][1]:
                n_info_step[i][j].extend(info_step[n_station_loc[i][j]:n_station_loc[i][j+1]+1])
            else:
                n_info_step[i][j].extend(info_step[n_station_loc[i][j+1]:n_station_loc[i][j]+1][::-1])
    return n_info_step



def get_list(info, start_end):

    line_len = start_end[-1]
    output_list = [0]
    point = 0
    present_end = start_end[point + 1]
    present = info[point]

    for location in range(1, line_len + 1):
        if location <= present_end:
            output_list.append(present)
        elif location > present_end:
            if point == len(start_end) - 1:
                present = info[point]
                output_list.append(present)
                break
            point += 1
            present_end = start_end[point + 1]
            present = info[point]
            output_list.append(present)

    return output_list


def get_remain_step(start_end):

    output_list = [start_end[1]]
    for grs_i in range(len(start_end) - 1):
        for remain_step in range(start_end[grs_i + 1] - start_end[grs_i] - 1, -1, -1):
            output_list.append(remain_step)
    return output_list


# 获得下一个坡度或者速度限制
def get_next_info(start_end, info, line_len):


    output = np.zeros(line_len + 1)
    point = 1
    for l in range(1, line_len + 1):
        if l > start_end[point]:
            point += 1
        output[l] = info[point]
    return output


def get_now_info(start_end, info, line_len):

    output = np.zeros(line_len + 1)
    point = 1
    for l in range(1, line_len + 1):
        if l > start_end[point]:
            point += 1
        output[l] = info[point - 1]
    output[-1] = 0
    return output


def get_energy(gear_accelerated, train_info):

    mass = train_info['mass']
    if gear_accelerated < 0:
        fa = 0
    else:
        fa = math.fabs(gear_accelerated * mass) / 3600
    action_energy = fa
    return action_energy


def distributed_re_energy(re_energy, current_loc, other_loc, train_info):

    re_conv_loss_a = train_info['re_conv_loss_a']
    re_conv_loss_b = train_info['re_conv_loss_b']
    accum_len = 0
    ob_re_energy = other_loc
    for train, loc in other_loc.items():
        other_loc[train] = abs(loc-current_loc)/1000
        accum_len += other_loc[train]
    for train, current_len in other_loc.items():
        # 该车获得的再生能量 = 再生能量*分配比例*衰减率
        ob_re_energy[train] = re_energy*(current_len/accum_len)*(re_conv_loss_a*(current_len**2)+re_conv_loss_b)
    return ob_re_energy


def get_re_energy(speed, gear_accelerated, train_info):

    fa = 0
    c_70 = train_info['re_energy_a_70']
    c_294 = train_info['re_energy_b2_294']
    x_294 = train_info['re_energy_b1_294']
    c_350 = train_info['re_energy_c2_350']
    x_350 = train_info['re_energy_c1_350']
    if speed*3.6 < 70:
        fa = c_70/3600
    elif speed*3.6 < 294:
        fa = (x_294*speed*3.6+c_294)/3600
    elif speed*3.6 < 350:
        fa = (x_350*speed*3.6+c_350)/3600

    if gear_accelerated > 0:
        fa = 0

    action_energy = fa
    return action_energy


def get_jerk(last_a, a):
    return abs(last_a-a)





def get_slope_acc(loc, P, slope_seg, slope, train_info):
    train_len = train_info['len']
    slope_accelerated = 0.0
    G = 9.81
    if loc <= train_len + slope_seg[P]:
        if P == 0:
            slope_accelerated = (slope[P] * G * loc) / (train_len * 1000)
        else:
            slope_accelerated = (slope[P] * G * (loc - slope_seg[P])) / (train_len * 1000) + (slope[P - 1] * G *
                                (train_len + slope_seg[P] - loc)) / (train_len * 1000)
    if loc > train_len + slope_seg[P]:
        slope_accelerated = slope[P] * G / 1000


    return slope_accelerated


def get_slope_accelerated(location, slope_seg, slope, train_info):

    train_len = train_info['len']
    slope_accelerated = 0.0
    G = 9.81
    for section in range(len(slope_seg)):
        if slope_seg[section] < location & location <= slope_seg[section + 1]:
            if location <= train_len + slope_seg[section]:
                if section == 0:
                    slope_accelerated = (slope[section] * G * location) / (train_len * 1000)
                    break
                else:
                    slope_accelerated = (slope[section] * G * (location - slope_seg[section])) / (
                            train_len * 1000) \
                                        + (slope[section - 1] * G * (
                            train_len + slope_seg[section] - location)) / (
                                            train_len * 1000)
                    break

            if location > train_len + slope_seg[section]:
                slope_accelerated = slope[section] * G / 1000
                break

    return slope_accelerated


def get_basic_acc(loc, last_v, train_info):

    # return gear_accelerated - slope_accelerated - ( .drag_coefficient_a + .drag_coefficient_b * last_speed * 3.6 + .drag_coefficient_c * last_speed * 3.6 * last_speed * 3.6) * G / 1000
    co_a = train_info['drag_coefficient_a']
    co_b = train_info['drag_coefficient_b']
    co_c = train_info['drag_coefficient_c']
    mass = train_info['mass']

    G = 9.81
    if loc == 1:
        return 0
    else:
        basic_acc = (co_a + co_b * last_v * 3.6 + co_c * (last_v * 3.6) ** 2) / mass
        return basic_acc



def get_accelerated(last_v, gear_acc, slope_acc, loc, train_info):

    # return gear_accelerated - slope_accelerated - ( .drag_coefficient_a + .drag_coefficient_b * last_speed * 3.6 + .drag_coefficient_c * last_speed * 3.6 * last_speed * 3.6) * G / 1000
    co_a = train_info['drag_coefficient_a']
    co_b = train_info['drag_coefficient_b']
    co_c = train_info['drag_coefficient_c']
    mass = train_info['mass']

    G = 9.81
    if loc == 1:
        return gear_acc - slope_acc
    else:
        basic_acc = (co_a + co_b * last_v * 3.6 + co_c * (last_v * 3.6) ** 2) / mass
        return gear_acc - slope_acc - basic_acc


def get_basic_accelerated(last_speed, loc, train_info):

    co_a = train_info['drag_coefficient_a']
    co_b = train_info['drag_coefficient_b']
    co_c = train_info['drag_coefficient_b']
    G = 9.81
    if loc == 1:
        return 0
    else:
        return (
                    co_a + co_b * last_speed * 3.6 + co_c * last_speed * 3.6 * last_speed * 3.6) * G / 1000


def get_speed(last_speed, accelerated, loc):

    if loc == 1:
        return math.sqrt(math.fabs(last_speed * last_speed + 2 * accelerated))
    else:
        if last_speed * last_speed + 2 * accelerated >= 0:
            speed = math.sqrt(math.fabs(last_speed * last_speed + 2 * accelerated))
        else:
            speed = 0
    return speed



def get_move_time(speed, last_speed, accelerated):

    if last_speed != speed:
        time = math.fabs((speed - last_speed) / accelerated)
    else:
        time = math.fabs(1 / speed)
    return time


def get_py_draw_line(x, y, x_rate=0.5, y_rate=0.5, x_shift=0.25, y_shift=0.25, line_len=40000):
    screen_width = 700
    screen_height = 650
    x_zoom = screen_width / (line_len + 1)
    y_zoom = screen_height / 310

    if not isinstance(x, np.ndarray):
        x = np.array(x)
    if not isinstance(y, np.ndarray):
        y = np.array(y)
    speed_limit_xs = x * x_zoom
    speed_limit_ys = y * y_zoom
    speed_limit_xys = \
        list(zip((speed_limit_xs * x_rate + screen_width * x_shift),
                 (speed_limit_ys * y_rate + screen_height * y_shift)))
    return speed_limit_xys


def get_line_to_pygame(x, y, x_max, y_max, width, height, shift_x=0, shift_y=0):
    if not isinstance(x, np.ndarray):
        x = np.array(x)
    if not isinstance(y, np.ndarray):
        y = np.array(y)
    x = shift_x + x * width/(x_max + 1)
    y = shift_y + height - y * height/(y_max + 1)
    xy = list(zip(x, y))
    return xy

def get_point_to_pygame(x, y, x_max, y_max, width, height, shift_x=0, shift_y=0):
    if not isinstance(x, np.ndarray):
        x = np.array(x)
    if not isinstance(y, np.ndarray):
        y = np.array(y)
    x = shift_x + x * width/(x_max + 1)
    y = shift_y + height - y * height/(y_max + 1)
    return x, y

def get_height(slope, slope_seg):

    height = [0]
    for slope_i in range(len(slope) - 1):
        height.append(round(slope[slope_i] + height[slope_i], 3))
    height_y = [0]
    for height_i in range(len(height) - 1):
        space = int(float(slope_seg[height_i + 1] - slope_seg[height_i]))
        start = float(height[height_i])
        end = float(height[height_i + 1])
        temp = np.linspace(start, end, space)
        height_y = np.concatenate((height_y, temp))
    return height_y


def get_length(point):

    length = []
    for p_i in range(len(point) - 1):
        length.append(point[p_i + 1] - point[p_i])
    return length


def get_avg_v(len_seg, time_seg):

    avg_v = []
    for len_i, time_i in zip(len_seg, time_seg):  # 遍历每个段的长度和时间
        avg_v.append(len_i / time_i)  # 计算平均速度并添加到列表中
    return avg_v


def get_v_sort(avg_v):

    seg = np.arange(0, len(avg_v), 1)  # 初始化 seg 数组，包含从 0 到 len(avg_v)-1 的整数
    v_sort = copy.deepcopy(avg_v)  # 复制 avg_v 到 v_sort

    for i in range(len(seg)):
        for j in range(0, len(seg) - i - 1):
            if v_sort[j] < v_sort[j + 1]:  # 如果前一个元素小于后一个元素
                v_sort[j], v_sort[j + 1] = v_sort[j + 1], v_sort[j]  # 交换两个元素的位置
                seg[j], seg[j + 1] = seg[j + 1], seg[j]  # 在 seg 数组中也交换对应元素的位置

    for i in range(len(v_sort)-1):
        if v_sort[i] != v_sort[i+1] and math.fabs(v_sort[i+1] - v_sort[i]) <= 0.0000000001:
            v_sort[i+1] = v_sort[i]
    return seg, v_sort


def get_block(seg, v_sort):


    groups = defaultdict(list)
    for i in range(len(v_sort)):
        groups[v_sort[i]].append(seg[i])  # 将 seg 中的元素添加到对应的 v_sort 值的列表中
    block = []
    for key in sorted(groups.keys(), reverse=True):  # 按照键值从大到小遍历 groups 字典
        block.append(groups[key])  # 将每个键对应的列表添加到 block 中
    return block


def get_block_sum_len(block, seg_len):

    block_sum_len = []
    for group in block:  # 遍历 block 中的每个块
        group_sum = 0
        for bsl_index in group:  # 遍历组内每个段
            group_sum += seg_len[bsl_index]  # 将 seg_len 中对应位置的元素累加到组内元素之和中
        block_sum_len.append(group_sum)  # 将组内元素之和添加到 block_sum_len 列表中
    return block_sum_len


def get_block_len(block, seg_len):

    block_len = []
    for group in block:  # 遍历 block 中的每个块
        group_values = []
        for bl_index in group:  # 遍历组内每个段
            group_values.append(seg_len[bl_index])  # 将 seg_len 中对应位置的元素添加到组内元素列表中
        block_len.append(group_values)  # 将组内元素列表添加到 block_len 列表中
    return block_len


def get_block_avg_v(block, avg_v):

    block_avg_v = []
    for i in range(len(block)):  # 遍历 block 中的每个块
        block_avg_v.append(avg_v[block[i][0]])  # 获得每个块的平均速度
    return block_avg_v


def allocate_runtime(seg_time, block, block_len, block_sum_len, rs_time):

    first_block = block[0]
    first_block_len = block_len[0]
    first_block_sum_len = block_sum_len[0]
    for i in range(len(first_block)):  # 遍历第一个组中的每个元素
        seg_index = first_block[i]  # 获取元素在 seg_time 中的位置
        seg_length = first_block_len[i]  # 获取元素对应的长度
        allocated_time = (seg_length / first_block_sum_len) * rs_time  # 计算分配给该元素的时间
        seg_time[seg_index] += allocated_time  # 将分配的时间累加到 seg_time 中对应位置的元素上
    return seg_time



def get_area_list(seg, info, station):
    area_seg_speed_lim = []
    area_speed_lim = []


    for s in sorted(station):
        if s not in seg:
            # 找到要插入的位置
            index = next(x[0] for x in enumerate(seg) if x[1] > s)
            # 插入 station 到 seg
            seg.insert(index, s)
            # 插入相应位置的速度限制到 info
            info.insert(index, info[index - 1])

    index = 0

    for i in range(len(station)-1):
        if station[0] < station[1]:
           start = station[i]
           end = station[i+1]
        else:
            start = station[i]
            end = station[i+1]
        area_seg = []
        area = []
        while True:
            area_seg.append(seg[index]-start)
            area.append(info[index])
            index += 1
            if seg[index] == end:
                area_seg.append(end-start)
                area.append(0)
                area_seg_speed_lim.append(area_seg)
                area_speed_lim.append(area)
                break

    return area_seg_speed_lim, area_speed_lim


def clip_to_n_station(seg, info, station_loc):
    n_station_seg = []
    n_station_info = []
    station_P = 1
    shift_i = 0
    shift_loc = 0
    for i in range(len(seg)-1):
        if seg[i] < station_loc[station_P] <= seg[i+1]:
            n_station_seg.append(np.array(seg[shift_i: i+1])-shift_loc)
            n_station_info.append(np.array(info[shift_i: i+1]))
            if station_P != 1:
                n_station_seg[station_P-1] = np.insert(n_station_seg[station_P-1], 0, 0)
                n_station_info[station_P-1] = np.insert(n_station_info[station_P-1], 0, n_station_info[station_P-2][-2])

            n_station_seg[station_P-1] = np.append(n_station_seg[station_P-1], station_loc[station_P]-shift_loc)
            n_station_info[station_P-1] = np.append(n_station_info[station_P-1], 0)
            shift_i = i+1
            shift_loc = station_loc[station_P]
            station_P += 1
    return n_station_seg, n_station_info



def get_mri(speed_lim_seg, speed_lim, slope_seg, slope, train_info):

    # 初始化 v[0]为起点开始的上升速度，v[1]为终点开始的减速速度
    # v_frontier 是达到限速的位置或限速点
    # v_lim 是每米的限速

    line_len = speed_lim_seg[-1]-speed_lim_seg[0]

    v = np.array([np.zeros(line_len + 1), np.zeros(line_len + 1)])
    v_frontier = np.array([speed_lim_seg, speed_lim_seg])
    v_lim = get_now_info(speed_lim_seg, speed_lim, line_len)
    max_traction = train_info['traction_140']
    power = train_info['power']
    mass_factor = train_info['mass_factor']
    mass = train_info['mass']
    co_a = train_info['drag_coefficient_a']
    co_b = train_info['drag_coefficient_b']
    co_c = train_info['drag_coefficient_c']
    max_braking = train_info['braking_140']
    train_len = train_info['len']
    seg = 1
    loc = 0
    while True:
        loc += 1
        if v[0][loc - 1] == 0:
            max_gear_action = max_traction
        else:
            max_gear_action = min(max_traction, power / (
                    (1+mass_factor) * mass * v[0][loc - 1]))

        gear_acc = max_gear_action
        slope_acc = get_slope_accelerated(loc, slope_seg, slope, train_info)
        last_v = v[0][loc - 1]
        acc = get_accelerated(last_v, gear_acc, slope_acc, loc, train_info)
        now_v = min(v_lim[loc] / 3.6, math.sqrt(last_v ** 2 + 2 * acc))
        v[0][loc] = now_v
        if now_v == v_lim[loc] / 3.6:
            for i in range(len(speed_lim_seg)):
                if speed_lim_seg[i] < loc <= speed_lim_seg[i + 1]:
                    v_frontier[0][seg] = loc
                    loc = speed_lim_seg[i + 1]
                    if loc != speed_lim_seg[-2]:
                        v[0][loc] = min(now_v, v_lim[loc + 1] / 3.6)
                    seg += 1
                    break
        if loc == speed_lim_seg[seg]:
            seg += 1
        if loc == speed_lim_seg[-2]:
            break

    seg = -2
    loc = line_len + 1
    while True:
        loc -= 1
        gear_acc = max_braking
        slope_acc = get_slope_accelerated(loc - 1, slope_seg, slope, train_info)
        # now_v**2-last_v**2 = 2(gear_acc-slope_acc-basic_acc)train_step
        # basic_acc = a + b*last_v*3.6 + c*(last_v*3.6)**2
        now_v = v[1][loc]
        if now_v != v_lim[loc - 1] / 3.6:
            a = 1 - 2 * co_c * 3.6 ** 2 * (9.81 / 1000)
            b = -2 * co_b * 3.6 * (9.81 / 1000)
            c = -now_v ** 2 + 2 * (gear_acc - slope_acc - co_a * (9.81 / 1000))
            last_v = min(v_lim[loc - 1] / 3.6, (-b + math.sqrt(b ** 2 - 4 * a * c)) / (2 * a))
        else:
            last_v = v_lim[loc - 1] / 3.6
        v[1][loc - 1] = last_v
        if last_v == v_lim[loc] / 3.6:
            for i in range(len(speed_lim_seg)):
                if speed_lim_seg[i] < loc <= speed_lim_seg[i + 1]:
                    v_frontier[1][seg] = loc
                    loc = speed_lim_seg[i] + 1
                    if loc != speed_lim_seg[1] + 1:
                        v[1][loc - 1] = last_v
                    seg -= 1
                    break
        if loc == speed_lim_seg[seg]:
            seg -= 1
        if loc == speed_lim_seg[1] + 1:
            break

    for i in range(1, len(speed_lim_seg) - 1):
        if v_frontier[0][i] >= v_frontier[1][i]:
            for j in range(speed_lim_seg[i], speed_lim_seg[i + 1]):
                if v[0][j] <= v[1][j]:
                    v[0][j] = v[1][j] = min(v[0][j], v[1][j])
                    v[0][j + 1:speed_lim_seg[i + 1] - 1] = 0
                    v[1][speed_lim_seg[i] + 1:j - 1] = 0

    mri = []
    for l, r, m in zip(v[0], v[1], v_lim):
        if l == r != 0:
            mri.append(min(l, r))
            continue
        if l == 0 and r == 0:
            mri.append(m / 3.6)
        elif l == 0:
            mri.append(r)
        else:
            mri.append(l)

    i = 0
    seg_t = np.zeros(len(speed_lim_seg) - 1)
    for loc in range(len(mri)):
        loc += 1
        seg_t[i] += 2 / (mri[loc - 1] + mri[loc])
        if loc == speed_lim_seg[i + 1]:
            i += 1
        if mri[loc] == 0:
            break

    return np.array(mri), sum(seg_t), seg_t, v_frontier

'''
def get_mri1(speed_lim_seg, speed_lim, slope_seg, slope, train_info):

    v = np.array([np.zeros(pram.line_len + 1), np.zeros(pram.line_len + 1)])
    v_frontier = np.array([pram.speed_lim_seg, pram.speed_lim_seg])
    v_lim = get_now_info(pram.speed_lim_seg, pram.speed_lim)

    seg = 1
    loc = 0
    while True:
        loc += 1
        if v[0][loc - 1] == 0:
            max_gear_action = pram.max_traction
        else:
            max_gear_action = min(pram.max_traction,
                                  pram.train_power / (pram.train_mass_factor * pram.train_weight * v[0][loc - 1]))

        gear_acc = max_gear_action
        slope_acc = get_slope_accelerated(loc)
        last_v = v[0][loc - 1]
        acc = get_accelerated(last_v, gear_acc, slope_acc, loc)
        now_v = min(v_lim[loc] / 3.6, math.sqrt(last_v ** 2 + 2 * acc * pram.train_step))
        v[0][loc] = now_v
        if now_v == v_lim[loc] / 3.6:
            for i in range(len(pram.speed_lim_seg)):
                if pram.speed_lim_seg[i] < loc <= pram.speed_lim_seg[i + 1]:
                    v_frontier[0][seg] = loc
                    loc = pram.speed_lim_seg[i + 1]
                    if loc != pram.speed_lim_seg[-2]:
                        v[0][loc] = min(now_v, v_lim[loc + 1] / 3.6)
                    seg += 1
                    break
        if loc == pram.speed_lim_seg[seg]:
            seg += 1
        if loc == pram.speed_lim_seg[-2]:
            break

    seg = -2
    loc = pram.line_len + 1
    while True:
        loc -= 1
        gear_acc = pram.max_braking
        slope_acc = get_slope_accelerated(loc - 1)
        # now_v**2-last_v**2 = 2(gear_acc-slope_acc-basic_acc)train_step
        # basic_acc = a + b*last_v*3.6 + c*(last_v*3.6)**2
        now_v = v[1][loc]
        if now_v != v_lim[loc - 1] / 3.6:
            a = 1 - 2 * pram.train_step * pram.drag_coefficient_c * 3.6 ** 2 * (9.81 / 1000)
            b = -2 * pram.train_step * pram.drag_coefficient_b * 3.6 * (9.81 / 1000)
            c = -now_v ** 2 + 2 * pram.train_step * (gear_acc - slope_acc - pram.drag_coefficient_a * (9.81 / 1000))
            last_v = min(v_lim[loc - 1] / 3.6, (-b + math.sqrt(b ** 2 - 4 * a * c)) / (2 * a))
        else:
            last_v = v_lim[loc - 1] / 3.6
        v[1][loc - 1] = last_v
        if last_v == v_lim[loc] / 3.6:
            for i in range(len(pram.speed_lim_seg)):
                if pram.speed_lim_seg[i] < loc <= pram.speed_lim_seg[i + 1]:
                    v_frontier[1][seg] = loc
                    loc = pram.speed_lim_seg[i] + 1
                    if loc != pram.speed_lim_seg[1] + 1:
                        v[1][loc - 1] = last_v
                    if last_v > pram.speed_lim[i - 1] / 3.6:
                        v[1][loc - 1] = pram.speed_lim[i - 1] / 3.6
                    seg -= 1
                    break
        if loc == pram.speed_lim_seg[seg]:
            seg -= 1
        if loc == pram.speed_lim_seg[1] + 1:
            break

    for i in range(1, len(pram.speed_lim_seg) - 1):
        if v_frontier[0][i + 1] >= v_frontier[1][i]:
            for j in range(pram.speed_lim_seg[i] + 1, pram.speed_lim_seg[i + 1]):
                if v[0][j] > v[1][j] != 0:
                    nnn = j
                    v[0][j] = v[1][j] = min(v[0][j], v[1][j])
                    v[0][j:pram.speed_lim_seg[i + 1] + 1] = 0
                    v[1][pram.speed_lim_seg[i] + 1:j] = 0


    mri = []
    for l, r, m in zip(v[0], v[1], v_lim):
        if l == r != 0:
            mri.append(min(l, r))
            continue
        if l == 0 and r == 0:
            mri.append(m / 3.6)
        elif l == 0:
            mri.append(r)
        else:
            mri.append(l)

    # TODO：
    mri = np.maximum(v[0], v[1])
    for i in range(len(mri)):

        if mri[i] == 0:
            mri[i] = v_lim[i] / 3.6

    i = 0
    seg_t = np.zeros(len(pram.speed_lim_seg) - 1)
    for loc in range(len(mri)):
        loc += 1
        seg_t[i] += (2 * pram.train_step) / (mri[loc - 1] + mri[loc])
        if loc == pram.speed_lim_seg[i + 1]:
            i += 1
        if mri[loc] == 0:
            break

    return np.array(mri), sum(seg_t), seg_t, v_frontier
    '''


def get_psi_part_a(mri, v_frontier, plan_time):
    """
       根据MRI数据和其他参数计算psi值和最大速率比率
       参数:
       psi_a (float): 原始psi值
        mri (float): mri值
        v_frontier (float): 牵引-巡航边界点
        sat_seg_t (float): sat段时间
        rate (float): 转换速度的比率
       返回值:
       psi: 一维数组，表示计算得到的psi值
       max_rate: 最大速率比率
    """

    # 初始化
    psi = np.array(mri)
    is_done = False

    # 折半搜索
    low = 0
    high = v_frontier[0][2]

    while True:
        l_cr = (low + high) // 2
        v_cr = np.ones(len(psi)) * psi[l_cr]
        temp_psi = np.minimum(v_cr, psi)
        temp_time = get_line_time(temp_psi)
        if temp_time < plan_time:
            high = l_cr - 1
        else:
            low = l_cr + 1

        if low >= high:
            psi = copy.deepcopy(temp_psi)
            break

        '''
        time_start_loc = pram.speed_lim_seg[seg]
        time_start_seg = seg
        temp_psi = copy.deepcopy(np.array(psi))
        join_a = (low + high) // 2
        # 先找到join_b点，连接这两点
        for i in range(seg+1, len(pram.speed_lim_seg)):
            if temp_psi[join_a] >= temp_psi[pram.speed_lim_seg[i]]:
                seg = i-1

        for join in range(v_frontier[1][seg], pram.speed_lim_seg[seg+1]):
            if temp_psi[join] >= temp_psi[join_a] >= temp_psi[join + 1]:
                # b连接点
                join_b = join
                # 让b点的速度降低到a点 并且匀速[a,b-1]
                temp_psi[join_a:join_b + 1] = temp_psi[join_a]
                break

        # 计算这段时间
        t = 0
        for loc in range(time_start_loc, pram.speed_lim_seg[seg+1]+1):
            if temp_psi[loc - 1] + temp_psi[loc] == 0:
                raise Exception("v0+v1==0")
            t += (2 * pram.train_step) / (temp_psi[loc - 1] + temp_psi[loc])
        # 查找
        if t > np.sum(sat_seg_t[time_start_seg:seg+1]):
            low = join_a + 1
        else:
            high = join_a - 1
        if low >= high:
            psi = copy.deepcopy(temp_psi)
            if seg == len(pram.speed_lim_seg)-2:
                is_done = True
            break
        seg = time_start_seg
        '''

    max_rate = np.max(np.array(mri)) / np.max(psi)
    return psi, max_rate, psi[l_cr]


def get_psi_part_b(psi, mri, v_frontier, sat_seg_t, rate, line_len, speed_lim_seg, slope_seg, slope, train_info, last_cov=0, last_rate=0):

    psi_rate = np.zeros(line_len + 1)
    for i, (psi_value, mri_value) in enumerate(zip(psi, mri)):
        if psi_value * rate >= mri_value:
            psi_rate[i] = mri_value
        else:
            psi_rate[i] = psi_value * rate
    if last_cov == 0:
        low = v_frontier[0][2]
        high = line_len
    else:
        if abs(last_rate-rate) <= 0.2:
            low = max(int(last_cov-line_len*0.1), v_frontier[0][2])
            high = min(int(last_cov+line_len*0.1), line_len)
        else:
            low = v_frontier[0][2]
            high = line_len
    convert = [0, 0]
    while True:
        coasting_loc = (low + high) // 2
        temp_psi = copy.deepcopy(psi_rate)
        is_renew = False
        for loc in range(coasting_loc, line_len):
            loc += 1

            gear_action = 0
            slope_acc = get_slope_accelerated(loc, slope_seg, slope, train_info)
            last_v = temp_psi[loc - 1]
            acc = get_accelerated(last_v, gear_action, slope_acc, loc, train_info)
            if last_v ** 2 + 2 * acc < 0:
                now_v = 0
            else:
                now_v = math.sqrt(last_v ** 2 + 2 * acc)
            if now_v == 0 and loc != line_len:
                is_renew = True
                break
            temp_psi[loc] = now_v
            if temp_psi[loc] >= psi_rate[loc]:
                temp_psi[loc:] = psi_rate[loc:]
                if loc > speed_lim_seg[-2]:
                    break
                '''
                else:
                    loc = speed_lim_seg[-2]
                    continue
                '''
        if is_renew:
            low = coasting_loc + 1
            continue
        t = 0
        for loc in range(speed_lim_seg[1], line_len + 1):
            if temp_psi[loc - 1] + temp_psi[loc] == 0:
                raise Exception("v0+v1==0")
            t += 2 / (temp_psi[loc - 1] + temp_psi[loc])
        convert = [coasting_loc, temp_psi[coasting_loc]]
        if convert[0] == line_len-1:
            convert = [0, 0]
        if t > np.sum(sat_seg_t[1:]):
            low = coasting_loc + 1
        else:
            high = coasting_loc - 1
        if low >= high and line_len <= 40000:
            psi_rate = copy.deepcopy(temp_psi)
            break
        if low + 10 >= high and line_len > 40000:
            psi_rate = copy.deepcopy(temp_psi)
            break
    return psi_rate, convert


def get_line_step_time(line):
    ts = [0]
    t = 0
    for loc in range(1, len(line + 1)):
        if line[loc - 1] + line[loc] == 0:
            raise Exception("v0+v1==0")
        t += 2 / (line[loc - 1] + line[loc])
        ts.append(t)
    return ts


def get_line_step_energy(line, slope_seg, slope, train_info):
    es = [0]
    e = 0
    max_traction = train_info['traction_140']
    max_braking = train_info['braking_140']
    mass = train_info['mass']
    for loc in range(len(line) - 1):
        loc += 1
        out_acc = (line[loc] ** 2 - line[loc - 1] ** 2) / 2
        slope_acc = get_slope_accelerated(loc, slope_seg, slope, train_info)
        last_v = line[loc - 1]
        non_gear_acc = get_accelerated(last_v, 0, slope_acc, loc, train_info)
        gear_acc = max(min(out_acc - non_gear_acc, max_traction), max_braking)
        if gear_acc > 0:
            # kwh
            e += (gear_acc * mass) / 3600
        es.append(e)

    return es


def get_line_time(line):

    t = 0
    for loc in range(1, len(line+1)):
        if line[loc - 1] + line[loc] == 0:
            raise Exception("v0+v1==0")
        t += 2 / (line[loc - 1] + line[loc])
    return t


def get_line_cr_energy(line, slope_seg, slope, train_info):

    max_traction = train_info['traction_140']
    max_braking = train_info['braking_140']
    mass = train_info['mass']
    e = 0
    for loc in range(len(line)-1):
        loc += 1
        out_acc = (line[loc]**2-line[loc-1]**2)/2
        slope_acc = get_slope_accelerated(loc, slope_seg, slope, train_info)
        last_v = line[loc - 1]
        v = line[loc]
        non_gear_acc = get_accelerated(last_v, 0, slope_acc, loc, train_info)
        gear_acc = max(min(out_acc - non_gear_acc, max_traction), max_braking)
        if gear_acc > 0 and last_v == v:
            # kwh
            e += (gear_acc * mass) / 3600

    return e


def get_line_energy(line, slope_seg, slope, train_info):

    max_traction = train_info['traction_140']
    max_braking = train_info['braking_140']
    mass = train_info['mass']
    e = 0
    for loc in range(len(line)-1):
        loc += 1
        out_acc = (line[loc]**2-line[loc-1]**2)/2
        slope_acc = get_slope_accelerated(loc, slope_seg, slope, train_info)
        last_v = line[loc - 1]
        non_gear_acc = get_accelerated(last_v, 0, slope_acc, loc, train_info)
        gear_acc = max(min(out_acc - non_gear_acc, max_traction), max_braking)
        if gear_acc > 0:
            # kwh
            e += (gear_acc * mass) / 3600

    return e


def get_line_re_energy(line, slope_seg, slope, train_info):

    max_traction = train_info['traction_140']
    max_braking = train_info['braking_140']
    mass = train_info['mass']
    e = 0
    for loc in range(len(line)-1):
        loc += 1
        out_acc = (line[loc]**2-line[loc-1]**2)/2
        slope_acc = get_slope_accelerated(loc, slope_seg, slope, train_info)
        last_v = line[loc - 1]
        non_gear_acc = get_accelerated(last_v, 0, slope_acc, loc, train_info)
        gear_acc = max(min(out_acc - non_gear_acc, max_traction), max_braking)
        if gear_acc < 0:
            # kwh
            e += (abs(gear_acc) * mass) / 3600

    return e


def get_line_action(line, slope_seg, slope, train_info):

    max_traction = train_info['traction_140']
    train_power = train_info['power']
    mass = train_info['mass']
    max_braking = train_info['braking_140']

    actions = [0]
    for loc in range(len(line) - 1):

        loc += 1
        out_acc = (line[loc] ** 2 - line[loc - 1] ** 2) / 2

        slope_acc = get_slope_accelerated(loc, slope_seg, slope, train_info)  # 获取坡度加速度

        last_v = line[loc - 1]

        non_gear_acc = get_accelerated(last_v, 0, slope_acc, loc, train_info)

        if line[loc - 1] == 0:
            max_gear_action = max_traction
        else:
            max_gear_action = min(max_traction, train_power / (mass * line[loc - 1]))

        gear_acc = max(min(out_acc - non_gear_acc, max_gear_action), max_braking)
        if gear_acc >= 0:
            action = gear_acc/max_gear_action
        else:
            action = gear_acc / max_braking
        actions.append(action)
    return actions






def get_psi_min_energy(psi_a, mri, v_frontier, sat_seg_t, max_rate, plan_time, slope_seg, slope, train_info, line_len, speed_lim_seg):

    rates = np.linspace(1, max_rate, 100)
    start = 0
    end = len(rates) - 1
    best_energy = get_line_energy(psi_a, slope_seg, slope, train_info)
    best_energy_rate = rates[0]

    psi_b, cov = get_psi_part_b(psi_a, mri, v_frontier, sat_seg_t, rates[(start + end) // 2], line_len, speed_lim_seg, slope_seg, slope, train_info)
    energy = get_line_energy(psi_b, slope_seg, slope, train_info)




    while start < end:
        mid = (start + end) // 2
        time = math.fabs(plan_time - get_line_time(psi_b))

        psi_left, cov_left = get_psi_part_b(psi_a, mri, v_frontier, sat_seg_t, rates[(start + mid) // 2], line_len,
                                            speed_lim_seg, slope_seg, slope, train_info)
        energy_left = get_line_energy(psi_left, slope_seg, slope, train_info)
        time_left = math.fabs(plan_time - get_line_time(psi_left))




        psi_right, cov_right = get_psi_part_b(psi_a, mri, v_frontier, sat_seg_t, rates[(end + mid) // 2], line_len,
                                              speed_lim_seg, slope_seg, slope, train_info)
        energy_right = get_line_energy(psi_right, slope_seg, slope, train_info)
        time_right = math.fabs(plan_time - get_line_time(psi_right))




        if energy <= best_energy:
            best_energy = energy
            best_energy_rate = rates[mid]

        if abs(energy_right-energy_left) <= 0.5:
            break

        if max(time, time_left, time_right) >= 1 or energy_right >= energy_left:
            end = mid
            psi_b = psi_left
            energy = energy_left
            continue
        if energy_left > energy_right:
            start = mid
            psi_b = psi_right
            energy = energy_right


    return best_energy, best_energy_rate


def get_on_time_max_rate(psi_a, mri, v_frontier, sat_seg_t, max_rate, plan_time, line_len, speed_lim_seg, slope_seg, slope, train_info):

    rates = np.linspace(1, max_rate, 100)
    left = 0
    right = len(rates) - 1

    while left < right:
        mid = (left + right) // 2
        psi_b, cnv = get_psi_part_b(psi_a, mri, v_frontier, sat_seg_t, rates[mid], line_len, speed_lim_seg, slope_seg, slope, train_info)
        time_error = math.fabs(plan_time - get_line_time(psi_b))
        if time_error >= 1:
            right = mid
        else:
            left = mid + 1

    boundary = left
    return rates[boundary]



def get_bessel_curve(psi_a, mri, v_frontier, sat_seg_t, max_rate, energy_saving_rate, line_len, speed_lim_seg, slope_seg, slope, train_info, precision=100):

    rates = [1, (1+energy_saving_rate)/2, energy_saving_rate, (energy_saving_rate+max_rate)/2, max_rate]

    x = []
    y = []
    max_psi_b = psi_a
    min_psi_b = psi_a

    for rate in rates:
        psi_b, cnv = get_psi_part_b(psi_a, mri, v_frontier, sat_seg_t, rate, line_len, speed_lim_seg, slope_seg, slope, train_info)

        max_psi_b = np.maximum(max_psi_b, psi_b)
        min_psi_b = np.minimum(min_psi_b, psi_b)
        x.append(cnv[0])
        y.append(cnv[1])
    x = np.array(x)
    y = np.array(y)

    i = 0
    while True:
        if x[i] == x[i + 1] or x[i] == 0:
            x = np.delete(x, i)
            y = np.delete(y, i)
            i -= 1
        i += 1
        if i == len(x) - 1:
            break

    try:
        f = interp1d(x, y, kind='cubic')
    except:
        f = interp1d(x, y, kind='linear')
    x = np.arange(np.min(x), np.max(x), 1)
    y = f(x)
    bessel = np.zeros(line_len + 1)
    bessel[x] = y

    return np.maximum(max_psi_b, bessel), min_psi_b


def get_actions(psi_a, mri, v_frontier, sat_seg_t, max_rate, line_len, speed_lim_seg, slope_seg, slope, train_info, precision=100):
    rates = np.linspace(1, max_rate, precision)
    actions = []
    for rate in rates:
        psi_b, _ = get_psi_part_b(psi_a, mri, v_frontier, sat_seg_t, rate, line_len, speed_lim_seg, slope_seg, slope, train_info)
        actions.append(get_line_action(psi_b, slope_seg, slope, train_info))
    actions = np.array(actions)
    return actions




def anticipate_energy_target(mri, psi, v_cr, max_v_lim, seg_slope, slope, train_info):
    basic_acc = get_basic_acc(0, v_cr, train_info)
    mt_acc = min(train_info['traction_140'], train_info['power'] / (
            (1 + train_info['mass_factor']) * train_info['mass'] * v_cr))
    cr_acc = basic_acc
    mt_energy = get_energy(mt_acc, train_info)
    cr_energy = get_energy(cr_acc, train_info)
    fast_cr_energy = get_line_cr_energy(mri, seg_slope, slope, train_info)
    min_cr_energy = get_line_cr_energy(psi, seg_slope, slope, train_info)

    v_lim_negative = 2*(1/(1+np.exp(1)**(-0.1*(max_v_lim - v_cr*3.6)))-1)
    traction_cost_positive = 1-(cr_energy/mt_energy)
    cruise_mileage_positive = min_cr_energy/fast_cr_energy

    anticipate_energy = min_cr_energy*0.1*(v_lim_negative+(traction_cost_positive+cruise_mileage_positive)/2)



    return anticipate_energy


def allocation_surplus_time(min_time, seg_times, speed_lim_seg, plan_time):
    point = speed_lim_seg
    # plan time
    p_time = plan_time
    # segment length
    seg_lens = get_length(point)
    # total supplement time
    ts_time = p_time - min_time
    if ts_time < 0:
        raise Exception(f"p_t{p_time:.2f}<min_t{min_time:.2f}", )

    while True:
        avg_vs = get_avg_v(seg_lens, seg_times)
        seg, v_sort = get_v_sort(avg_vs)
        block = get_block(seg, v_sort)
        block_sum_len = get_block_sum_len(block, seg_lens)
        block_avg_v = get_block_avg_v(block, avg_vs)
        block_len = get_block_len(block, seg_lens)

        delta_rs_time = 1 / block_avg_v[1] - 1 / block_avg_v[0]
        rs_time = delta_rs_time * float(block_sum_len[0])
        if ts_time < rs_time:
            rs_time = ts_time
        ts_time -= rs_time

        seg_times = allocate_runtime(seg_times, block, block_len, block_sum_len, rs_time)
        avg_vs = get_avg_v(seg_lens, seg_times)
        if ts_time == 0:
            break

    return seg_times, avg_vs


def planing_speed_interval(mri, sat_seg_t, v_frontier, plan_time, slope_seg, slope, train_info, line_len, speed_lim_seg, v_lim=310):
    psi_a, max_rate, v_cr = get_psi_part_a(mri, v_frontier, plan_time)
    min_energy, min_energy_rate = get_psi_min_energy(psi_a, mri, v_frontier, sat_seg_t, max_rate, plan_time, slope_seg, slope, train_info, line_len, speed_lim_seg)

    min_energy_psi, _ = get_psi_part_b(psi_a, mri, v_frontier, sat_seg_t, min_energy_rate, line_len, speed_lim_seg, slope_seg, slope, train_info)
    aet_energy = anticipate_energy_target(mri, min_energy_psi, v_cr, v_lim, slope_seg, slope, train_info)
    min_energy_action = get_line_action(min_energy_psi, slope_seg, slope, train_info)
    min_energy_times = get_line_step_time(min_energy_psi)
    min_energy_step = get_line_step_energy(min_energy_psi, slope_seg, slope, train_info)
    on_time_max_mate = get_on_time_max_rate(psi_a, mri, v_frontier, sat_seg_t, max_rate, plan_time, line_len, speed_lim_seg, slope_seg, slope, train_info)
    max_rate = min(max(min_energy_rate+(min_energy_rate-1), min_energy_rate*1.1), max_rate)

    upper_bound_psi, lower_bound_psi = get_bessel_curve(psi_a, mri, v_frontier, sat_seg_t, max_rate, min_energy_rate, line_len, speed_lim_seg, slope_seg, slope, train_info)

    return min_energy_psi, upper_bound_psi, lower_bound_psi, min_energy, aet_energy, min_energy_action, min_energy_times, min_energy_step



