"""
slope = [-3, 27.83, -12.439, 0, 0]
slope_seg = [0, 400, 641, 971, 1056]
speed_lim = [50.0, 75.0, 50.0, 0]
speed_lim_seg = [0, 120, 920, 1056]
93

slope = [0, 3, -8, 3, 10, 0, 0]
slope_seg = [0, 152, 620, 1231, 1909, 3510, 4611]
speed_lim = [50.0, 75.0, 65, 50, 0]
speed_lim_seg = [0, 100, 1800, 3300, 4611]
330

slope = [0, 3, -8, 3, 10, 0, 0]
slope_seg = [0, 1520, 6200, 12310, 19090, 35100, 46110]
speed_lim = [80.0, 300.0, 250, 300, 250, 0]
speed_lim_seg = [0, 1000, 18000, 19500, 33000, 46110]
900

slope = [0, 3, -8, 3, 10, 0, 0]
slope_seg = [0, 152, 620, 1231, 1909, 3510, 4611]
speed_lim = [20.0, 75.0, 62.5, 75, 70, 0]
speed_lim_seg = [0, 100, 1800, 1950, 3300, 4611]
400
"""
import numpy as np

slope = [0, 3, -8, 3, 10, 0, 0]
slope_seg = [0, 1520, 6200, 12310, 19090, 35100, 46110]
speed_lim = [80.0, 310.0, 280, 310, 280, 80, 0]
speed_lim_seg = [0, 1000, 22000, 23500, 36500, 45110, 46110]
line_len = slope_seg[-1]

'''
slope = [0, 3, -8, 3, 10, 0, 0]
slope_seg = [0, 1520, 6200, 12310, 19090, 35100, 46110]
speed_lim = [80.0, 310.0, 280, 310, 280, 80, 0]
speed_lim_seg = [0, 1000, 18000, 19500, 33000, 45110, 46110]
line_len = slope_seg[-1]

plan_time = 900
train_type = 6
train_section_len = 20
train_len = 120
train_weight = 180
train_max_traction = 144
max_action = 0.8
min_action = -0.8
drag_coefficient_a = 2.7551
drag_coefficient_b = 0.014
drag_coefficient_c = 0.00075
'''
plan_time = 1100
train_type = 8
train_section_len = 25.375
train_len = 203
train_power = 8760
# train_weight = 490
train_weight = 490
train_mass_factor = 1.03
train_max_traction = 310.72
max_action = 1
min_action = 0
max_braking = -1
max_traction = 0.5

drag_coefficient_a = 1.954
drag_coefficient_b = 0.00622
drag_coefficient_c = 0.0004954134
'''
drag_coefficient_a = 5.6
drag_coefficient_b = 0.036
drag_coefficient_c = 0.000121
'''
reward_c = 5
reward_x = 1
train_step = 1
step = 50
wave = 0.05
fsb_scale = 0.6
max_speed = 75
screen_width = 700
screen_height = 650
x_zoom = screen_width / (line_len+1)
y_zoom = screen_height / (max(speed_lim))

