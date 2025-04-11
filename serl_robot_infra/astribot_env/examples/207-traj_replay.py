#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Copyright (c) 2024, Astribot Co., Ltd.
# All rights reserved.
# License: BSD 3-Clause License
# -----------------------------------------------------------------------------
# Author: Astribot Team
# -----------------------------------------------------------------------------

"""
File: 207-traj_replay.py
Brief:  Example code for Astribot Robotics.
        Replay a trajectory, using joy to control the forward and reverse playback of trajectory.
"""

import sys
import h5py
from core.sdk_client.astribot_client import Astribot

if __name__ == '__main__':
    # Connect astribot
    if len(sys.argv) < 2:
        raise ValueError("Please provide the path to the hdf5 file.")
    path_to_hdf5 = sys.argv[1]
    astribot = Astribot()
    astribot.move_to_home()

    with h5py.File(path_to_hdf5, 'r') as root:
        joints_action_obs = root['joints_dict/joints_position_command'][()].tolist()
        arm_left_cartesian_action_obs = root['poses_dict/astribot_arm_left'][()].tolist()
        arm_right_cartesian_action_obs = root['poses_dict/astribot_arm_right'][()].tolist()
        time_obs = root['time'][()].tolist()

    waypoints, waypoint, time_list = list(), list(), list()
    for index, action in enumerate(joints_action_obs):
        if index == 0:
            time_list.append(2.0) #TODO(@11): 应自动判断duration，不可写死
        else:
            time_list.append(time_list[index-1] + time_obs[index] - time_obs[index-1])
        start_idx = 0
        for dof in astribot.whole_body_dofs:
            waypoint.append(action[start_idx:start_idx+dof])
            start_idx += dof
        waypoints.append(waypoint)

    astribot.move_joints_waypoints(astribot.whole_body_names, waypoints, time_list, use_wbc=False, joy_controller=False)
