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
File: 206-move_cartesian_waypoints.py
Brief:  Example code for Astribot Robotics.
        Follow sequence of joints' waypoints or cartesian waypoints.
"""

import sys
import rospy
from core.sdk_client.astribot_client import Astribot

if __name__ == '__main__':
    # Connect astribot
    astribot = Astribot()
    astribot.move_to_home()

    names = [astribot.torso_name, astribot.arm_names[1], astribot.effector_names[1]]
    waypoints = list()
    time_list = list()
    waypoints.append([[ 0.0, 0.0, 1.1, 0.0, 0.0, 0.0, 1.0],
                        [ 0.4, -0.222, 0.75, 0.0, 0.0, 0.707, 0.707],
                        [ 0.0]])
    time_list.append(3.0)
    waypoints.append([[ 0.0, 0.0, 1.1, 0.0, 0.0, 0.0, 1.0],
                        [ 0.4, -0.222, 0.75, 0.0, 0.0, 0.707, 0.707],
                        [ 100.0]])
    time_list.append(4.0)
    waypoints.append([[ 0.0, 0.0, 1.25, 0.0, 0.0, 0.0, 1.0],
                        [ 0.4, -0.222, 0.99, 0.0, 0.0, 0.707, 0.707],
                        [ 100.0]])
    time_list.append(6.0)
    waypoints.append([[ 0.0, 0.0, 1.25, 0.0, 0.0, 0.0, 1.0],
                        [ 0.285, -0.222, 0.99, 0.0, 0.0, 0.707, 0.707],
                        [ 0.0]])
    time_list.append(8.0)

    response = astribot.move_cartesian_waypoints(names, waypoints, time_list, use_wbc=True)
    print("Response from move_cartesian_waypoints:", response)
