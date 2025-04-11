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
File: 205-move_joints_waypoints.py
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

    names = [astribot.torso_name, astribot.arm_names[0], astribot.effector_names[0]]
    waypoints = list()
    time_list = list()
    waypoints.append([[ 0.7, -1.4,  0.7, 0.0],
                      [-0.57, 0.0, -1.5709, 1.0, 0.0, 0, 0],
                      [ 0.0]])
    time_list.append(3.0)
    waypoints.append([[ 0.7, -1.4,  0.7, 0.0],
                      [-0.57, 0.0, -1.5709, 1.0, 0.0, 0, 0],
                      [ 100.0]])
    time_list.append(4.0)
    waypoints.append([[ 0.2, -0.4,  0.2, 0.0],
                      [-0.57, 0.0, -1.5709, 1.0, 0.0, 0, 0],
                      [ 100.0]])
    time_list.append(6.0)
    waypoints.append([[ 0.2, -0.4,  0.2, 0.0],
                      [ 0.0, 0.0, -1.5709, 1.5709, 0.0, 0, 0],
                      [ 0.0]])
    time_list.append(8.0)

    response = astribot.move_joints_waypoints(names, waypoints, time_list, use_wbc=True)

    astribot.move_to_home()
    print("Response from move_joint_waypoints:", response)
