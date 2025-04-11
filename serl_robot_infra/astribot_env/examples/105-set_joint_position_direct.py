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
File: 105-set_joint_position_direct.py
Brief:  Example code for Astribot Robotics.
        Move and set joint positions of Astribot.
"""

import rospy
from core.sdk_client.astribot_client import Astribot

if __name__ == '__main__':
    # Connect astribot
    astribot = Astribot()
    astribot.move_to_home()

    names = [astribot.torso_name, astribot.arm_names[0], astribot.arm_names[1]]
    command_list = astribot.get_desired_joints_position(names=names)
    vel_cmd = [[ 0.21, -0.42, 0.21, 0.0],
                [ 0.15, -0.15, 0.00, 0.15, 0.0, 0.0, 0.0],
                [-0.15, -0.15, 0.00, 0.15, 0.0, 0.0, 0.0]]
    freq = 30.0
    dt = 1.0 / freq
    rate = rospy.Rate(freq)
    while not rospy.is_shutdown():
        if command_list[0][0] < 0.7:
            for i in range(len(command_list)):
                for j in range(len(command_list[i])):
                    command_list[i][j] += vel_cmd[i][j] * dt
        astribot.set_joints_position(names, command_list, control_way="direct", use_wbc=False, freq=freq)
        rate.sleep()
