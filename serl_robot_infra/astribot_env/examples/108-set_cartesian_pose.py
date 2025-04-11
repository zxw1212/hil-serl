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
File: 108-set_cartesian_pose.py
Brief:  Example code for Astribot Robotics.
        Move and Set Cartesian Pose of Astribot.
"""

import rospy
from core.sdk_client.astribot_client import Astribot

if __name__ == '__main__':
    # Connect astribot
    astribot = Astribot()
    astribot.move_to_home()

    names = [astribot.torso_name, astribot.arm_names[0], astribot.arm_names[1]]
    command_list = astribot.get_desired_cartesian_pose(names=names)

    rate = rospy.Rate(30.0)
    while not rospy.is_shutdown():
        if command_list[1][1] < 0.45:
            command_list[1][1] += 0.004
        if command_list[2][1] > -0.45:
            command_list[2][1] -= 0.004
        astribot.set_cartesian_pose(names, command_list, control_way="filter", use_wbc=True)
        rate.sleep()

