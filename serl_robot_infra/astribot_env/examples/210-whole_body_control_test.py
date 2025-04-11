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
File: 210-whole_body_control_test.py
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
    command_list = astribot.get_desired_cartesian_pose(names=names)

    #TODO(@jinbo): 改到500hz，加print,表示wbc模式已经开启，用户可以尝试推动腰部，手臂末端会保持不动
    rate = rospy.Rate(250.0)
    while not rospy.is_shutdown():
        astribot.set_cartesian_pose(names, command_list, control_way="direct", use_wbc=True)
        rate.sleep()
