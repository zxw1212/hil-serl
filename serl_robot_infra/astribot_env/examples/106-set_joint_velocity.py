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
File: 106-set_joint_velocity.py
Brief:  Example code for Astribot Robotics.
        Move and set joint velocity of Astribot.
"""

import rospy
from core.sdk_client.astribot_client import Astribot

if __name__ == '__main__':
    # Connect astribot
    astribot = Astribot()
    rate = rospy.Rate(30.0)
    
    while not rospy.is_shutdown():
        position_state = astribot.get_current_joints_position([astribot.arm_left_name])
        if position_state[0][3] < 1.0:
            vel_list = [[0.0, 0.0, 0.0, 0.1, 0.0, 0.0]]
        else:
            vel_list = [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]
        astribot.set_joints_velocity([astribot.arm_left_name], vel_list)
        rate.sleep()
