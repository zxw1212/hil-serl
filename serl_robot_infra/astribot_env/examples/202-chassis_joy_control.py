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
File: 202-chassis_joy_control.py
Brief:  Example code for Astribot Robotics.
        Control Astribot chassis with joy.
"""

import rospy
from tools.joy_tools import XboxController
from core.sdk_client.astribot_client import Astribot

if __name__ == '__main__':
    # Connect astribot
    astribot = Astribot()
    joy_controller = XboxController(mode='chassis_control')

    freq = 250.0
    rate = rospy.Rate(freq)
    pos_cmd = [0.0, 0.0, 0.0]
    while not rospy.is_shutdown():
        vel_cmd = joy_controller.get_vel()
        pos_cmd[0] += vel_cmd[0] / freq
        pos_cmd[1] += vel_cmd[1] / freq
        pos_cmd[2] += vel_cmd[2] / freq
        astribot.set_joints_position([astribot.chassis_name], [pos_cmd])
        rate.sleep()
