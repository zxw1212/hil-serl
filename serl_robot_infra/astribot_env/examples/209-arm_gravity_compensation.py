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
File: 209-arm_gravity_compensation.py
Brief: Example code for Astribot Robotics.
"""

import rospy
from core.sdk_client.astribot_client import Astribot

if __name__ == '__main__':
    # Connect astribot
    astribot = Astribot()

    names = astribot.arm_names
    torque_cmd = [[ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                  [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]
    freq = 500.0
    dt = 1.0 / freq
    rate = rospy.Rate(freq)
    while not rospy.is_shutdown():
        astribot.set_joints_torque(names, torque_cmd)#TODO(@11): 重力补偿跑飞，loop中加一个超过一定速度后失能机器人的判断
        rate.sleep()
