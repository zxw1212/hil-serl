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
File: 101-get_joint_states.py
Brief:  Example code for Astribot Robotics.
        Get current and desired joint states of Astribot.
"""

from core.sdk_client.astribot_client import Astribot

if __name__ == '__main__':
    # Connect astribot
    astribot = Astribot()
    current_joints_position = astribot.get_current_joints_position()
    desired_joints_position = astribot.get_desired_joints_position()
    current_joints_velocity = astribot.get_current_joints_velocity()
    desired_joints_velocity = astribot.get_desired_joints_velocity()
    current_joints_torque = astribot.get_current_joints_torque()
    desired_joints_torque = astribot.get_desired_joints_torque()
    print("The results are displayed in the following order:", astribot.whole_body_names, "\n")
    print("Current joints position:", current_joints_position)
    print("Desired joints position:", desired_joints_position)
    print("Current joints velocity:", current_joints_velocity)
    print("Desired joints velocity:", desired_joints_velocity)
    print("Current joints torque:", current_joints_torque)
    print("Desired joints torque:", desired_joints_torque)
