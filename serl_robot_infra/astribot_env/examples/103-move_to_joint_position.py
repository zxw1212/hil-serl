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
File: 103-move_to_joint_position.py
Brief:  Example code for Astribot Robotics.
        Move and set joint positions of Astribot.
"""

from core.sdk_client.astribot_client import Astribot

if __name__ == '__main__':
    # Connect astribot
    astribot = Astribot()
    astribot.move_to_home()

    names = [astribot.torso_name, astribot.arm_names[0], astribot.arm_names[1], astribot.effector_names[0], astribot.effector_names[1]]
    command_list = [[ 0.7, -1.4,  0.7, 0.0],
                    [ 0.5, -0.5, -1.9, 2.0, 0.0, 0, 0],
                    [-0.5, -0.5,  1.9, 2.0, 0.0, 0, 0],
                    [ 50.0],
                    [ 50.0]]
    response = astribot.move_joints_position(names, command_list, duration=3.0, use_wbc=True)
    print("Response from move_joints_position:", response)
