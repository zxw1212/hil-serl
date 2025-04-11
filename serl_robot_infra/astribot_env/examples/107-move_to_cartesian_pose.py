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
File: 107-move_to_cartesian_pose.py
Brief:  Example code for Astribot Robotics.
        Move and Set Cartesian Pose of Astribot.
"""

from core.sdk_client.astribot_client import Astribot

if __name__ == '__main__':
    # Connect astribot
    astribot = Astribot()
    astribot.move_to_home()

    names = [astribot.torso_name, astribot.arm_names[0], astribot.arm_names[1], astribot.effector_names[0], astribot.effector_names[1]]
    command_list = [[ 0.0, 0.0, 1.2, 0.0, 0.0, 0.0, 1.0],
                    [ 0.35,  0.35, 1.0, 0.0, 0.0, 0.707, 0.707],
                    [ 0.35, -0.35, 1.0, 0.0, 0.0, 0.707, 0.707],
                    [ 40.0],
                    [ 40.0]]
    response = astribot.move_cartesian_pose(names, command_list, duration=3.0, use_wbc=True)
    print("Response from move_cartesian:", response)
