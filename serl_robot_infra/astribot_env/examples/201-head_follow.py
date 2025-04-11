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
File: 201-head_follow.py
Brief:  Example code for Astribot Robotics.
        Set the head motion mode to following effector.
"""

from core.sdk_client.astribot_client import Astribot

if __name__ == '__main__':
    # Connect astribot
    astribot = Astribot()
    astribot.move_to_home()
    astribot.set_head_follow_effector(True)

    names = [astribot.torso_name, astribot.arm_left_name, astribot.arm_right_name]
    torso_cartesian_pose = astribot.get_current_cartesian_pose(names=[astribot.torso_name])
    left_arm_cartesian_pose = astribot.get_current_cartesian_pose(names=[astribot.arm_left_name])
    right_arm_cartesian_pose = astribot.get_current_cartesian_pose(names=[astribot.arm_right_name])

    torso_cartesian_pose = torso_cartesian_pose[0]
    left_arm_cartesian_pose = left_arm_cartesian_pose[0]
    right_arm_cartesian_pose = right_arm_cartesian_pose[0]

    left_arm_cartesian_pose[0] -= 0.08
    right_arm_cartesian_pose[0] -= 0.08
    command_list = [torso_cartesian_pose, left_arm_cartesian_pose, right_arm_cartesian_pose]
    astribot.move_cartesian_pose(names, command_list, duration=1.0, use_wbc=True)

    for idx in range(4):
        if idx % 2 == 0:
            left_arm_cartesian_pose[0] += 0.2
            right_arm_cartesian_pose[0] += 0.2
        else:
            left_arm_cartesian_pose[0] -= 0.2
            right_arm_cartesian_pose[0] -= 0.2
        command_list = [torso_cartesian_pose, left_arm_cartesian_pose, right_arm_cartesian_pose]
        astribot.move_cartesian_pose(names, command_list, duration=1.1, use_wbc=True)
    
    astribot.set_head_follow_effector(False)
