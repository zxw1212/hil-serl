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
File: 208-get_closest_dist.py
Brief:  Example code for Astribot Robotics.
        Get the closest part and distance value of the robot itself, for the collision detection.
"""

from core.sdk_client.astribot_client import Astribot

if __name__ == '__main__':
    # Connect astribot
    astribot = Astribot()

    min_distance, link_A_name, link_B_name, \
        closest_point_on_A_of_torso_frame, \
        closest_point_on_B_of_torso_frame = \
        astribot.get_self_closest_point()

    print(f"Minimum distance: {min_distance}")
    print(f"Link A name: {link_A_name}")
    print(f"Link B name: {link_B_name}")
    print(f"Closest point pose on A of torso frame: {closest_point_on_A_of_torso_frame}")
    print(f"Closest point pose on B of torso frame: {closest_point_on_B_of_torso_frame}")
