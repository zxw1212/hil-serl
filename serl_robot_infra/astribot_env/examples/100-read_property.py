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
File: 100-read_property.py
Brief:  Example code for Astribot Robotics.
        Print some robot properties.
"""

from core.sdk_client.astribot_client import Astribot

# Connect astribot
astribot = Astribot()
astribot.move_to_home()
# Print Astribot information
astribot.get_info()
print("DoF: ", astribot.get_dof())
upper_limit, lower_limit = astribot.get_joints_position_limit()
print(f"Upper position limit: {upper_limit}")
print(f"Lower position limit: {lower_limit}")
print("Velocity limit: ", astribot.get_joints_velocity_limit())
print("Torque limit: ", astribot.get_joints_torque_limit())
