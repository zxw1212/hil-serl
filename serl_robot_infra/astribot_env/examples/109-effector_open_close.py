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
File: 109-effector_open_close.py
Brief:  Example code for Astribot Robotics.
        Open and close effector of Astribot.
"""

import time
from core.sdk_client.astribot_client import Astribot

if __name__ == '__main__':
    astribot = Astribot()

    for idx in range(4):
        if idx % 2 == 0:
            #set gripper max force, right_max_force is changed by each grasp
            astribot.set_effector_max_force(left_max_force=40, right_max_force=idx*10)
            astribot.close_effector()
        else:
            astribot.open_effector()
        
        time.sleep(1.0)
