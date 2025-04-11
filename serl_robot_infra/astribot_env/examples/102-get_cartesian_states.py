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
File: 102-get_cartesian_states.py
Brief:  Example code for Astribot Robotics.
        Get current and desired cartesian states of Astribot.
"""

from core.sdk_client.astribot_client import Astribot
from scipy.spatial.transform import Rotation as R

if __name__ == '__main__':
    # Connect astribot
    astribot = Astribot()
    current_cartesian_pose = astribot.get_current_cartesian_pose(frame=astribot.world_frame_name)
    desired_cartesian_pose = astribot.get_desired_cartesian_pose(frame=astribot.chassis_frame_name)
    current_cartesian_pose_tf = astribot.get_current_cartesian_pose_tf()
    print("The results are displayed in the following order:", astribot.whole_body_names, "\n")
#     print("Current cartesian pose:", current_cartesian_pose)
#     print("Desired cartesian pose:", desired_cartesian_pose)
#     print("Current cartesian pose tf:", current_cartesian_pose_tf)

    print("right_arm_cartesian_pose:", current_cartesian_pose[4])
    quaternion = current_cartesian_pose[4][3:7]
    rpy = R.from_quat(quaternion).as_euler('xyz', degrees=False)
    print("right_arm_cartesian_pose rpy:", rpy)

