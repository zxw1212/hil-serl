#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Copyright (c) 2025, Astribot Co., Ltd.
# All rights reserved.
# License: BSD 3-Clause License
# -----------------------------------------------------------------------------
# Author: Astribot Team
# -----------------------------------------------------------------------------

"""
File: 301-get_lidar_scan.py
Brief: Example code for Astribot Robotics.
       Get real-time LiDAR scan data from the robot.
"""

import rospy
from core.sdk_client.astribot_client import Astribot

import numpy as np



def livox2numpy(msg) -> np.array:
    """
    Converts a livox custom message containing LiDAR points to a NumPy array.

    Args:
        msg (CustomMsg): livox CustomMsg.

    Returns:
        np.array: A NumPy array of shape (N, 4) where N is the number of points.
                  Each row contains the x, y, z coordinates and intensity of a point.
    """
    msg_points = msg.points
    x = np.array([p.x for p in msg_points])
    y = np.array([p.y for p in msg_points])
    z = np.array([p.z for p in msg_points])
    intensity = np.array([p.reflectivity for p in msg_points])
    xyzi = np.stack((x, y, z, intensity), axis=-1)

    return xyzi

def custom_msg_callback(msg):
    pcd_np = livox2numpy(msg)
    print("Get pcd_np:", pcd_np)
    

if __name__ == '__main__':
    astribot = Astribot()

    # Activate LiDAR
    lidar_config={
        'rate': 10,
        'range': 8.0,
        'angle_min': -1.57,
    }
    astribot.activate_lidar(lidar_config)
    
    # Subscribe to the LiDAR topic
    # from livox_ros_driver2.msg import CustomMsg
    # rospy.Subscriber("/livox/lidar_192_168_1_12", CustomMsg, custom_msg_callback)
    
    #If you want to back lidar, you can use the following code
    # rospy.Subscriber("/livox/lidar_192_168_1_13", CustomMsg, custom_msg_callback)

    
