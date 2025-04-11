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
File: 300-get_images.py
Brief:  Example code for Astribot Robotics.
        Get real-time images data from the robot camera.
"""

import cv2
import rospy
from core.sdk_client.astribot_client import Astribot

if __name__ == '__main__':
    astribot = Astribot()
    
    #1: Activate camera 
    cameras_setting = {
        'right_D405': {'flag_getdepth': True, 'flag_getIR': False},
        'Bolt': {'flag_getdepth': False},
        'Stereo': {'resolution':[1280,480]}
    }

    astribot.activate_camera(cameras_setting) #you can also activate camera in Orin astribot_multi_camera_driver
    
    #close the camera
    # astribot.deactivate_camera()
    
    #2: Get camera images
    while not rospy.is_shutdown():
        rgb_dict, depth_dict, ir_dict, time= astribot.get_images_dict()

        if rgb_dict is not None:
            for rgb in rgb_dict:
                if rgb_dict[rgb] is not None:
                    cv2.imshow(rgb+"_rgb", rgb_dict[rgb])

        if depth_dict is not None:
            for depth in depth_dict:
                if depth_dict[depth] is not None:
                    cv2.imshow(depth+"_depth", depth_dict[depth])

        if ir_dict is not None:
            for ir in ir_dict:
                for cam_id in ir_dict[ir]:
                    if ir_dict[ir][cam_id] is not None:
                        cv2.imshow(ir+"_ir", ir_dict[ir][cam_id])

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
