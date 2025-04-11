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
File: 999.stop_robot.py
Brief:  Code for Stop Astribot Robotics.
"""

import time
import threading
from core.sdk_client.astribot_client import Astribot

if __name__ == '__main__':
    # Connect astribot
    astribot = Astribot()

    move_to_home_thread = threading.Thread(target=astribot.move_to_home)
    move_to_home_thread.start()

    time.sleep(3)

    print(astribot.stop_robot())

    time.sleep(3)

    print(astribot.restart_robot())
    astribot.move_to_home(duration=2.0)
