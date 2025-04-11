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
File: 302-get_and_play_audio.py
Brief: Example code for Astribot Robotics.
       Simplified audio recording, playback, and status display.
"""

import rospy
import cv2
import numpy as np
from core.sdk_client.astribot_client import Astribot

# Global variables
recording = False
audio_data = None
record_config={
    'channels': 1,
    'rate': 16000,
}
image_message = "Press 'r' to record, 's' to stop, 'p' to play, 'q' to quit"

def update_status_image(status):
    """Update the status image with the current message."""
    img = np.zeros((200, 400, 3), dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, status, (10, 100), font, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return img

if __name__ == '__main__':
    rospy.init_node('audio_publisher', anonymous=True)
    astribot = Astribot()

    # Activate speaker
    microhpone_config = {
        'channels': 1,
        'rate': 16000,
    }
    astribot.activate_microphone(microhpone_config)


    speaker_config = {
        'scale': 1.0, #bigger, louder, range: 0.0-1.0
        'rate': 16000,
    }
    astribot.activate_speaker(speaker_config)

    # Create initial status image
    show_image=np.zeros((200, 400, 3), dtype=np.uint8)
    show_image=cv2.putText(show_image, image_message, (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

    # Main loop
    while True:
        cv2.namedWindow("Audio Publisher")
        cv2.imshow("Audio Publisher", show_image)
        key = cv2.waitKey(1) & 0xFF  # Listen for key press events

        if key == ord('r'):
            print("Recording started...")
            astribot.record_audio(record_config)  # Replace with appropriate duration

        elif key == ord('s'):
            print("Recording stopped.")
            audio_data = astribot.stop_record()  # Get recorded audio data

        elif key == ord('p'):
            if audio_data:
                print("Playing audio...")
                astribot.pub_speaker(audio_data)
            else:
                print("No audio to play, Press 'r' to record.")

        elif key == ord('q'):
            print("Exiting program.")
            break


    cv2.destroyAllWindows()
