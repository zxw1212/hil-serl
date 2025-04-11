
import cv2
import rospy
import numpy as np
from math import sqrt
from tools.joy_tools import XboxController
from core.sdk_client.astribot_client import Astribot

def factor_distance(distance, d_relative_safe, d_absolute_safe):
    if distance <= d_absolute_safe:
        return 0.0
    elif distance <= d_relative_safe:
        return (distance - d_absolute_safe) / (d_relative_safe - d_absolute_safe)
    else:
        return 1.0

def adjust_velocity(cmd_vel, sensor_distances, d_relative_safe, d_absolute_safe):

    adjusted_vel = np.array([cmd_vel[0], cmd_vel[1], cmd_vel[2]], dtype=np.float32)
    sensor_index_list = list()

    if cmd_vel[0] > 0:
        sensor_index_list.append([0, 1, 7])
    else:
        sensor_index_list.append([3, 4, 5])
    if cmd_vel[1] > 0:
        sensor_index_list.append([5, 6, 7])
    else:
        sensor_index_list.append([1, 2, 3])

    for i, sensor_index in enumerate(sensor_index_list):
        distance_list = [sensor_distances[index] for index in sensor_index]
        factor_list = [factor_distance(distance, d_relative_safe, d_absolute_safe) for distance in distance_list]
        factor = min(factor_list)
        if i == 0:
            adjusted_vel[0] = cmd_vel[0] * factor
        else:
            adjusted_vel[1] = cmd_vel[1] * factor

    return adjusted_vel

def init_plot_ultrasonic(d_relative_safe, d_absolute_safe):
    ultrasonic_image = np.ones((500, 500, 3), dtype=np.uint8)
    ultrasonic_image = np.multiply(ultrasonic_image, 255)

    pixel_relative_safe = [[250, 250-75.0*d_relative_safe], [250+75.0*sqrt(2.0)/2.0*d_relative_safe, 250-75.0*sqrt(2.0)/2.0*d_relative_safe],
                           [250+75.0*d_relative_safe, 250], [250+75.0*sqrt(2.0)/2.0*d_relative_safe, 250+75.0*sqrt(2.0)/2.0*d_relative_safe],
                           [250, 250+75.0*d_relative_safe], [250-75.0*sqrt(2.0)/2.0*d_relative_safe, 250+75.0*sqrt(2.0)/2.0*d_relative_safe],
                           [250-75.0*d_relative_safe, 250], [250-75.0*sqrt(2.0)/2.0*d_relative_safe, 250-75.0*sqrt(2.0)/2.0*d_relative_safe]]
    
    pixel_absolute_safe = [[250, 250-75.0*d_absolute_safe], [250+75.0*sqrt(2.0)/2.0*d_absolute_safe, 250-75.0*sqrt(2.0)/2.0*d_absolute_safe],
                           [250+75.0*d_absolute_safe, 250], [250+75.0*sqrt(2.0)/2.0*d_absolute_safe, 250+75.0*sqrt(2.0)/2.0*d_absolute_safe],
                           [250, 250+75.0*d_absolute_safe], [250-75.0*sqrt(2.0)/2.0*d_absolute_safe, 250+75.0*sqrt(2.0)/2.0*d_absolute_safe],
                           [250-75.0*d_absolute_safe, 250], [250-75.0*sqrt(2.0)/2.0*d_absolute_safe, 250-75.0*sqrt(2.0)/2.0*d_absolute_safe]]
    
    for start_idx in range(len(pixel_relative_safe)):
        if start_idx == len(pixel_relative_safe) - 1:
            end_idx = 0
        else:
            end_idx = start_idx + 1

        cv2.line(ultrasonic_image, (int(round(pixel_relative_safe[start_idx][0])), int(round(pixel_relative_safe[start_idx][1]))),
                    (int(round(pixel_relative_safe[end_idx][0])), int(round(pixel_relative_safe[end_idx][1]))), (0,0,0), 2)
        cv2.line(ultrasonic_image, (int(round(pixel_absolute_safe[start_idx][0])), int(round(pixel_absolute_safe[start_idx][1]))),
                    (int(round(pixel_absolute_safe[end_idx][0])), int(round(pixel_absolute_safe[end_idx][1]))), (0,0,0), 2)
    
    return ultrasonic_image

def plot_ultrasonic(ultrasonic_image, sensor_distances, d_relative_safe, d_absolute_safe, vis=True):
    realtime_ultrasonic_image = ultrasonic_image.copy()
    sensor_pixel = [[250, 250-75.0*sensor_distances[0]], [250+75.0*sqrt(2.0)/2.0*sensor_distances[1], 250-75.0*sqrt(2.0)/2.0*sensor_distances[1]],
                    [250+75.0*sensor_distances[2], 250], [250+75.0*sqrt(2.0)/2.0*sensor_distances[3], 250+75.0*sqrt(2.0)/2.0*sensor_distances[3]],
                    [250, 250+75.0*sensor_distances[4]], [250-75.0*sqrt(2.0)/2.0*sensor_distances[5], 250+75.0*sqrt(2.0)/2.0*sensor_distances[5]],
                    [250-75.0*sensor_distances[6], 250], [250-75.0*sqrt(2.0)/2.0*sensor_distances[7], 250-75.0*sqrt(2.0)/2.0*sensor_distances[7]]]
    
    for start_idx in range(len(sensor_pixel)):
        pixel_start = (int(round(sensor_pixel[start_idx][0])), int(round(sensor_pixel[start_idx][1])))
        if start_idx == len(sensor_pixel) - 1:
            end_idx = 0
        else:
            end_idx = start_idx + 1
        pixel_end = (int(round(sensor_pixel[end_idx][0])), int(round(sensor_pixel[end_idx][1])))
        
        color = cal_color(sensor_distances, d_relative_safe, d_absolute_safe, start_idx, end_idx)
        cv2.line(realtime_ultrasonic_image, pixel_start, pixel_end, color, 7)

    if vis:
        cv2.imshow('ultrasonic', realtime_ultrasonic_image)
        cv2.waitKey(1)
    
    return realtime_ultrasonic_image

def cal_color(sensor_distances, d_relative_safe, d_absolute_safe, start_idx, end_idx):
    if sensor_distances[start_idx] > d_relative_safe and sensor_distances[end_idx] > d_relative_safe:
        return (255, 0, 0)
    
    elif sensor_distances[start_idx] <= d_absolute_safe or sensor_distances[end_idx] <= d_absolute_safe:
        return (0, 0, 255)
    
    else:
        min_distance = min(sensor_distances[start_idx], sensor_distances[end_idx])
        color_factor = 255.0 * (1.0 - factor_distance(min_distance, d_relative_safe, d_absolute_safe))
        return (255-color_factor, 0, color_factor)

if __name__ == "__main__":
    astribot = Astribot()
    joy_controller = XboxController(mode='chassis_control')

    freq = 250.0
    rate = rospy.Rate(freq)
    pos_cmd = [0.0, 0.0, 0.0]
    d_relative_safe = 1.0
    d_absolute_safe = 0.5
    astribot.activate_ultrasonic()

    theta = 0.0
    sensor_distances = [2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0]
    image = init_plot_ultrasonic(d_relative_safe, d_absolute_safe)

    while not rospy.is_shutdown():
        vel_cmd = joy_controller.get_vel()
        sensor_distances = astribot.get_ultrasonic_distance()
        plot_ultrasonic(image, sensor_distances, d_relative_safe, d_absolute_safe, vis=True)
        vel_cmd = adjust_velocity(vel_cmd, sensor_distances, d_relative_safe, d_absolute_safe)
        pos_cmd[0] += vel_cmd[0] / freq
        pos_cmd[1] += vel_cmd[1] / freq
        pos_cmd[2] += vel_cmd[2] / freq
        astribot.set_joints_position([astribot.chassis_name], [pos_cmd])
        rate.sleep()

    cv2.destroyAllWindows()
    