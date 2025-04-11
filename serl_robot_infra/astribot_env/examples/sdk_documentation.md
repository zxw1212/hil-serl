# Astribot SDK Documentation

## Overview
This document introduces the usage of Astribot Robot SDK, including robot state acquisition, motion control, function modules, etc. It is intended for developers who want to control Astribot robot through programming.

## Table of Contents
- [Fixed Parameters and State Acquisition](#fixed-parameters-and-state-acquisition)
- [Real-time State Acquisition](#real-time-state-acquisition)
- [Kinematics](#kinematics)
- [Robot Control](#robot-control)
- [Function Modules](#function-modules)
- [Others](#others)
- [Appendix](#appendix)

## Quick Start
Here's a simple example showing how to move the robot to home position:

```python
from astribot import Robot

# Create robot instance
robot = Robot()

# Move to home position
result = robot.move_to_home(duration=5.0)
print(result)
```

## Fixed Parameters and State Acquisition

#### Regular Parameters
The following regular parameters can be accessed directly as class variables
```
is_alive            # Robot activation flag
whole_body_names    # Names of all robot parts
chassis_name        # Robot chassis name
head_name           # Robot head name
torso_name          # Robot torso name
arm_left_name       # Robot left arm name
arm_right_name      # Robot right arm name
effector_left_name  # Robot left hand effector name
effector_right_name # Robot right hand effector name
arm_names           # Robot arms names, order: [left_arm, right_arm]
effector_names      # Robot hand effectors names, order: [left_hand, right_hand]
world_frame_name    # World coordinate frame name
chassis_frame_name  # Chassis coordinate frame name
whole_body_dofs     # Joint degrees of freedom for all robot parts
whole_body_cartesian_dofs   # Cartesian space degrees of freedom for all robot parts
chassis_dof         # Robot chassis joint space degrees of freedom
head_dof            # Robot head joint space degrees of freedom
torso_dof           # Robot torso joint space degrees of freedom
arm_dof             # Robot arm joint space degrees of freedom
effector_dof        # Robot effector joint space degrees of freedom
```

#### Get Joint Degrees of Freedom
```
function name:      get_info
input parameters:   names:list=None
output type:        list
function description: Input a list of names, output the joint space degrees of freedom corresponding to each part name in names. Returns degrees of freedom for all parts with default input.
eg: input:  [head_name[0], arm_names[0], arm_names[1]]
    output: [2, 7, 7]
```

#### Get Joint Position Limits
```
function name:      get_joints_position_limit
input parameters:   names:list=None
output type:        list, list
function description: Input a list of names, output the joint position upper and lower limits corresponding to each part name in names. Returns joint position limits for all parts with default input.
eg: input:  [head_name[0], arm_names[0], effector_names[0]]
    output: [[-1.57, -1.22], [-3.01, -1.5, -3.14, 0.001, -2.4, -0.75, -1.57], [0.0]] ,
            [[1.57, 1.22], [3.01, 0.25, 3.14, 2.618, 2.4, 0.75, 1.57], [100.0]]
```

#### Get Joint Velocity Limits
```
function name:      get_joints_velocity_limit
input parameters:   names:list=None
output type:        list
function description: Input a list of names, output the joint velocity limits corresponding to each part name in names. Returns joint velocity limits for all parts with default input.
others: To protect robot hardware, velocity limits vary under different modes.
eg: input:  [head_name[0], arm_names[0], effector_names[0]]
    output: [[4.0, 4.0], [7.4, 7.4, 7.4, 13.1, 18.1, 18.1, 18.1], [1000.0]]
```

#### Get Joint Torque Limits
```
function name:      get_joints_torque_limit
input parameters:   names:list=None
output type:        list
function description: Input a list of names, output the joint torque limits corresponding to each part name in names. Returns joint torque limits for all parts with default input.
others: To protect robot hardware, torque limits vary under different modes.
eg: input:  [head_name[0], arm_names[0], effector_names[0]]
    output: [[100.0, 100.0], [265.0, 265.0, 132.5, 142.0, 71.0, 20.0, 20.0], [1.0]]
```

## Real-time State Acquisition

#### Get Current Joint Positions
```
function name:      get_current_joints_position
input parameters:   names:list=None
output type:        list
function description: Input a list of names, output the current joint position states corresponding to each part name in names. Returns current joint positions for all parts with default input.
others: Output order and dimensions match input names; list dimension for each part equals its joint degrees of freedom.
eg: input:  [head_name[0], arm_names[0], effector_names[0]]
    output: [[head_q0, head_q1], [arm_q0, arm_q1, arm_q2, arm_q3, arm_q4, arm_q5, arm_q6], [effector_q0]]
```

#### Get Current Joint Velocities
```
function name:      get_current_joints_velocity
input parameters:   names:list=None
output type:        list
function description: Input a list of names, output the current joint velocity states corresponding to each part name in names. Returns current joint velocities for all parts with default input.
others: Output order and dimensions match input names; list dimension for each part equals its joint degrees of freedom.
eg: input:  [head_name[0], arm_names[0], effector_names[0]]
    output: [[head_qdot0, head_qdot1], [arm_qdot0, arm_qdot1, arm_qdot2, arm_qdot3, arm_qdot4, arm_qdot5, arm_qdot6], [effector_qdot0]]
```

#### Get Current Joint Accelerations
```
function name:      get_current_joints_acceleration
input parameters:   names:list=None
output type:        list
function description: Input a list of names, output the current joint acceleration states corresponding to each part name in names. Returns current joint accelerations for all parts with default input.
others: Output order and dimensions match input names; list dimension for each part equals its joint degrees of freedom.
eg: input:  [head_name[0], arm_names[0], effector_names[0]]
    output: [[head_qddot0, head_qddot1], [arm_qddot0, arm_qddot1, arm_qddot2, arm_qddot3, arm_qddot4, arm_qddot5, arm_qddot6], [effector_qddot0]]
```

#### Get Current Joint Torques
```
function name:      get_current_joints_torque
input parameters:   names:list=None
output type:        list
function description: Input a list of names, output the current joint torque states corresponding to each part name in names. Returns current joint torques for all parts with default input.
others: Output order and dimensions match input names; list dimension for each part equals its joint degrees of freedom.
eg: input:  [head_name[0], arm_names[0], effector_names[0]]
    output: [[head_qtor0, head_qtor1], [arm_qtor0, arm_qtor1, arm_qtor2, arm_qtor3, arm_qtor4, arm_qtor5, arm_qtor6], [effector_qtor0]]
```

#### Get Desired Joint Positions
```
function name:      get_desired_joints_position
input parameters:   names:list=None
output type:        list
function description: Input a list of names, output the current desired joint position values corresponding to each part name in names. Returns current desired joint positions for all parts with default input.
others: Output order and dimensions match input names; list dimension for each part equals its joint degrees of freedom.
eg: input:  [head_name[0], arm_names[0], effector_names[0]]
    output: [[head_q0, head_q1], [arm_q0, arm_q1, arm_q2, arm_q3, arm_q4, arm_q5, arm_q6], [effector_q0]]
```

#### Get Desired Joint Velocities
```
function name:      get_desired_joints_velocity
input parameters:   names:list=None
output type:        list
function description: Input a list of names, output the current desired joint velocity values corresponding to each part name in names. Returns current desired joint velocities for all parts with default input.
others: Output order and dimensions match input names; list dimension for each part equals its joint degrees of freedom.
eg: input:  [head_name[0], arm_names[0], effector_names[0]]
    output: [[head_qdot0, head_qdot1], [arm_qdot0, arm_qdot1, arm_qdot2, arm_qdot3, arm_qdot4, arm_qdot5, arm_qdot6], [effector_qdot0]]
```

#### Get Desired Joint Torques
```
function name:      get_desired_joints_torque
input parameters:   names:list=None
output type:        list
function description: Input a list of names, output the current desired joint torque values corresponding to each part name in names. Returns current desired joint torques for all parts with default input.
others: Output order and dimensions match input names; list dimension for each part equals its joint degrees of freedom.
eg: input:  [head_name[0], arm_names[0], effector_names[0]]
    output: [[head_qtor0, head_qtor1], [arm_qtor0, arm_qtor1, arm_qtor2, arm_qtor3, arm_qtor4, arm_qtor5, arm_qtor6], [effector_qtor0]]
```

#### Get Current Cartesian Pose
```
function name:      get_current_cartesian_pose
input parameters:   names:list=None, frame:str="chassis"
output type:        list
function description: Input a list of names and a coordinate frame name. Output the current Cartesian space states of each part name in names relative to the specified frame. Returns current Cartesian states for all parts with default input, using chassis frame by default.
others: Output order and dimensions match input names; list dimension for each part equals its Cartesian space degrees of freedom. For grippers, it's 1D and returns joint space position; for other parts, it's 7D with format xyz + quaternion (qx, qy, qz, qw).
eg: input:  [arm_names[0], effector_names[0]], chassis_frame_name
    output: [[x, y, z, qx, qy, qz, qw], [joint_q0]]
```

#### Get Current Cartesian Pose Through TF Tree
```
function name:      get_current_cartesian_pose_tf
input parameters:   target_frame:str, source_frame:str
output type:        list
function description: Get the pose of source_frame relative to target_frame through the tf tree. This function supports getting relative poses between any two parts of the robot, such as getting the arm end-effector pose relative to the torso frame.
others: Output is 7D data in format xyz + quaternion (qx, qy, qz, qw). target_frame and source_frame can be coordinate frames of any robot part.
eg: input:  "astribot_torso", "astribot_arm_left_link7" 
    output: [x, y, z, qx, qy, qz, qw]  # Left arm end-effector pose relative to torso
```

#### Get Desired Cartesian Pose
```
function name:      get_desired_cartesian_pose
input parameters:   names:list=None, frame:str="chassis"
output type:        list
function description: Input a list of names and a coordinate frame name. Output the current desired Cartesian space values of each part name in names relative to the specified frame. Returns current desired Cartesian values for all parts with default input, using chassis frame by default.
others: Output order and dimensions match input names; list dimension for each part equals its Cartesian space degrees of freedom. For grippers, it's 1D and returns joint space position; for other parts, it's 7D with format xyz + quaternion (qx, qy, qz, qw).
eg: input:  [arm_names[0], effector_names[0]], chassis_frame_name
    output: [[x, y, z, qx, qy, qz, qw], [joint_q0]]
```

#### Notes on Real-time State Acquisition
- Difference between current states and desired values: Desired values are the commands currently sent to the robot, while states are the actual positions the robot has reached. There will be small errors between desired values and states, including steady-state errors and delays. When performing closed-loop control, use current desired values as input; when needing to get the robot's current precise position, use current states.
- Difference between get_current_cartesian_pose and get_current_cartesian_pose_tf: get_current_cartesian_pose only supports chassis frame and world frame, while get_current_cartesian_pose_tf supports using any robot part as the relative coordinate frame, such as getting arm end-effector position relative to torso frame.
- Coordinate frame explanation: World frame is fixed at the chassis center position when robot starts, with z-axis up and x-axis forward, following right-hand rule. Chassis frame is at chassis center, with z-axis up and x-axis forward, following right-hand rule. When robot just starts or chassis hasn't moved, world frame and chassis frame coincide.

## Kinematics

#### Forward Kinematics
```
function name:      get_forward_kinematics
input parameters:   names:list, joints_position:list
output type:        dict
function description: Input a list of names and a list of joint positions (joints_position) corresponding to each part in names, output the robot's Cartesian space positions under the input joint positions in chassis frame. Note: Must input at least torso name and its corresponding joint positions, as arm or head forward kinematics cannot be calculated without torso position. Also, gripper names and joint positions are not supported as grippers only have one joint and don't involve kinematics processing.
eg: input:  [torso_name[0], arm_names[0]], [[torso_q0, torso_q1, torso_q2, torso_q3], [arm_q0, arm_q1, arm_q2, arm_q3, arm_q4, arm_q5, arm_q6]]
    output: {'astribot_arm_left': [x, y, z, qx, qy, qz, qw],
             'astribot_torso': [x, y, z, qx, qy, qz, qw]}
```

#### Inverse Kinematics
```
function name:      get_inverse_kinematics
input parameters:   names:list, cartesian_pose_list:list
output type:        dict
function description: Input a list of names and a list of Cartesian space positions (in chassis frame) corresponding to each part in names, output the robot's joint space solutions under the input Cartesian positions. Note: Currently only supports torso and arms input.
eg: input:  [torso_name[0], arm_names[0]], [[x, y, z, qx, qy, qz, qw], [x, y, z, qx, qy, qz, qw]]
    output: {'astribot_arm_left': [arm_q0, arm_q1, arm_q2, arm_q3, arm_q4, arm_q5, arm_q6],
             'astribot_torso': [torso_q0, torso_q1, torso_q2, torso_q3]}
```

## Robot Control

#### Move Robot from Current Position to Home Pose
```
function name:      move_to_home
input parameters:   duration=5.0, use_wbc:bool=False
output type:        string
function description: Move all parts of the robot from current position to the preset home pose in the specified time.
parameters description: duration is the specified time in seconds, default is 5 seconds. use_wbc is the flag for enabling WBC (Whole Body Control).
output description: Function call starts the motion, returns a string indicating motion result after completion.
```

#### Move Robot Joint Position from Current Position to Target
```
function name:      move_joints_position
input parameters:   names:list, commands:list, duration=5.0, use_wbc:bool=False, add_default_torso:bool=True
output type:        string
function description: Move specified parts of the robot from current position to input target joint space positions in the specified time.
parameters description: names is a list containing specified part names, indicating which parts of the robot to move. commands is the joint position setpoints, order must match names. duration is the specified time in seconds, default is 5 seconds. use_wbc is the flag for enabling WBC. add_default_torso determines whether to add default torso target.
output description: Function call starts the motion, returns a string indicating motion result after completion.
```

#### Move Robot Cartesian Position from Current Position to Target
```
function name:      move_cartesian_pose
input parameters:   names:list, commands:list, duration=5.0, use_wbc:bool=True, frame:str="chassis", add_default_torso:bool=True
output type:        string
function description: Move specified parts of the robot from current position to input target Cartesian space positions (in specified frame) in the specified time.
parameters description: names is a list containing specified part names, indicating which parts of the robot to move. commands is the Cartesian space position setpoints, order must match names. duration is the specified time in seconds, default is 5 seconds. use_wbc is the flag for enabling WBC. frame is the coordinate frame name indicating which frame the user's commands are in. add_default_torso determines whether to add default torso target.
output description: Function call starts the motion, returns a string indicating motion result after completion.
```

#### Real-time Joint Position Control
```
function name:      set_joints_position
input parameters:   names:list, position:list, control_way:str="filter", use_wbc:bool=False, freq=None, add_default_torso:bool=True
output type:        None
function description: Move specified parts of the robot in real-time from current position to input target joint space positions.
parameters description: names is a list containing specified part names, indicating which parts of the robot to move. position is the joint space position targets, order must match names. control_way specifies the command sending method, can be "direct" or "filter". "direct" sends commands directly with higher precision but requires smooth inputs without jumps to avoid velocity limit errors; "filter" sends filtered commands that are smoother but may have delays affecting precision. use_wbc is the flag for enabling WBC. freq is required only when control_way is "direct" and use_wbc=False, as this is the most direct joint position control method that sends user commands without any processing, requiring user to provide their calling frequency for velocity target calculation. add_default_torso determines whether to add default torso target.
```

#### Real-time Joint Velocity Control
```
function name:      set_joints_velocity
input parameters:   names:list, velocity:list
output type:        None
function description: Make all joints of specified robot parts run in real-time at desired velocities.
parameters description: names is a list containing specified part names, indicating which parts of the robot to move. velocity is the joint space desired velocities, order must match names.
```

#### Real-time Joint Torque Control
```
function name:      set_joints_torque
input parameters:   names:list, torque:list
output type:        None
function description: Make all joints of specified robot parts run in real-time with desired torques.
parameters description: names is a list containing specified part names, indicating which parts of the robot to move. torque is the joint space desired torques, order must match names.
```

#### Real-time Cartesian Position Control
```
function name:      set_cartesian_pose
input parameters:   names:list, cartesian_pose:list, control_way:str="filter", use_wbc:bool=False, frame:str="chassis", add_default_torso:bool=True
output type:        None
function description: Move specified parts of the robot in real-time from current position to input target Cartesian space positions (in specified frame).
parameters description: names is a list containing specified part names, indicating which parts of the robot to move. cartesian_pose is the Cartesian space position targets, order must match names. control_way specifies the command sending method, can be "direct" or "filter". "direct" sends commands directly with higher precision but requires smooth inputs without jumps to avoid velocity limit errors; "filter" sends filtered commands that are smoother but may have delays affecting precision. use_wbc is the flag for enabling WBC. frame is the coordinate frame name indicating which frame the user's cartesian_pose is in. add_default_torso determines whether to add default torso target.
```

#### Joint Space Multi-point Following
```
function name:      move_joints_waypoints
input parameters:   names:list, waypoints:list, time_list:list, use_wbc:bool=False, joy_controller=False, add_default_torso:bool=True
output type:        string
function description: Move specified parts of the robot from current position to a sequence of input target joint space waypoints in specified times.
parameters description: names is a list containing specified part names, indicating which parts of the robot to move. waypoints is a series of joint position setpoints, a 3D list where each waypoint's order must match names. time_list is a list of time parameters matching the number of waypoints, specifying the absolute time (in seconds) to reach each point, starting from 0. use_wbc is the flag for enabling WBC. joy_controller determines whether to enable joystick playback control. add_default_torso determines whether to add default torso target.
output description: Function call starts the motion, returns a string indicating motion result after completion.
```

#### Cartesian Space Multi-point Following
```
function name:      move_cartesian_waypoints
input parameters:   names:list, waypoints:list, time_list:list, use_wbc:bool=True, joy_controller=False, frame:str="chassis", add_default_torso:bool=True
output type:        string
function description: Move specified parts of the robot from current position to a sequence of input target Cartesian space waypoints (in specified frame) in specified times.
parameters description: names is a list containing specified part names, indicating which parts of the robot to move. waypoints is a series of Cartesian position setpoints, a 3D list where each waypoint's order must match names. time_list is a list of time parameters matching the number of waypoints, specifying the absolute time (in seconds) to reach each point, starting from 0. use_wbc is the flag for enabling WBC. joy_controller determines whether to enable joystick playback control. frame is the coordinate frame name indicating which frame the user's waypoints are in. add_default_torso determines whether to add default torso target.
output description: Function call starts the motion, returns a string indicating motion result after completion.
```

#### Close Effector
```
function name:      close_effector
input parameters:   names:list=None
output type:        string
function description: Close specified robot effectors.
parameters description: names is a list containing specified part names, only accepts effector part names, closes all effectors with default input.
output description: Function call starts the motion, returns a string indicating motion result after completion.
```

#### Open Effector
```
function name:      open_effector
input parameters:   names:list=None
output type:        string
function description: Open specified robot effectors.
parameters description: names is a list containing specified part names, only accepts effector part names, opens all effectors with default input.
output description: Function call starts the motion, returns a string indicating motion result after completion.
```

#### Notes on Robot Control
- Difference between move and set: move functions run blockingly, where users specify target points and time, and the robot moves there with zero start and end velocities. set functions are online driving, where users should cyclically call set functions at a certain frequency to send targets, and the robot follows targets in real-time in each cycle period. set driving is online, real-time, and non-blocking.
- WBC functionality: Without WBC in joint space control, each joint independently controls movement to target position. With WBC in joint space control, joint positions may not reach target joint positions but will reach the Cartesian space position corresponding to target joint configuration. WBC is a whole-body control that comprehensively considers control targets of all parts. If arm precision is more important (e.g., only arm commands given), it will use torso compensation to help arm reach target position.
- add_default_torso functionality: When arm pose targets are given without torso pose targets and WBC is enabled, WBC might calculate a large torso movement to track arm poses. However, this may not be what users want - they may have only given arm pose targets, want to use WBC, but don't want torso movement. In this case, setting add_default_torso to True will automatically add a torso target to prevent large torso movements when user input torso targets are missing.
- joy_controller functionality: In multi-point following (path following), if joy_controller is True, joystick control playback functionality will be enabled. Hold RT button for forward motion, hold LT button for reverse motion.

## Function Modules

#### Set Filter Parameters
```
function name:      set_filter_parameters
input parameters:   filter_scale, gripper_filter_scale
output type:        string
function description: When controlling robot real-time movement through set_ functions, you can set control_way="filter" to filter user commands. There is an internal filter parameter, which users can set through this function. Different filter parameters can be set for effector and non-effector parts.
parameters description: filter_scale is the filter parameter for non-effector parts, gripper_filter_scale is the filter parameter for effectors. Filter parameters range from 0 to 1, closer to 0 means stronger filtering, closer to 1 means closer to no filtering.
output description: Returns a string indicating whether setting was successful or failed.
```

#### Head Following Effectors
```
function name:      set_head_follow_effector
input parameters:   enable:bool=True
output type:        string
function description: When function is called with True input, enables head tracking of dual arm effectors. Head will not respond to other control commands. This function is for keeping head camera observing operation scene while robot performs tasks. When called with False input, disables head tracking of dual arm effectors.
output description: Returns a string indicating whether setting was successful or failed.
```

#### Robot Self-collision Avoidance
```
function name:      set_wbc_collision_avoidance
input parameters:   enable:bool=True
output type:        string
function description: When function is called with True input, enables robot self-collision avoidance functionality, only effective in WBC mode. If potential self-collision situations occur during robot movement, control commands will be constrained to prevent self-collision. When called with False input, disables robot self-collision avoidance functionality.
output description: Returns a string indicating whether setting was successful or failed.
```

#### Get Robot Self Closest Points
```
function name:      get_self_closest_point
input parameters:   None
output type:        min_distance:float32, link_A_name:string, link_B_name:string, closest_point_on_A_of_torso_frame:list, closest_point_on_B_of_torso_frame:list
function description: Only has valid return values after using WBC to control robot movement. Calling function returns the closest two points on robot self in current state.
output description: Returns the minimum distance between two closest points on robot self (min_distance), returns names of parts where two points are located (link_A_name and link_B_name), returns positions of these two points in torso frame (closest_point_on_A_of_torso_frame and closest_point_on_B_of_torso_frame).
```

#### Activate Camera
```
function name:      activate_camera
input parameters:   cameras_info:dict=None
output type:        string
function description: Turn on cameras. After calling this function, real-time images can be obtained through get_images_dict.
parameters description: Input a dictionary to specify which cameras to activate.
{
    'left_D405': {'flag_getdepth': True, 'flag_getIR': True},
    'right_D405': {'flag_getdepth': True, 'flag_getIR': True},
    'Gemini335': {'flag_getdepth': True, 'flag_getIR': True},
    'Bolt': {'flag_getdepth': True},
    'Stereo': {}
}
This is an example of all camera inputs. When a name exists in dictionary, that camera's RGB image is activated. If flag_getdepth is True in a name's sub-dictionary, that camera's depth image is activated. If flag_getIR is True in a name's sub-dictionary, that camera's IR image is activated. left_D405, right_D405, Gemini335 have all three types of images, Bolt only has RGB and depth images, Stereo only has RGB.
output description: Returns a string indicating whether activation was successful or failed.
```

#### Get Images
```
function name:      get_images_dict
input parameters:   None
output type:        dict, dict, dict
function description: After cameras are activated, real-time images can be obtained through this function.
output description: Returns three dictionaries: RGB image dictionary, depth image dictionary, and IR image dictionary. Names in dictionaries are camera names, i.e., left_D405, right_D405, Gemini335, Bolt, Stereo.
```

## Others

#### Soft Emergency Stop
```
function name:      stop_robot
input parameters:   None
output type:        string
function description: When called, robot immediately stops at current position and stops responding to any motion commands.
output description: Returns a string indicating whether stop was successful or failed.
```

#### Soft Emergency Stop Recovery
```
function name:      restart_robot
input parameters:   None
output type:        string
function description: When called, robot exits soft emergency stop state and starts responding to user motion commands again.
output description: Returns a string indicating whether recovery was successful or failed.
```

## Appendix

### Appendix A: Technical Terms

#### Coordinate Systems
- **World Frame**: Fixed coordinate frame at robot chassis center position when robot starts, with z-axis up and x-axis forward, following right-hand rule.
- **Chassis Frame**: Relative coordinate frame fixed at robot chassis center, with z-axis up and x-axis forward, following right-hand rule. When robot just starts, world frame and chassis frame coincide.

#### Space Representation
- **Joint Space**: Space that describes robot position using joint angles. For example, an arm with 7 joints uses 7 angle values to represent its position.
- **Cartesian Space**: Space that describes robot end-effector position and orientation in 3D space using position (x,y,z) and orientation (quaternion).

#### Control Methods
- **WBC (Whole Body Control)**: A control method that considers the robot's entire body movement. It coordinates all joint movements to achieve targets, such as using torso adjustment to help arms reach target positions.
- **Filter Control**: Control method that smooths user commands to avoid sudden changes in robot movement.
- **Direct Control**: Control method that directly executes user commands, providing higher precision but requiring smooth inputs.

#### Other Concepts
- **DOF (Degree of Freedom)**: Number of independent movements possible for each robot part. For example, an arm with 7 joints has 7 degrees of freedom.
- **Effector**: Robot's end mechanism, such as grippers.
- **Self-collision Avoidance**: Functionality that automatically prevents robot self-collision by monitoring distances between robot parts.

## Appendix B: FAQ

### 1. How to Choose Appropriate Control Method?
Q: When should I use joint space control vs. Cartesian space control?
A: Use joint space control when you care about specific joint angles; use Cartesian space control when you care about end-effector position and orientation.

### 2. WBC Control Related Questions
Q: When should WBC be enabled?
A: Enable WBC when you need to coordinate multiple parts' movements, or when you need torso compensation to help arms reach target positions.

### 3. Common Error Handling
Q: Why did the robot suddenly stop responding to commands?
A: Soft emergency stop might have been triggered. Check if:
- Motion commands exceeded limits
- Potential self-collision was detected
- stop_robot() was manually called 