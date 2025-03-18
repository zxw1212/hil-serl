# Source the setup.bash file for the first ROS workspace
source ~/software/catkin_ws/devel/setup.bash

# Set ROS master URI to localhost
export ROS_MASTER_URI=http://localhost:11311

# Run the first instance of franka_server.py in the background
python franka_server.py \
    --robot_ip=192.168.0.3 \
    --gripper_type=Franka \
    --reset_joint_target=0,0,0,-1.9,-0,2,0 \
    --flask_url=127.0.0.1 \
    --ros_port=11311