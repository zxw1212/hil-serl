# Astribot SDK 使用文档

## 概述
本文档介绍了Astribot机器人SDK的使用方法,包括机器人状态获取、运动控制、功能模块等内容。适用于想要通过编程方式控制Astribot机器人的开发者。

## 目录
- [固定参数或状态获取](#固定参数或状态获取) 
- [实时状态获取](#实时状态获取)
- [运动学](#运动学)
- [机器人控制](#机器人控制)
- [功能模块](#功能模块)
- [其他](#其他)
- [附录](#附录)

## 快速开始
以下是一个简单的示例,展示如何控制机器人移动到home位置:

```python
from astribot import Robot

# 创建机器人实例
robot = Robot()

# 移动到home位置
result = robot.move_to_home(duration=5.0)
print(result)
```

## 固定参数或状态获取

#### 常规参数
以下常规参数直接访问类内变量即可
```
is_alive            # 机器人是否启动的标志
whole_body_names    # 机器人所有部位 names
chassis_name        # 机器人底盘 name
head_name           # 机器人头部 name
torso_name          # 机器人腰部 name
arm_left_name       # 机器人左臂 name
arm_right_name      # 机器人右臂 name
effector_left_name  # 机器人左手执行器 name
rigth_effector_name # 机器人右手执行器 name
arm_names           # 机器人双臂 names，顺序为 [左臂, 右臂]
effector_names      # 机器人双手执行器 names，顺序为 [左手, 右手]
world_frame_name    # 世界坐标系 name
chassis_frame_name  # 底盘坐标系 name
whole_body_dofs     # 机器人所有部位的关节自由度
whole_body_cartesian_dofs   # 机器人所有部位的笛卡尔空间自由度
chassis_dofs        # 机器人底盘关节空间自由度
head_dofs           # 机器人头部关节空间自由度
torso_dofs          # 机器人腰部关节空间自由度
arm_dofs            # 机器人手臂关节空间自由度
effector_dofs       # 机器人执行器关节空间自由度
```
#### 获取关节自由度
```
function name:      get_info
input parameters:   names:list=None
output type:        list
function description: 输入一组names，输出names中每一个部位name对应的关节空间自由度。默认输入下返回所有部位的关节自由度
eg: input:  [head_name[0], arm_names[0], arm_names[1]]
    output: [2, 7, 7]
```
#### 获取关节位置限幅
```
function name:      get_joints_position_limit
input parameters:   names:list=None
output type:        list, list
function description: 输入一组names，输出names中每一个部位name对应的关节位置上限和下限。默认输入下返回所有部位的关节位置限幅
eg: input:  [head_name[0], arm_names[0], effector_names[0]]
    output: [[-1.57, -1.22], [-3.01, -1.5, -3.14, 0.001, -2.4, -0.75, -1.57], [0.0]] ,
            [[1.57, 1.22], [3.01, 0.25, 3.14, 2.618, 2.4, 0.75, 1.57], [100.0]]
```
#### 获取关节速度限幅
```
function name:      get_joints_velocity_limit
input parameters:   names:list=None
output type:        list
function description: 输入一组names，输出names中每一个部位name对应的关节速度限幅。默认输入下返回所有部位的关节速度限幅
others: 为了保护机器人硬件，不同模式设置下速度限幅不同。
eg: input:  [head_name[0], arm_names[0], effector_names[0]]
    output: [[4.0, 4.0], [7.4, 7.4, 7.4, 13.1, 18.1, 18.1, 18.1], [1000.0]]
```
#### 获取关节力矩限幅
```
function name:      get_joints_torque_limit
input parameters:   names:list=None
output type:        list
function description: 输入一组names，输出names中每一个部位name对应的关节力矩限幅。默认输入下返回所有部位的关节力矩限幅
others: 为了保护机器人硬件，不同模式设置下力矩限幅不同。
eg: input:  [head_name[0], arm_names[0], effector_names[0]]
    output: [[100.0, 100.0], [265.0, 265.0, 132.5, 142.0, 71.0, 20.0, 20.0], [1.0]]
```



## 实时状态获取

#### 获取机器人当前关节位置状态值
```
function name:      get_current_joints_position
input parameters:   names:list=None
output type:        list
function description: 输入一组names，输出names中每一个部位name对应的当前关节位置状态值。默认输入下返回所有部位的当前关节位置状态值
others: 输出的顺序和输入的names顺序一致，维度一致；每个部位对应的list维度等于该部位关节自由度。
eg: input:  [head_name[0], arm_names[0], effector_names[0]]
    output: [[head_q0, head_q1], [arm_q0, arm_q1, arm_q2, arm_q3, arm_q4, arm_q5, arm_q6], [effector_q0]]
```
#### 获取机器人当前关节速度状态值
```
function name:      get_current_joints_velocity
input parameters:   names:list=None
output type:        list
function description: 输入一组names，输出names中每一个部位name对应的当前关节速度状态值。默认输入下返回所有部位的当前关节速度状态值
others: 输出的顺序和输入的names顺序一致，维度一致；每个部位对应的list维度等于该部位关节自由度。
eg: input:  [head_torso[0], arm_names[0], effector_names[0]]
    output: [[head_qdot0, head_qdot1], [arm_qdot0, arm_qdot1, arm_qdot2, arm_qdot3, arm_qdot4, arm_qdot5, arm_qdot6], [effector_qdot0]]
```
#### 获取机器人当前关节加速度状态值
```
function name:      get_current_joints_acceleration
input parameters:   names:list=None
output type:        list
function description: 输入一组names，输出names中每一个部位name对应的当前关节加速度状态值。默认输入下返回所有部位的当前关节加速度状态值
others: 输出的顺序和输入的names顺序一致，维度一致；每个部位对应的list维度等于该部位关节自由度。
eg: input:  [head_name[0], arm_names[0], effector_names[0]]
    output: [[head_qddot0, head_qddot1], [arm_qddot0, arm_qddot1, arm_qddot2, arm_qddot3, arm_qddot4, arm_qddot5, arm_qddot6], [effector_qddot0]]
```
#### 获取机器人当前关节力矩状态值
```
function name:      get_current_joints_torque
input parameters:   names:list=None
output type:        list
function description: 输入一组names，输出names中每一个部位name对应的当前关节力矩状态值。默认输入下返回所有部位的当前关节力矩状态值
others: 输出的顺序和输入的names顺序一致，维度一致；每个部位对应的list维度等于该部位关节自由度。
eg: input:  [head_name[0], arm_names[0], effector_names[0]]
    output: [[head_qtor0, head_qtor1], [arm_qtor0, arm_qtor1, arm_qtor2, arm_qtor3, arm_qtor4, arm_qtor5, arm_qtor6], [effector_qtor0]]
```
#### 获取机器人当前关节位置期望值
```
function name:      get_desired_joints_position
input parameters:   names:list=None
output type:        list
function description: 输入一组names，输出names中每一个部位name对应的当前关节位置期望值。默认输入下返回所有部位的当前关节位置期望值
others: 输出的顺序和输入的names顺序一致，维度一致；每个部位对应的list维度等于该部位关节自由度。
eg: input:  [head_name[0], arm_names[0], effector_names[0]]
    output: [[head_q0, head_q1], [arm_q0, arm_q1, arm_q2, arm_q3, arm_q4, arm_q5, arm_q6], [effector_q0]]
```
#### 获取机器人当前关节速度期望值
```
function name:      get_desired_joints_velocity
input parameters:   names:list=None
output type:        list
function description: 输入一组names，输出names中每一个部位name对应的当前关节速度期望值。默认输入下返回所有部位的当前关节速度期望值
others: 输出的顺序和输入的names顺序一致，维度一致；每个部位对应的list维度等于该部位关节自由度。
eg: input:  [head_name[0], arm_names[0], effector_names[0]]
    output: [[head_qdot0, head_qdot1], [arm_qdot0, arm_qdot1, arm_qdot2, arm_qdot3, arm_qdot4, arm_qdot5, arm_qdot6], [effector_qdot0]]
```
#### 获取机器人当前关节力矩期望值
```
function name:      get_desired_joints_torque
input parameters:   names:list=None
output type:        list
function description: 输入一组names，输出names中每一个部位name对应的当前关节力矩期望值。默认输入下返回所有部位的当前关节力矩期望值
others: 输出的顺序和输入的names顺序一致，维度一致；每个部位对应的list维度等于该部位关节自由度。
eg: input:  [head_name[0], arm_names[0], effector_names[0]]
    output: [[head_qtor0, head_qtor1], [arm_qtor0, arm_qtor1, arm_qtor2, arm_qtor3, arm_qtor4, arm_qtor5, arm_qtor6], [effector_qtor0]]
```
#### 获取机器人当前笛卡尔空间位置状态值
```
function name:      get_current_cartesian_pose
input parameters:   names:list=None, frame:str="chassis"
output type:        list
function description: 输入一组names，和一个坐标系name。输出在指定坐标系下，names中每一个部位name对应的当前笛卡尔空间状态值。names默认输入下返回所有部位的当前笛卡尔空间状态值，frame默认输入下为底盘坐标系。
others: 输出的顺序和输入的names顺序一致，维度一致；每个部位对应的list维度等于该部位笛卡尔空间自由度，夹爪为1维，仍返回关节空间位置；其他部位为7维，数据格式为xyz+四元数(qx, qy, qz, qw)。
eg: input:  [harm_names[0], effector_names[0]], chassis_frame_name
    output: [[x, y, z, qx, qy, qz, qw], [joint_q0]]
```


#### 获取机器人当前笛卡尔空间位置期望值
```
function name:      get_desired_cartesian_pose
input parameters:   names:list=None, frame:str="chassis"
output type:        list
function description: 输入一组names，和一个坐标系name。输出在指定坐标系下，names中每一个部位name对应的当前笛卡尔空间期望值。names默认输入下返回所有部位的当前笛卡尔空间期望值，frame默认输入下为底盘坐标系。
others: 输出的顺序和输入的names顺序一致，维度一致；每个部位对应的list维度等于该部位笛卡尔空间自由度，夹爪为1维，仍返回关节空间位置；其他部位为7维，数据格式为xyz+四元数(qx, qy, qz, qw)。
eg: input:  [harm_names[0], effector_names[0]], chassis_frame_name
    output: [[x, y, z, qx, qy, qz, qw], [joint_q0]]
```

#### 获取机器人当前笛卡尔空间位置状态值——通过tf树
```
function name:      get_current_cartesian_pose_tf
input parameters:   target_frame:str, source_frame:str
output type:        list
function description: 通过tf树获取source_frame相对于target_frame的位姿。这个函数支持获取机器人任意两个部位之间的相对位姿,比如可以获取手臂末端相对于腰部坐标系的位置。
others: 输出为7维数据,格式为xyz+四元数(qx, qy, qz, qw)。target_frame和source_frame可以是机器人任意部位的坐标系。
eg: input:  "astribot_torso", "astribot_arm_left_link7" 
    output: [x, y, z, qx, qy, qz, qw]  # 左臂末端相对于腰部的位姿
```

#### 关于实时状态获取部分的额外说明
- 当前状态值和期望值的区别：期望值是当前给机器人下发的指令，状态值是机器人实际运动到的位置，期望值与状态值之间会有微小的误差，包括稳态误差和滞后，当进行回环控制时，应该使用当前期望值作为回环计算的输入，当要获取机器人当前精确位置时使用当前状态值。
- get_current_cartesian_pose 和 get_current_cartesian_pose_tf的区别： get_current_cartesian_pose仅支持底盘坐标系和世界坐标系，get_current_cartesian_pose_tf支持以任意机器人部位作为相对坐标系，比如可以求手臂末端相对于腰部坐标系的位置。
- 坐标系说明：世界系为机器人启动时底盘中心位置，z轴向上，x轴向前，符合右手系定义。底盘坐标系为底盘中心位置，z轴向上，x轴向前，符合右手系定义。在机器人刚启动时或底盘没有移动的情况下，世界系和底盘系是重合的。




## 运动学

#### 正运动学
```
function name:      get_forward_kinematics
input parameters:   names:list, joints_position:list
output type:        dict
function description: 输入一组names，和names中每个部位对应的关节位置组成的list——joints_position，输出在输入关节位置下机器人笛卡尔空间位置，该笛卡尔空间位置在底盘坐标系下。注意，需要至少输入腰部name和腰部对应的关节位置，因为无法在腰部位置缺失的情况下单独计算手臂或头部的正向运动学，另外，不支持夹爪name和关节位置输入，夹爪仅有一个关节，不涉及运动学的处理。
eg: input:  [torso_name[0], arm_names[0]], [[torso_q0, torso_q1, torso_q2, torso_q3], [arm_q0, arm_q1, arm_q2, arm_q3, arm_q4, arm_q5, arm_q6]]
    output: {'astribot_arm_left': [x, y, z, qx, qy, qz, qw],
             'astribot_torso': [x, y, z, qx, qy, qz, qw]}
```
#### 逆运动学
```
function name:      get_inverse_kinematics
input parameters:   names:list, cartesian_pose_list:list
output type:        dict
function description: 输入一组names，和names中每个部位对应的笛卡尔空间位置（底盘坐标系下）组成的list——cartesian_pose_list，输出在输入笛卡尔空间位置下机器人关节空间解。注意，目前只支持腰和双臂的输入。
eg: input:  [torso_name[0], arm_names[0]], [[x, y, z, qx, qy, qz, qw], [x, y, z, qx, qy, qz, qw]]
    output: {'astribot_arm_left': [arm_q0, arm_q1, arm_q2, arm_q3, arm_q4, arm_q5, arm_q6],
             'astribot_torso': [torso_q0, torso_q1, torso_q2, torso_q3]}
```





## 机器人控制

#### 机器人从当前位置运动到 home pose
```
function name:      move_to_home
input parameters:   duration=5.0, use_wbc:bool=False
output type:        string
function description: 机器人的所有部位在设定时间下从当前位置运动到设定好的home pose。
parameters description: duration是设定时间，单位为秒，默认为5秒。use_wbc是是否启用[WBC](#控制方式)的标志。
output description: 调用函数即开始运动，运动结束后返回一个字符串，仅用来说明运动结果。
```
#### 机器人关节位置从当前位置运动到设定点
```
function name:      move_joints_position
input parameters:   names:list, commands:list, duration=5.0, use_wbc:bool=False, add_default_torso:bool=True
output type:        string
function description: 机器人的指定部位在设定时间下从当前位置运动到输入的目标关节空间位置。
parameters description: names是一个包含指定部位name的list，用来说明需要运动机器人的哪些部位。commands是关节位置设定点，顺序需要和names一致。duration是设定时间，单位为秒，默认为5秒。use_wbc是是否启用wbc的标志，wbc的相关说明见本章末尾。add_default_torso用来决定是否添加默认腰部期望，相关说明见本章末尾。
output description: 调用函数即开始运动，运动结束后返回一个字符串，仅用来说明运动结果。
```
#### 机器人笛卡尔空间从当前位置运动到设定点
```
function name:      move_cartesian_pose
input parameters:   names:list, commands:list, duration=5.0, use_wbc:bool=True, frame:str="chassis", add_default_torso:bool=True
output type:        string
function description: 机器人的指定部位在设定时间下从当前位置运动到输入的目标笛卡尔空间位置（指定坐标系下）。
parameters description: names是一个包含指定部位name的list，用来说明需要运动机器人的哪些部位。commands是笛卡尔空间位置设定点，顺序需要和names一致。duration是设定时间，单位为秒，默认为5秒。use_wbc是是否启用wbc的标志，wbc的相关说明见本章末尾。frame是坐标系name，用来说明用户下发的commands是在哪个坐标系下的。add_default_torso用来决定是否添加默认腰部期望，相关说明见本章末尾。
output description: 调用函数即开始运动，运动结束后返回一个字符串，仅用来说明运动结果。
```
#### 机器人关节位置在线驱动
```
function name:      set_joints_position
input parameters:   names:list, position:list, control_way:str="filter" , use_wbc:bool=False, freq=None, add_default_torso:bool=True
output type:        None
function description: 机器人的指定部位实时地从当前位置运动到输入的目标关节空间位置。
parameters description: names是一个包含指定部位name的list，用来说明需要运动机器人的哪些部位。position是关节空间位置期望，顺序需要和names一致。control_way是期望下发的方式，可选输入为direct和filter，direct是直接下发，精度更高，但对用户输入要求高，要求没有突变，否则会因为速度跳变超限而报错；filter是滤波下发，下发的指令更平滑，不易发生跳变，但是可能存在滞后导致精度变差。use_wbc是是否启用wbc的标志，wbc的相关说明见本章末尾。freq是用户调用set_joints_position的函数，当且仅当control_way为direct，use_wbc=False时需要，当control_way为direct，use_wbc=False时是一种最直接的关节位置控制方式，将直接下发用户的命令不做任何额外处理，这就需要用户提供其调用的频率用来计算速度期望。add_default_torso用来决定是否添加默认腰部期望，相关说明见本章末尾。
```
#### 机器人关节速度在线驱动
```
function name:      set_joints_velocity
input parameters:   names:list, velocity:list
output type:        None
function description: 机器人的指定部位的所有关节实时地按照期望速度运行。
parameters description: names是一个包含指定部位name的list，用来说明需要运动机器人的哪些部位。velocity是关节空间期望速度，顺序需要和names一致。
```
#### 机器人关节力矩在线驱动
```
function name:      set_joints_torque
input parameters:   names:list, torque:list
output type:        None
function description: 机器人的指定部位的所有关节实时地按照期望力矩运行。
parameters description: names是一个包含指定部位name的list，用来说明需要运动机器人的哪些部位。torque是关节空间期望力矩，顺序需要和names一致。
```
#### 机器人笛卡尔空间位置在线驱动
```
function name:      set_cartesian_pose
input parameters:   names:list, cartesian_pose:list, control_way:str="filter", use_wbc:bool=False, frame:str="chassis", add_default_torso:bool=True
output type:        None
function description: 机器人的指定部位实时地从当前位置运动到输入的目标笛卡尔空间位置（指定坐标系下）。
parameters description: names是一个包含指定部位name的list，用来说明需要运动机器人的哪些部位。cartesian_pose是笛卡尔空间位置期望，顺序需要和names一致。control_way是期望下发的方式，可选输入为direct和filter，direct是直接下发，精度更高，但对用户输入要求高，要求没有突变，否则会因为速度跳变超限而报错；filter是滤波下发，下发的指令更平滑，不易发生跳变，但是可能存在滞后导致精度变差。use_wbc是是否启用wbc的标志，wbc的相关说明见本章末尾。frame是坐标系name，用来说明用户下发的cartesian_pose是在哪个坐标系下的。add_default_torso用来决定是否添加默认腰部期望，相关说明见本章末尾。
```
#### 机器人关节空间多点跟随
```
function name:      move_joints_waypoints
input parameters:   names:list, waypoints:list, time_list:list, use_wbc:bool=False, joy_controller=False, add_default_torso:bool=True
output type:        string
function description: 机器人的指定部位在设定时间下从当前位置依次运动到输入的目标关节空间路径点。
parameters description: names是一个包含指定部位name的list，用来说明需要运动机器人的哪些部位。waypoints是一系列关节位置设定点，是一个三维list，每一个waypoint内的顺序需要和names一致。time_list是一个由时间参数组成的list，维度和waypoints数量一致，用来指定到达每一个点的时间，单位为秒，是从0开始的绝对时间。use_wbc是是否启用wbc的标志，wbc的相关说明见本章末尾。joy_controller用来决定是否启用手柄的播放控制，相关说明见本章末尾。add_default_torso用来决定是否添加默认腰部期望，相关说明见本章末尾。
output description: 调用函数即开始运动，运动结束后返回一个字符串，仅用来说明运动结果。
```
#### 机器人笛卡尔空间多点跟随
```
function name:      move_cartesian_waypoints
input parameters:   names:list, waypoints:list, time_list:list, use_wbc:bool=True, joy_controller=False, frame:str="chassis", add_default_torso:bool=True
output type:        string
function description: 机器人的指定部位在设定时间下从当前位置依次运动到输入的目标笛卡尔空间路径点（指定坐标系下）。
parameters description: names是一个包含指定部位name的list，用来说明需要运动机器人的哪些部位。waypoints是一系列笛卡尔位置设定点，是一个三维list，每一个waypoint内的顺序需要和names一致。time_list是一个由时间参数组成的list，维度和waypoints数量一致，用来指定到达每一个点的时间，单位为秒，是从0开始的绝对时间。use_wbc是是否启用wbc的标志，wbc的相关说明见本章末尾。joy_controller用来决定是否启用手柄的播放控制，相关说明见本章末尾。frame是坐标系name，用来说明用户下发的waypoints是在哪个坐标系下的。add_default_torso用来决定是否添加默认腰部期望，相关说明见本章末尾。
output description: 调用函数即开始运动，运动结束后返回一个字符串，仅用来说明运动结果。
```
#### 闭合执行器
```
function name:      close_effector
input parameters:   names:list=None
output type:        string
function description: 机器人闭合指定执行器。
parameters description: names是一个包含指定部位name的list，仅支持接收执行器部位的name，默认值下会闭合所有执行器。
output description: 调用函数即开始运动，运动结束后返回一个字符串，仅用来说明运动结果。
```
#### 张开执行器
```
function name:      open_effector
input parameters:   names:list=None
output type:        string
function description: 机器人张开指定执行器。
parameters description: names是一个包含指定部位name的list，仅支持接收执行器部位的name，默认值下会张开所有执行器。
output description: 调用函数即开始运动，运动结束后返回一个字符串，仅用来说明运动结果。
```
#### 关于机器人控制部分的额外说明
- move和set的区别：move是阻塞运行的，设定好机器人的到达时间，机器人会在规定时间内运动过去，用户只关注目标点和时间，起点和终点的速度都是0。set是在线驱动，用户应该循环调用set函数按一定频率下发期望，机器人在每个循环周期内会跟随目标，set驱动是在线的、实时的、非阻塞的。
- wbc功能说明：如果在关节空间控制下未启用wbc，则是每个关节各自独立控制运动到目标位置；如果关节空间控制下启用了wbc，关节位置可能并不会保证到目标关节位置，而是到达目标关节构型对应的笛卡尔空间位置。wbc是一种全身控制，它会综合地考量所有部位的控制目标，如果更关注臂的精度（如仅下发了臂的命令）则会通过腰的补偿来让臂达到目标位置。
- add_default_torso功能说明：当给定了双臂/单臂位姿期望但没有给腰部位姿期望，且打开了wbc时，为了追踪臂的位姿，wbc有可能解算出腰部产生一个较大幅度的运动。但很多时候这可能并不是用户想要的，用户很可能只给了臂的位姿期望，想用wbc，但又不想腰部运动。这时候，将add_default_torso置True，会自动补全一个腰部期望，让腰在缺少用户输入期望的情况下也不会大幅运动。
- joy_controller功能说明：在多点跟随（路径跟随）中，如果joy_controller置True，则会启动手柄控制播放的功能，按住手柄RT按键则前向运动，按住手柄LT按键反向运动。


## 功能模块

#### 设置滤波参数
```
function name:      set_filter_parameters
input parameters:   filter_scale, gripper_filter_scale
output type:        string
function description: 在通过set_函数控制机器人实时运动时，可以设置control_way="filter"，对用户下发的命令进行滤波，内部会有一个滤波参数，如果用户想自己设置也可以通过该函数进行设置，执行器和非执行器部位可以设置不同的滤波参数。
parameters description: filter_scale是非执行器部分的滤波参数，gripper_filter_scale是执行器的滤波参数，滤波参数的设置范围是0到1，越接近0滤波越狠，越接近1越趋同于不滤波。
output description: 返回一个字符串，仅用来说明设置成功或失败。
```
#### 机器人头部跟踪双手
```
function name:      set_head_follow_effector
input parameters:   enable:bool=True
output type:        string
function description: 当调用函数输入True时，即开启头部跟踪双臂执行器的功能，头部将不响应其他控制指令，这个功能是为了机器人在执行操作任务时头部相机能一直观察到操作场景。当调用函数输入False时，关闭头部跟踪双臂执行器的功能。
output description: 返回一个字符串，仅用来说明设置成功或失败。
```
#### 机器人自避障
```
function name:      set_wbc_collision_avoidance
input parameters:   enable:bool=True
output type:        string
function description: 当调用函数输入True时，即开启机器人[自避障](#其他概念)功能，仅在[WBC](#控制方式)模式下有效，机器人在运动过程中如出现可以发生自碰撞的情况，则会约束控制指令，让机器人不会发生自碰撞。当调用函数输入False时，关闭机器人自避障功能。
output description: 返回一个字符串，仅用来说明设置成功或失败。
```
#### 获取机器人自身各模块最近距离
```
function name:      get_self_closest_point
input parameters:   None
output type:        min_distance:float32, link_A_name:string, link_B_name:string, closest_point_on_A_of_torso_frame:list, closest_point_on_B_of_torso_frame:list
function description: 仅当使用了wbc控制机器人运动后有有效返回值，调用函数即返回当前状态下机器人自身最近的两点。
output description: 返回机器人自身最近的两点的距离min_distance，返回两点所在部位的name，link_A_name和link_B_name，返回这两个点在腰部坐标系下的位置，closest_point_on_A_of_torso_frame和closest_point_on_B_of_torso_frame。
```
#### 启动相机
```
function name:      activate_camera
input parameters:   cameras_info:dict=None
output type:        string
function description: 开启相机，调用该函数后，后续可以通过get_images_dict获取实时图像。
parameters description: 输入一个字典，来说明启动哪些相机。
{
    'left_D405': {'flag_getdepth': True, 'flag_getIR': True},
    'right_D405': {'flag_getdepth': True, 'flag_getIR': True},
    'Gemini335': {'flag_getdepth': True, 'flag_getIR': True},
    'Bolt': {'flag_getdepth': True},
    'Stereo': {}
}
这是全部相机输入的示例，当一个name存在字典里，则启动该相机的rgb图像，如果一个name下属字典里flag_getdepth为True，则启动该相机的depth图像，如果一个name下属字典里flag_getIR为True，则启动该相机的IR图像。left_D405，right_D405，Gemini335三种图像都有，Bolt只有rgb和depth图像，Stereo只有rgb。
output description: 返回一个字符串，仅用来说明启动成功或失败。
```
#### 获取图像
```
function name:      get_images_dict
input parameters:   None
output type:        dict, dict, dict
function description: 当相机启动后，可以通过该函数获取相机实时图像。
output description: 返回三个字典，分别是rgb图像的字典，depth图像的字典，ir图像的字典，字典内的name是相机的name，即left_D405，right_D405，Gemini335，Bolt，Stereo。
```


## 其他

#### 软急停
```
function name:      stop_robot
input parameters:   None
output type:        string
function description: 调用后，机器人立刻停止在当前位置，并且不再响应任何运动指令。
output description: 返回一个字符串，仅用来说明停止成功或失败。
```
#### 软急停恢复
```
function name:      restart_robot
input parameters:   None
output type:        string
function description: 调用后，机器人从软急停状态退出，重新开始响应用户的运动指令。
output description: 返回一个字符串，仅用来说明软急停恢复成功或失败。
```

## 附录

### 附录A：专业术语解释

#### 坐标系统
- **世界坐标系(world frame)**: 固定在机器人启动时底盘中心位置的绝对坐标系,z轴向上,x轴向前,符合右手系定义。
- **底盘坐标系(chassis frame)**: 固定在机器人底盘中心的相对坐标系,z轴向上,x轴向前,符合右手系定义。机器人启动时,世界坐标系与底盘坐标系重合。

#### 空间表示
- **关节空间(joint space)**: 用关节角度来描述机器人位置的空间。例如手臂有7个关节,则用7个角度值来表示手臂的位置。
- **笛卡尔空间(cartesian space)**: 用位置(x,y,z)和姿态(四元数)来描述机器人末端在三维空间中的位置和朝向。

#### 控制方式
- **WBC(Whole Body Control)**: 全身控制,是一种考虑机器人整体运动的控制方法。它会协调所有关节的运动来实现目标,比如可以通过调整腰部来帮助手臂达到目标位置。
- **滤波控制(filter control)**: 对用户下发的控制指令进行平滑处理的控制方式,可以避免机器人运动的突变。
- **直接控制(direct control)**: 直接执行用户下发的控制指令,精度更高但要求输入指令平滑。

#### 其他概念
- **自由度(DOF, Degree of Freedom)**: 描述机器人各个部位可以独立运动的数量。例如一个手臂有7个关节,就说它有7个自由度。
- **执行器(effector)**: 机器人末端的执行机构,如夹爪等。
- **自避障**: 机器人通过检测自身各部件之间的距离,自动避免发生自碰撞的功能。

## 附录B：FAQ

### 1. 如何选择合适的控制方式?
Q: 什么时候该用关节空间控制,什么时候该用笛卡尔空间控制?
A: 当你关注机器人具体关节角度时使用关节空间控制;当你关注末端执行器的位置和姿态时使用笛卡尔空间控制。

### 2. WBC控制相关问题
Q: 什么情况下应该启用WBC?
A: 当你需要协调多个部位配合运动,或需要通过腰部辅助手臂达到目标位置时,应该启用WBC。

### 3. 常见错误处理
Q: 为什么机器人突然停止响应命令?
A: 可能触发了软急停。检查是否:
- 运动指令超出限制
- 检测到潜在的自碰撞
- 手动触发了stop_robot()

