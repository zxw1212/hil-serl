import time

import mujoco
import mujoco.viewer
import numpy as np

from franka_sim import envs
import gymnasium as gym
# import gym

# import joystick wrapper
from franka_env.envs.wrappers import JoystickIntervention

# env = envs.PandaPickCubeGymEnv(render_mode="human", image_obs=True)
env = gym.make("PandaPickCubeVision-v0", render_mode="human", image_obs=True)
env = JoystickIntervention(env)

env.reset()
m = env.unwrapped.model
d = env.unwrapped.data
# intervene on position control
with mujoco.viewer.launch_passive(model=m, data=d, show_left_ui=False, show_right_ui=False) as viewer_1, \
     mujoco.viewer.launch_passive(model=m, data=d, show_left_ui=False, show_right_ui=False) as viewer_2:
    for i in range(100000):
        env.step(np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
        viewer_1.sync()
        viewer_2.sync()
