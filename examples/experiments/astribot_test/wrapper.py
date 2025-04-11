from typing import OrderedDict
from franka_env.camera.rs_capture import RSCapture
from franka_env.camera.video_capture import VideoCapture
from franka_env.utils.rotations import euler_2_quat
import numpy as np
import requests
import copy
import gymnasium as gym
import time
from astribot_env.envs.astribot_env import AstribotEnv

class AstribotTestEnv(AstribotEnv):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def init_cameras(self, name_serial_dict=None):
        """Init both wrist cameras."""
        if self.cap is not None:  # close cameras if they are already open
            self.close_cameras()

        self.cap = OrderedDict()
        for cam_name, kwargs in name_serial_dict.items():
            if cam_name == "side_classifier":
                self.cap["side_classifier"] = self.cap["side_policy"]
            else:
                cap = VideoCapture(
                    RSCapture(name=cam_name, **kwargs)
                )
                self.cap[cam_name] = cap
    
    # gripper开---reset pose----返回obs
    def reset(self, **kwargs):
        self._recover()
        self._send_gripper_command(1.0)

        obs, info = super().reset(**kwargs)
        time.sleep(1)
        self.success = False
        self._update_currpos()
        obs = self._get_obs()
        return obs, info


class GripperPenaltyWrapper(gym.Wrapper):
    def __init__(self, env, penalty=-0.05):
        super().__init__(env)
        assert env.action_space.shape == (7,)
        self.penalty = penalty
        self.last_gripper_pos = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.last_gripper_pos = obs["state"][0, 0]
        return obs, info

    def step(self, action):
        """Modifies the :attr:`env` :meth:`step` reward using :meth:`self.reward`."""
        observation, reward, terminated, truncated, info = self.env.step(action)
        if "intervene_action" in info:
            action = info["intervene_action"]

        if (action[-1] < -0.5 and self.last_gripper_pos > 0.9) or (
            action[-1] > 0.5 and self.last_gripper_pos < 0.9
        ):
            info["grasp_penalty"] = self.penalty
        else:
            info["grasp_penalty"] = 0.0

        # TODO check the gripper pose index in state matrix
        self.last_gripper_pos = observation["state"][0, 0]
        return observation, reward, terminated, truncated, info
