"""Gym Interface for Franka"""
import os
import numpy as np
import gymnasium as gym
import cv2
import copy
from scipy.spatial.transform import Rotation
import time
import queue
import threading
from datetime import datetime
from collections import OrderedDict
from typing import Dict

from franka_env.camera.video_capture import VideoCapture
from franka_env.camera.rs_capture import RSCapture
from franka_env.utils.rotations import euler_2_quat, quat_2_euler

from core.sdk_client.astribot_client import Astribot


def print_green(x):
    return print("\033[92m {}\033[00m".format(x))


def print_yellow(x):
    return print("\033[93m {}\033[00m".format(x))

class ImageDisplayer(threading.Thread):
    def __init__(self, queue, name):
        threading.Thread.__init__(self)
        self.queue = queue
        self.daemon = True  # make this a daemon thread
        self.name = name

    def run(self):
        while True:
            img_array = self.queue.get()  # retrieve an image from the queue
            if img_array is None:  # None is our signal to exit
                break

            frame = np.concatenate(
                [cv2.resize(v, (128, 128)) for k, v in img_array.items() if "full" not in k], axis=1
            )

            cv2.imshow(self.name, frame)
            cv2.waitKey(1)


##############################################################################


class DefaultEnvConfig:
    """Default configuration for AstribotEnv. Fill in the values below."""
    REALSENSE_CAMERAS: Dict = {
        "wrist": "000000000000",
    }
    IMAGE_CROP: dict[str, callable] = {}
    TARGET_POSE: np.ndarray = np.zeros((6,))
    REWARD_THRESHOLD: np.ndarray = np.zeros((6,))
    ACTION_SCALE = np.zeros((3,))
    RESET_POSE = np.zeros((6,))
    RANDOM_RESET = False
    RANDOM_XY_RANGE = (0.0,)
    RANDOM_RZ_RANGE = (0.0,)
    ABS_POSE_LIMIT_HIGH = np.zeros((6,))
    ABS_POSE_LIMIT_LOW = np.zeros((6,))
    DISPLAY_IMAGE: bool = True
    MAX_EPISODE_LENGTH: int = 100
    JOINT_RESET_PERIOD: int = 0
    JOINT_RESET_POSITION: np.ndarray = np.array([-0.05389, -0.55, 1.26345, 1.61261, 0.5537, 0.21308, -0.313582]) # right arm
    EE_TARGET_POSE_FILTER_PARAM: float = 0.01

    RL_ACTION_WEIGHT: float = 1.0


##############################################################################


class CartesianPoseCtrlThread(threading.Thread):
    def __init__(self, astribot):
        super().__init__()
        self.astribot = astribot
        self.pose_command = None  # 用于存储位置命令
        self.running = False  # 控制线程运行的开关
        self.terminate = False
        self.lock = threading.Lock()  # 用于线程安全

    def run(self):
        """线程主循环，持续监听 pose_command 并调用 set_cartesian_pose。"""

        names = [self.astribot.arm_names[1]]  # right arm for now
        while not self.terminate:
            if not self.running:
                time.sleep(0.1)  # 如果线程未运行，稍作休眠
                continue

            if self.pose_command is not None:
                with self.lock:
                    command_list = self.pose_command  # 获取命令

                # 调用 set_cartesian_pose
                self.astribot.set_cartesian_pose(
                    names, command_list, control_way="filter", use_wbc=True
                )

            time.sleep(1.0 / 250.0)  # default astribot_sdk frequency as 250Hz

    def update_command(self, ee_command_list):
        """更新位置命令。"""
        assert len(ee_command_list[0]) == 7 
        with self.lock:
            self.pose_command = copy.deepcopy(ee_command_list)

    def start_running(self):
        """启动线程运行。"""
        self.running = True

    def stop_running(self):
        """停止线程运行。"""
        self.running = False
    
    def terminate_thread(self):
        self.terminate = True

##############################################################################

class AstribotEnv(gym.Env):
    def __init__(
        self,
        hz=10,
        fake_env=False,
        save_video=False,
        config: DefaultEnvConfig = None,
        bc_action_as_obs = False,
    ):
        # TODO{zengxw}: Only enable the right arm for now.
        self.astribot = Astribot()
        self.astribot.set_filter_parameters(config.EE_TARGET_POSE_FILTER_PARAM, 1)

        self.pose_thread = CartesianPoseCtrlThread(self.astribot)
        self.pose_thread.start()

        self.action_scale = config.ACTION_SCALE
        self._TARGET_POSE = config.TARGET_POSE
        self._RESET_POSE = config.RESET_POSE
        self._REWARD_THRESHOLD = config.REWARD_THRESHOLD
        self.config = config
        self.max_episode_length = config.MAX_EPISODE_LENGTH
        self.display_image = config.DISPLAY_IMAGE
        self.bc_action_as_obs = bc_action_as_obs

        # convert last 3 elements from euler to quat, from size (6,) to (7,)
        self.resetpos = np.concatenate(
            [config.RESET_POSE[:3], euler_2_quat(config.RESET_POSE[3:])]
        )
        self._update_currpos()
        self.randomreset = config.RANDOM_RESET
        self.random_xy_range = config.RANDOM_XY_RANGE
        self.random_rz_range = config.RANDOM_RZ_RANGE
        self.hz = hz
        self.joint_reset_cycle = config.JOINT_RESET_PERIOD  # reset the robot joint every 200 cycles

        self.save_video = save_video
        if self.save_video:
            print("Saving videos!")
            self.recording_frames = []

        # boundary box
        self.xyz_bounding_box = gym.spaces.Box(
            config.ABS_POSE_LIMIT_LOW[:3],
            config.ABS_POSE_LIMIT_HIGH[:3],
            dtype=np.float64,
        )
        self.rpy_bounding_box = gym.spaces.Box(
            config.ABS_POSE_LIMIT_LOW[3:],
            config.ABS_POSE_LIMIT_HIGH[3:],
            dtype=np.float64,
        )
        # Action/Observation Space
        self.action_space = gym.spaces.Box(
            np.ones((7,), dtype=np.float32) * -1,
            np.ones((7,), dtype=np.float32),
        )

        # action offset like bc reference
        self.bc_action_in_ee = None
        self.bc_action_in_base = None

        self.rl_action_weight = config.RL_ACTION_WEIGHT

        if bc_action_as_obs:
            self.observation_space = gym.spaces.Dict(
                {
                    "state": gym.spaces.Dict(
                        {
                            "tcp_pose": gym.spaces.Box(
                                -np.inf, np.inf, shape=(7,)
                            ),  # xyz + quat
                            "gripper_pose": gym.spaces.Box(-1, 1, shape=(1,)),
                            "BC_action": gym.spaces.Box(-1, 1, shape=(7,)), # delta xyz rpy and gripper
                        }
                    ),
                    "images": gym.spaces.Dict(
                        {key: gym.spaces.Box(0, 255, shape=(128, 128, 3), dtype=np.uint8) 
                                    for key in config.REALSENSE_CAMERAS}
                    ),
                }
            )
        else:
            self.observation_space = gym.spaces.Dict(
                {
                    "state": gym.spaces.Dict(
                        {
                            "tcp_pose": gym.spaces.Box(
                                -np.inf, np.inf, shape=(7,)
                            ),  # xyz + quat
                            "gripper_pose": gym.spaces.Box(-1, 1, shape=(1,)),
                        }
                    ),
                    "images": gym.spaces.Dict(
                        {key: gym.spaces.Box(0, 255, shape=(128, 128, 3), dtype=np.uint8) 
                                    for key in config.REALSENSE_CAMERAS}
                    ),
                }
            )
        self.cycle_count = 0

        if fake_env:
            return

        self.cap = None
        self.init_cameras(config.REALSENSE_CAMERAS)
        if self.display_image:
            self.img_queue = queue.Queue()
            self.displayer = ImageDisplayer(self.img_queue, "astribot_SERL")
            self.displayer.start()

        if not fake_env:
            from pynput import keyboard
            self.terminate = False
            def on_press(key):
                if key == keyboard.Key.esc:
                    self.terminate = True
            self.listener = keyboard.Listener(on_press=on_press)
            self.listener.start()

        print("Initialized AstribotEnv")

    def clip_safety_box(self, pose: np.ndarray) -> np.ndarray:
        """Clip the pose to be within the safety box."""
        pose[:3] = np.clip(
            pose[:3], self.xyz_bounding_box.low, self.xyz_bounding_box.high
        )
        euler = Rotation.from_quat(pose[3:]).as_euler("xyz")

        # Clip first euler angle separately due to discontinuity from pi to -pi
        sign = np.sign(euler[0])
        euler[0] = sign * (
            np.clip(
                np.abs(euler[0]),
                self.rpy_bounding_box.low[0],
                self.rpy_bounding_box.high[0],
            )
        )

        euler[1:] = np.clip(
            euler[1:], self.rpy_bounding_box.low[1:], self.rpy_bounding_box.high[1:]
        )
        pose[3:] = Rotation.from_euler("xyz", euler).as_quat()

        return pose

    def step(self, action: np.ndarray) -> tuple:
        """standard gym step function."""
        start_time = time.time()
        action = np.clip(action, self.action_space.low, self.action_space.high)
        xyz_delta = action[:3]
        self.nextpos = self.currpos.copy()
        self.nextpos[:3] = self.nextpos[:3] + xyz_delta * self.action_scale[0]

        # GET ORIENTATION FROM ACTION
        self.nextpos[3:] = (
            Rotation.from_euler("xyz", action[3:6] * self.action_scale[1])
            * Rotation.from_quat(self.currpos[3:])
        ).as_quat()

        gripper_action = action[6] * self.action_scale[2]

        self._send_gripper_command(gripper_action)
        self._send_pos_command(self.clip_safety_box(self.nextpos))

        self.curr_path_length += 1
        dt = time.time() - start_time
        time.sleep(max(0, (1.0 / self.hz) - dt))

        self._update_currpos()
        ob = self._get_obs()
        reward = self.compute_reward(ob)
        done = self.curr_path_length >= self.max_episode_length or reward or self.terminate
        if done:
            if reward:
                print_green("Step success!")
            if self.curr_path_length >= self.max_episode_length:
                print_yellow("Step terminated: max episode length reached!")
        return ob, int(reward), done, False, {"succeed": reward}

    def compute_reward(self, obs) -> bool:
        current_pose = obs["state"]["tcp_pose"]
        # convert from quat to euler first
        current_rot = Rotation.from_quat(current_pose[3:]).as_matrix()
        target_rot = Rotation.from_euler("xyz", self._TARGET_POSE[3:]).as_matrix()
        diff_rot = current_rot.T  @ target_rot
        diff_euler = Rotation.from_matrix(diff_rot).as_euler("xyz")
        delta = np.abs(np.hstack([current_pose[:3] - self._TARGET_POSE[:3], diff_euler]))
        # print(f"Delta: {delta}")
        if np.all(delta < self._REWARD_THRESHOLD) and self.curr_gripper_pos < 0.85:
            return True
        else:
            # print(f'Goal not reached, the difference is {delta}, the desired threshold is {self._REWARD_THRESHOLD}')
            return False

    def get_im(self) -> Dict[str, np.ndarray]:
        """Get images from the realsense cameras."""
        images = {}
        display_images = {}
        full_res_images = {}  # New dictionary to store full resolution cropped images
        for key, cap in self.cap.items():
            try:
                rgb = cap.read()
                cropped_rgb = self.config.IMAGE_CROP[key](rgb) if key in self.config.IMAGE_CROP else rgb
                resized = cv2.resize(
                    cropped_rgb, self.observation_space["images"][key].shape[:2][::-1]
                )
                images[key] = resized[..., ::-1]
                display_images[key] = resized
                display_images[key + "_full"] = cropped_rgb
                full_res_images[key] = copy.deepcopy(cropped_rgb)  # Store the full resolution cropped image
            except queue.Empty:
                input(
                    f"{key} camera frozen. Check connect, then press enter to relaunch..."
                )
                cap.close()
                self.init_cameras(self.config.REALSENSE_CAMERAS)
                return self.get_im()

        # Store full resolution cropped images separately
        if self.save_video:
            self.recording_frames.append(full_res_images)

        if self.display_image:
            self.img_queue.put(display_images)
        return images

    def interpolate_move(self, goal: np.ndarray, timeout: float):
        """Move the robot to the goal position. Blocking call."""

        self.pose_thread.stop_running()


        names = [self.astribot.arm_names[1]]
        if goal.shape == (6,):
            goal = np.concatenate([goal[:3], euler_2_quat(goal[3:])])
        command_list = [goal.tolist()]
        response = self.astribot.move_cartesian_pose(names, command_list, duration=timeout, use_wbc=True)
        self.nextpos = copy.deepcopy(goal)
        self._update_currpos()

    def go_to_reset(self, joint_reset=False):
        """
        The concrete steps to perform reset should be
        implemented each subclass for the specific task.
        Should override this method if custom reset procedure is needed.
        """

        self.pose_thread.stop_running()

        # Perform joint reset if needed
        if joint_reset:
            names = [self.astribot.arm_names[1]]
            command_list = [self.config.JOINT_RESET_POSITION.tolist()]
            response = self.astribot.move_joints_position(names, command_list, duration=1.0, use_wbc=True)

        # Perform Carteasian reset
        if self.randomreset:  # randomize reset position in xy plane
            reset_pose = self.resetpos.copy()
            reset_pose[:2] += np.random.uniform(
                -self.random_xy_range, self.random_xy_range, (2,)
            )
            euler_random = self._RESET_POSE[3:].copy()
            euler_random[-1] += np.random.uniform(
                -self.random_rz_range, self.random_rz_range
            )
            reset_pose[3:] = euler_2_quat(euler_random)
            self.interpolate_move(reset_pose, timeout=1)
        else:
            reset_pose = self.resetpos.copy()
            self.interpolate_move(reset_pose, timeout=1)

    def reset(self, joint_reset=True, **kwargs):
        if self.save_video:
            self.save_video_recording()

        self.cycle_count += 1
        if self.joint_reset_cycle!=0 and self.cycle_count % self.joint_reset_cycle == 0:
            self.cycle_count = 0
            joint_reset = True

        self._recover()
        self.go_to_reset(joint_reset=joint_reset)
        self._recover()
        self.curr_path_length = 0

        self._update_currpos()
        obs = self._get_obs()
        self.terminate = False
        return obs, {"succeed": False}

    def save_video_recording(self):
        try:
            if len(self.recording_frames):
                if not os.path.exists('./videos'):
                    os.makedirs('./videos')
                
                timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                
                for camera_key in self.recording_frames[0].keys():
                    video_path = f'./videos/left_{camera_key}_{timestamp}.mp4'
                    
                    # Get the shape of the first frame for this camera
                    first_frame = self.recording_frames[0][camera_key]
                    height, width = first_frame.shape[:2]
                    
                    video_writer = cv2.VideoWriter(
                        video_path,
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        10,
                        (width, height),
                    )
                    
                    for frame_dict in self.recording_frames:
                        video_writer.write(frame_dict[camera_key])
                    
                    video_writer.release()
                    print(f"Saved video for camera {camera_key} at {video_path}")
                
            self.recording_frames.clear()
        except Exception as e:
            print(f"Failed to save video: {e}")

    def init_cameras(self, name_serial_dict=None):
        """Init both wrist cameras."""
        if self.cap is not None:  # close cameras if they are already open
            self.close_cameras()

        self.cap = OrderedDict()
        for cam_name, kwargs in name_serial_dict.items():
            cap = VideoCapture(
                RSCapture(name=cam_name, **kwargs)
            )
            self.cap[cam_name] = cap

    def close_cameras(self):
        """Close both wrist cameras."""
        try:
            for cap in self.cap.values():
                cap.close()
        except Exception as e:
            print(f"Failed to close cameras: {e}")

    def _recover(self):
        """Internal function to recover the robot from error state."""
        pass

    def _send_pos_command(self, pos: np.ndarray):
        """Internal function to send position command to the robot.
        NOTE This method is designed as a Non-blocking call.
        """
        self.pose_thread.start_running()

        self._recover()
        command_list = [pos.tolist()]
        self.pose_thread.update_command(command_list)

    def _send_gripper_command(self, pos: float, mode="binary"):
        """Internal function to send gripper command to the robot."""
        if mode == "binary":
            if (pos <= -0.5) and (self.curr_gripper_pos > 0.85):  # close gripper
                self.astribot.close_effector(names = [self.astribot.effector_right_name])
            elif (pos >= 0.5) and (self.curr_gripper_pos < 0.85):  # open gripper
                self.astribot.open_effector(names = [self.astribot.effector_right_name])
            else: 
                return
        elif mode == "continuous":
            raise NotImplementedError("Continuous gripper control is optional")

    def _update_currpos(self):
        """
        Internal function to get the latest state of the robot and its gripper.
        """
        current_cartesian_pose = self.astribot.get_current_cartesian_pose(frame=self.astribot.world_frame_name)
        self.currpos = np.array(current_cartesian_pose[4])

        raw_gripper_pos = np.array(current_cartesian_pose[5])
        # raw gripper pos 0: open and 100: close, mapping to 1: open and -1: close
        self.curr_gripper_pos = np.clip(1 - (raw_gripper_pos / 100.0) * 2, -1, 1)

    def _get_obs(self) -> dict:
        images = self.get_im()
        if self.bc_action_in_ee is None:
            bc_action = np.zeros((7,))
        else:
            bc_action = self.bc_action_in_ee
        if self.bc_action_as_obs:
            state_observation = {
                "tcp_pose": self.currpos,
                "gripper_pose": self.curr_gripper_pos,
                "BC_action": bc_action,
            }
        else:
            state_observation = {
                "tcp_pose": self.currpos,
                "gripper_pose": self.curr_gripper_pos,
            }
        return copy.deepcopy(dict(images=images, state=state_observation))

    def close(self):
        print("Closing AstribotEnv...")
        if hasattr(self, 'listener'):
            self.listener.stop()
        self.close_cameras()
        self.pose_thread.stop_running()
        self.pose_thread.terminate_thread()
        self.pose_thread.join()
        print("Thread terminated")
        if self.display_image:
            self.img_queue.put(None)
            cv2.destroyAllWindows()
            self.displayer.join()
