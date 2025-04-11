import os
import jax
import numpy as np
import jax.numpy as jnp

from franka_env.envs.wrappers import (
    Quat2EulerWrapper,
    SpacemouseIntervention,
    MultiCameraBinaryRewardClassifierWrapper,
)
from franka_env.envs.relative_env import RelativeFrame
from franka_env.envs.franka_env import DefaultEnvConfig
from serl_launcher.wrappers.serl_obs_wrappers import SERLObsWrapper
from serl_launcher.wrappers.chunking import ChunkingWrapper
from serl_launcher.networks.reward_classifier import load_classifier_func

from experiments.config import DefaultTrainingConfig
from experiments.banana_pick_place.wrapper import BananaEnv, GripperPenaltyWrapper


class EnvConfig(DefaultEnvConfig):
    SERVER_URL: str = "http://127.0.0.1:5000/"
    REALSENSE_CAMERAS = {
        "wrist_1": {
            "serial_number": "230322272297",
            "dim": (1280, 720),
            "exposure": 17000,
        },
        "wrist_2": {
            "serial_number": "230322276360",
            "dim": (1280, 720),
            "exposure": 17000,
        },
    }
    IMAGE_CROP = {"wrist_1": lambda img: img[0:-1, 0:-1],
                  "wrist_2": lambda img: img[0:-1, 0:-1],
                  }
    TARGET_POSE = np.array([0.5860,0.05658,0.24477, 3.14, 0.0, 1.57])
    RESET_POSE = TARGET_POSE + np.array([-0.20, 0, 0.05, 0, 0, 0])
    ACTION_SCALE = np.array([0.04, 0.18, 1])
    RANDOM_RESET = False
    DISPLAY_IMAGE = True
    RANDOM_XY_RANGE = 0.01
    RANDOM_RZ_RANGE = 0.1
    ABS_POSE_LIMIT_HIGH = TARGET_POSE + np.array([0.05, 0.08, 0.08, 0.1, 0.1, 1.57])
    ABS_POSE_LIMIT_LOW = TARGET_POSE - np.array([0.25, 0.08, 0.08, 0.1, 0.1, 1.57])
    COMPLIANCE_PARAM = {
        "translational_stiffness": 2500,
        "translational_damping": 89,
        "rotational_stiffness": 200,
        "rotational_damping": 7,
        "translational_Ki": 0.0,
        "translational_clip_x": 0.02,
        "translational_clip_y": 0.02,
        "translational_clip_z": 0.02,
        "translational_clip_neg_x": 0.02,
        "translational_clip_neg_y": 0.02,
        "translational_clip_neg_z": 0.02,
        "rotational_clip_x": 0.03,
        "rotational_clip_y": 0.03,
        "rotational_clip_z": 0.03,
        "rotational_clip_neg_x": 0.03,
        "rotational_clip_neg_y": 0.03,
        "rotational_clip_neg_z": 0.03,
        "rotational_Ki": 0.0,
    }
    PRECISION_PARAM = {
        "translational_stiffness": 2500,
        "translational_damping": 89,
        "rotational_stiffness": 200,
        "rotational_damping": 7,
        "translational_Ki": 0.0,
        "translational_clip_x": 0.02,
        "translational_clip_y": 0.02,
        "translational_clip_z": 0.02,
        "translational_clip_neg_x": 0.02,
        "translational_clip_neg_y": 0.02,
        "translational_clip_neg_z": 0.02,
        "rotational_clip_x": 0.03,
        "rotational_clip_y": 0.03,
        "rotational_clip_z": 0.03,
        "rotational_clip_neg_x": 0.03,
        "rotational_clip_neg_y": 0.03,
        "rotational_clip_neg_z": 0.03,
        "rotational_Ki": 0.0,
    }
    MAX_EPISODE_LENGTH = 240


class TrainConfig(DefaultTrainingConfig):
    image_keys = ["wrist_1", "wrist_2"]
    classifier_keys = ["wrist_1", "wrist_2"]
    proprio_keys = ["tcp_pose", "tcp_vel", "gripper_pose", "BC_action"]
    checkpoint_period = 500
    cta_ratio = 2
    batch_size = 256
    random_steps = 0
    discount = 0.98
    buffer_period = 500
    encoder_type = "resnet-pretrained"
    setup_mode = "single-arm-learned-gripper"

    def get_environment(self, fake_env=False, save_video=False, classifier=False):
        env = BananaEnv(
            fake_env=fake_env, save_video=save_video, config=EnvConfig(), bc_action_as_obs=False
        )
        if not fake_env:
            env = SpacemouseIntervention(env)
        env = RelativeFrame(env)
        env = Quat2EulerWrapper(env)
        env = SERLObsWrapper(env, proprio_keys=self.proprio_keys)
        env = ChunkingWrapper(env, obs_horizon=1, act_exec_horizon=None)
        if classifier:
            classifier = load_classifier_func(
                key=jax.random.PRNGKey(0),
                sample=env.observation_space.sample(),
                image_keys=self.classifier_keys,
                checkpoint_path=os.path.abspath("classifier_ckpt/"),
            )

            def reward_func(obs):
                sigmoid = lambda x: 1 / (1 + jnp.exp(-x))
                return int(sigmoid(classifier(obs))[0] > 0.7)

            env = MultiCameraBinaryRewardClassifierWrapper(env, reward_func)
        # env = GripperPenaltyWrapper(env, penalty=-0.02)
        return env