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
from astribot_env.envs.astribot_env import DefaultEnvConfig
from serl_launcher.wrappers.serl_obs_wrappers import SERLObsWrapper
from serl_launcher.wrappers.chunking import ChunkingWrapper
from serl_launcher.networks.reward_classifier import load_classifier_func

from experiments.config import DefaultTrainingConfig
from experiments.astribot_test.wrapper import AstribotTestEnv, GripperPenaltyWrapper


class EnvConfig(DefaultEnvConfig):
    REALSENSE_CAMERAS = {
        "wrist": {
            "serial_number": "230322274773",
            "dim": (1280, 720),
            "exposure": 17000,
        },
    }
    IMAGE_CROP = {"wrist": lambda img: img[0:-1, 0:-1]}
    TARGET_POSE = np.array([0.45069316029548645, -0.30646514892578125, 0.8784106373786926, -0.02352749, -0.0531882, 1.53809226])
    RESET_POSE = TARGET_POSE + np.array([-0.2, 0, 0.05, 0, 0, 0])
    ACTION_SCALE = np.array([0.04, 0.1, 1])
    RANDOM_RESET = False
    DISPLAY_IMAGE = True
    RANDOM_XY_RANGE = 0.01
    RANDOM_RZ_RANGE = 0.1
    ABS_POSE_LIMIT_HIGH = TARGET_POSE + np.array([0.1, 0.1, 0.1, 0.5, 0.5, 0.5])
    ABS_POSE_LIMIT_LOW = TARGET_POSE - np.array([0.2, 0.1, 0.1, 0.5, 0.5, 0.5])
    MAX_EPISODE_LENGTH = 240
    JOINT_RESET_POSITION: np.ndarray = np.array([-0.05389, -0.55, 1.26345, 1.61261, 0.5537, 0.21308, -0.313582])
    EE_TARGET_POSE_FILTER_PARAM = 0.1
    RL_ACTION_WEIGHT = 1.0

    REWARD_THRESHOLD: np.ndarray = np.array([0.01, 0.1, 0.1, 0.2, 0.2, 0.2])


class TrainConfig(DefaultTrainingConfig):
    image_keys = ["wrist"]
    classifier_keys = ["wrist"]
    proprio_keys = ["tcp_pose","gripper_pose"]
    training_starts = 100
    checkpoint_period = 500
    cta_ratio = 2
    batch_size = 256
    random_steps = 0
    discount = 0.98
    buffer_period = 500
    encoder_type = "resnet-pretrained"
    setup_mode = "single-arm-learned-gripper"

    def get_environment(self, fake_env=False, save_video=False, classifier=False):
        env = AstribotTestEnv(
            fake_env=fake_env, save_video=save_video, config=EnvConfig(), bc_action_as_obs=False
        )
        if not fake_env:
            env = SpacemouseIntervention(env)
        env = RelativeFrame(env)
        env = Quat2EulerWrapper(env)
        env = SERLObsWrapper(env, proprio_keys=self.proprio_keys)
        env = ChunkingWrapper(env, obs_horizon=1, act_exec_horizon=None)
        # if classifier:
        if 0:
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
        env = GripperPenaltyWrapper(env, penalty=-0.02)
        return env