import os
from tqdm import tqdm
import numpy as np
import copy
import pickle as pkl
import datetime
from absl import app, flags
import time

import jax
import jax.numpy as jnp
from serl_launcher.agents.continuous.bc import BCAgent
from serl_launcher.utils.launcher import make_bc_agent
from serl_launcher.utils.train_utils import cal_rl_action_penalty
from flax.training import checkpoints

from experiments.mappings import CONFIG_MAPPING

FLAGS = flags.FLAGS
flags.DEFINE_string("exp_name", None, "Name of experiment corresponding to folder.")
flags.DEFINE_integer("successes_needed", 10, "Number of successful demos to collect.")
flags.DEFINE_bool("use_bc_offset", False, "Use BC offset for intervention.") # True for rl+bc, False for rl or bc single agent
flags.DEFINE_string("bc_checkpoint_path", '/home/zxw/hil_serl/main/hil-serl/examples/experiments/astribot_test/bc_ckpt', "Path to save checkpoints.")
flags.DEFINE_integer("seed", 42, "Random seed.")

devices = jax.local_devices()
num_devices = len(devices)
sharding = jax.sharding.PositionalSharding(devices)


def main(_):
    assert FLAGS.exp_name in CONFIG_MAPPING, 'Experiment folder not found.'
    config = CONFIG_MAPPING[FLAGS.exp_name]()
    env = config.get_environment(fake_env=False, save_video=False, classifier=True)
    
    obs, info = env.reset()
    print("Reset done")
    transitions = []
    success_count = 0
    success_needed = FLAGS.successes_needed
    pbar = tqdm(total=success_needed)
    trajectory = []
    returns = 0

    # init the sampling_rng for bc_agent
    if FLAGS.use_bc_offset:
        print("Using BC offset for intervention.")
        sample_obs = env.observation_space.sample()
        bc_agent: BCAgent = make_bc_agent(
            seed=FLAGS.seed,
            sample_obs=sample_obs,
            sample_action=env.action_space.sample(),
            image_keys=config.image_keys,
            encoder_type=config.encoder_type,
        )

        # Load BC agent checkpoint
        bc_ckpt = checkpoints.restore_checkpoint(
            FLAGS.bc_checkpoint_path,
            bc_agent.state,
        )
        bc_agent = bc_agent.replace(state=bc_ckpt)
    
        bc_rng = jax.random.PRNGKey(FLAGS.seed)
        bc_sampling_rng = jax.device_put(bc_rng, sharding.replicate())

    
    while success_count < success_needed:
        zero_actions = np.zeros(env.action_space.sample().shape)

        if FLAGS.use_bc_offset:
            # Sample actions from BC agent, note there is no need to update the sampling_rng in BC eval mode
            bc_rng, bc_key = jax.random.split(bc_sampling_rng)
            obs_for_bc = copy.deepcopy(obs)
            # obs_for_bc['state'][:, :7] = 0.0 # disable the 'bc_action' in state
            bc_actions = bc_agent.sample_actions(
                observations=jax.device_put(obs_for_bc),
                seed=bc_key,
            )
            bc_actions = np.asarray(jax.device_get(bc_actions))
            env.unwrapped.bc_action_in_ee = bc_actions
            actions = bc_actions + zero_actions
            only_mouse_action = zero_actions
        else:
            actions = zero_actions
            only_mouse_action = zero_actions

        next_obs, rew, done, truncated, info = env.step(actions)
        if "intervene_action" in info:
            actions = info["intervene_action"]
            only_mouse_action = info["only_mouse_action"]
            if not FLAGS.use_bc_offset:
                assert(actions.all() == only_mouse_action.all())
        
        # add rl action penalty
        rew -= cal_rl_action_penalty(only_mouse_action)

        returns += rew

        transition = copy.deepcopy(
            dict(
                observations=obs,
                actions=only_mouse_action,
                next_observations=next_obs,
                rewards=rew,
                masks=1.0 - done,
                dones=done,
                infos=info,
            )
        )
        trajectory.append(transition)
        
        pbar.set_description(f"Return: {returns}")

        obs = next_obs
        if done:
            if info["succeed"]:
                for transition in trajectory:
                    transitions.append(copy.deepcopy(transition))
                success_count += 1
                pbar.update(1)
            trajectory = []
            returns = 0
            obs, info = env.reset()
            
    if not os.path.exists("./demo_data"):
        os.makedirs("./demo_data")
    uuid = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    file_name = f"./demo_data/{FLAGS.exp_name}_{success_needed}_demos_{uuid}.pkl"
    with open(file_name, "wb") as f:
        pkl.dump(transitions, f)
        print(f"saved {success_needed} demos to {file_name}")

if __name__ == "__main__":
    app.run(main)