# Intro:
This package provide a simple Franka arm and Robotiq Gripper simulator written in Mujoco.
It includes a state-based and a vision-based Franka lift cube task environment.

# Installation:
- From `hil-serl` folder, cd into `franka_sim`.
- In your `hil-serl` conda environment, run `pip install -e .` to install this package.
- run `pip install -r requirements.txt` to install sim dependencies.

# Explore the Environments
- Run `python franka_sim/test/test_gym_env_human.py` to launch a display window and visualize the task.
- Run `python franka_sim/test/test_gym_env_joystick.py` to launch a display window and use joystick to control the arm.

# Run Experiments
- Run `python examples/experiments/record_success_fail_sim.py` to record success and failure trajectories which is used for training the reward model.
- Run `python examples/experiments/record_demos_sim.py` to record demonstrations for training the policy.
- Run `bash examples/experiments/pick_cube_sim/run_actor.sh` and `bash examples/experiments/pick_cube_sim/run_learner.sh` to train a rlpd agent to solve the pick cube task.

# Credits:
- This simulation is initially built by [Kevin Zakka](https://kzakka.com/).
- Under Kevin's permission, we adopted a Gymnasium environment based on it.

# Notes:
- Error due to `egl` when running on a CPU machine:
```bash
export MUJOCO_GL=egl
conda install -c conda-forge libstdcxx-ng
```
