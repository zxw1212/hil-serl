import numpy as np
import copy

def combine_actions(bc_actions: np.ndarray, sac_actions: np.ndarray, env, record_bc_action=True, bc_weight=1.0, sac_weight=0.01) -> np.ndarray:
    """
    Combine two actions using weighted sum with smoothed weights.
    """
    assert sac_actions.shape == bc_actions.shape
    assert sac_actions.shape[0] == 7
    sac_actions_scaled = copy.deepcopy(sac_actions)
    sac_actions_scaled *= sac_weight
    bc_scaled = copy.deepcopy(bc_actions)
    bc_scaled *= bc_weight
    actions = sac_actions_scaled + bc_scaled  # Combine actions (can use weighted sum if needed)

    if record_bc_action:
        env.unwrapped.record_sac_action_offset(bc_actions)  # Record the BC action offset

    return actions