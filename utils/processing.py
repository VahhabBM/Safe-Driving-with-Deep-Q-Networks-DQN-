"""
Task 2: Implement Observation and Action Processing
=====================================================

Two responsibilities live here:

1. flatten_obs(obs)
   Converts whatever MetaDrive hands back from reset()/step() into a single
   flat float32 numpy array (or torch tensor) suitable as DQN input.
   MetaDrive's default (non-image) observation is already a 1D numpy array,
   but this function is defensive: it also handles the case where the
   observation is a dict (e.g. if image_observation / extra sensors were
   enabled) by flattening and concatenating every entry, so Task 3+ code
   never has to worry about observation shape/type again.

2. discrete_to_continuous_action(action_idx)
   Maps one of NUM_DISCRETE_ACTIONS discrete DQN action indices to the
   continuous [steering, throttle/brake] action MetaDrive expects, each
   component living in [-1, 1] as required by MetaDrive's action space.

   UPDATE: the original table only had 3 steering levels -- full left
   (-1.0), straight (0.0), full right (+1.0) -- with nothing in between.
   Diagnosed via inspect_policy.py: even after fixing Q-value divergence
   (Double DQN + reward clipping in dqn_agent.py) and confirming the
   network genuinely varies its action choice by state, the agent still
   ran off-road ~80% of the time. Root cause: on an ordinary (non-hairpin)
   curve, full-lock steering oversteers/overshoots and straight understeers
   off the outside of the curve -- there is no discrete action in between
   for the network to ever have picked, no matter how well-trained it is.
   This isn't a reward or learning problem; it's a missing action.

   Fix: added a "slight" steering level (+-0.4) on each side, giving 5
   steering positions x 2 throttle/brake = 10 actions instead of 6. 0.4 is
   a starting point (MetaDrive's default vehicle steering range is large
   enough that even 0.4 is a noticeable turn) -- tune SLIGHT_STEER below if
   10 actions still isn't fine-grained enough after retraining.

        Action Index | Meaning
        -------------|---------------------
             0        | Hard Left   + Brake
             1        | Hard Left   + Forward
             2        | Slight Left + Brake
             3        | Slight Left + Forward
             4        | Straight    + Brake
             5        | Straight    + Forward
             6        | Slight Right + Brake
             7        | Slight Right + Forward
             8        | Hard Right  + Brake
             9        | Hard Right  + Forward

   IMPORTANT: any code that constructs a DQNAgent (train.py, evaluate.py,
   inspect_dead_relu.py, inspect_policy.py) must use action_size=10 now
   (or better, `action_size=NUM_DISCRETE_ACTIONS` imported from here) --
   this is a breaking change, old 6-action checkpoints are NOT compatible
   and need to be retrained from scratch.
"""

from typing import Union, Dict, Any

import numpy as np


# ---------------------------------------------------------------------------
# Observation processing
# ---------------------------------------------------------------------------
def flatten_obs(obs: Union[np.ndarray, Dict[str, Any]]) -> np.ndarray:
    """Flatten a MetaDrive observation into a 1D float32 numpy array.

    Args:
        obs: observation returned by `env.reset()` / `env.step()`. Usually a
            1D numpy array for MetaDrive's default state observation, but
            dict observations are also supported defensively.

    Returns:
        1D np.ndarray of dtype float32.
    """
    if isinstance(obs, dict):
        pieces = [np.asarray(v, dtype=np.float32).reshape(-1) for v in obs.values()]
        return np.concatenate(pieces, axis=0)

    return np.asarray(obs, dtype=np.float32).reshape(-1)


def get_state_size(obs: Union[np.ndarray, Dict[str, Any]]) -> int:
    """Convenience helper: number of features in the flattened observation."""
    return flatten_obs(obs).shape[0]


# ---------------------------------------------------------------------------
# Action processing
# ---------------------------------------------------------------------------
# How sharp the "slight" steering positions are, as a fraction of full lock.
# 1.0 would be identical to hard left/right (defeats the purpose); something
# in the 0.25-0.5 range gives a genuinely different, gentler turn. Tune this
# if 0.4 still isn't fine-grained enough for the curves in your maps.
SLIGHT_STEER = 0.4

NUM_DISCRETE_ACTIONS = 10

# index -> (steering, throttle_brake)
#   steering:        -1.0 = full left,  0.0 = straight, +1.0 = full right
#   throttle_brake:  -1.0 = full brake, +1.0 = full forward throttle
DISCRETE_ACTION_TABLE = {
    0: (-1.0, -1.0),           # Hard Left + Brake
    1: (-1.0, 1.0),            # Hard Left + Forward
    2: (-SLIGHT_STEER, -1.0),  # Slight Left + Brake
    3: (-SLIGHT_STEER, 1.0),   # Slight Left + Forward
    4: (0.0, -1.0),            # Straight + Brake
    5: (0.0, 1.0),             # Straight + Forward
    6: (SLIGHT_STEER, -1.0),   # Slight Right + Brake
    7: (SLIGHT_STEER, 1.0),    # Slight Right + Forward
    8: (1.0, -1.0),            # Hard Right + Brake
    9: (1.0, 1.0),             # Hard Right + Forward
}

ACTION_MEANINGS = {
    0: "Hard Left + Brake",
    1: "Hard Left + Forward",
    2: "Slight Left + Brake",
    3: "Slight Left + Forward",
    4: "Straight + Brake",
    5: "Straight + Forward",
    6: "Slight Right + Brake",
    7: "Slight Right + Forward",
    8: "Hard Right + Brake",
    9: "Hard Right + Forward",
}


def discrete_to_continuous_action(action_idx: int) -> np.ndarray:
    """Map a discrete DQN action index (0 to NUM_DISCRETE_ACTIONS-1) to a
    MetaDrive continuous action.

    Args:
        action_idx: integer in [0, NUM_DISCRETE_ACTIONS - 1].

    Returns:
        np.ndarray of shape (2,) dtype float32: [steering, throttle_brake].
    """
    if action_idx not in DISCRETE_ACTION_TABLE:
        raise ValueError(
            f"action_idx must be in [0, {NUM_DISCRETE_ACTIONS - 1}], got {action_idx}"
        )
    return np.array(DISCRETE_ACTION_TABLE[action_idx], dtype=np.float32)
