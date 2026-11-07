"""
Custom reward-shaping wrapper (speed cap + action smoothness + terminal
failure/success penalties), ADDED ON TOP OF MetaDrive's own dense reward.

--- v2: why this no longer discards MetaDrive's base reward ------------------

v1 of this wrapper threw away `base_reward` entirely and replaced it with a
binary "+0.2 if on-road, -20 if not" signal. That produced 100% out-of-road
rate after 800 episodes (see the training-curve regression: reward
110 -> -14, episode length 700 -> 195 steps, out-of-road 62.5% -> 93.75%).

The reason: MetaDrive's own reward (`use_lateral_reward=True` +
`driving_reward`, set via configs/env_config.py's CUSTOM_REWARD_BASE_OVERRIDES)
is a DENSE, CONTINUOUS signal -- it scales with how centered the car is in
the lane and how much route progress it made, every single step. A binary
on-road/off-road flag has no gradient: it can only tell the agent "you
already failed," never "you're drifting, correct now." Without that
continuous signal, there's nothing teaching the network to actually track a
curve, so it drives roughly straight and runs off the first bend it meets --
which is exactly what happened.

This version ADDS the speed-cap and smoothness shaping ON TOP of that dense
signal instead of replacing it, and folds crash / out-of-road / over-speed
into a SINGLE terminal penalty applied once (whichever failure happened),
instead of a per-step penalty that can repeat many times while the car
lingers in an "out of road" state before the episode actually ends.

REQUIRED pairing: use this together with
`configs.env_config.CUSTOM_REWARD_BASE_OVERRIDES` (zeroes out MetaDrive's
own out_of_road_penalty / crash_*_penalty / speed_reward / success_reward,
since this wrapper now supplies its own versions of those -- without the
override, the same failure event gets penalized by BOTH MetaDrive's config
and this wrapper, i.e. double-counted again).
"""

import numpy as np
import gymnasium as gym


class SpeedCapRewardWrapper(gym.Wrapper):
    """Wraps a MetaDriveEnv, ADDING speed-cap/smoothness/terminal shaping to
    MetaDrive's own dense (lateral-centering + route-progress) reward.

    Pair with `configs.env_config.CUSTOM_REWARD_BASE_OVERRIDES`:

        from configs.env_config import create_metadrive_env, CUSTOM_REWARD_BASE_OVERRIDES
        from reward_wrapper import SpeedCapRewardWrapper

        env = create_metadrive_env(config_overrides={**your_overrides, **CUSTOM_REWARD_BASE_OVERRIDES})
        env = SpeedCapRewardWrapper(env)
    """

    def __init__(
        self,
        env,
        speed_limit_kmh: float = 30.0,
        hard_speed_limit_kmh: float = 45.0,
        crash_penalty: float = -50.0,
        out_of_road_penalty: float = -20.0,
        arrival_bonus: float = 50.0,
        smooth_weight: float = 0.05,
        speed_bonus_weight: float = 0.2,
        speed_penalty_weight: float = 2.0,
    ):
        super().__init__(env)
        self.speed_limit_kmh = speed_limit_kmh
        self.hard_speed_limit_kmh = hard_speed_limit_kmh
        self.crash_penalty = crash_penalty
        self.out_of_road_penalty = out_of_road_penalty
        self.arrival_bonus = arrival_bonus
        self.smooth_weight = smooth_weight
        self.speed_bonus_weight = speed_bonus_weight
        self.speed_penalty_weight = speed_penalty_weight
        self._prev_action = None

    def reset(self, **kwargs):
        self._prev_action = None
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, base_reward, terminated, truncated, info = self.env.step(action)

        speed_kmh = self.env.agent.speed * 3.6

        # --- speed: bonus up to the limit, penalty beyond it --------------
        if speed_kmh <= self.speed_limit_kmh:
            r_speed = self.speed_bonus_weight * speed_kmh / self.speed_limit_kmh
        else:
            r_speed = -self.speed_penalty_weight * (
                (speed_kmh - self.speed_limit_kmh) / self.speed_limit_kmh
            )

        # --- action smoothness ----------------------------------------------
        action_arr = np.asarray(action, dtype=np.float32)
        if self._prev_action is None:
            r_smooth = 0.0
        else:
            delta = float(np.linalg.norm(action_arr - self._prev_action))
            r_smooth = self.smooth_weight * (1.0 - np.clip(delta, 0.0, 1.0))
        self._prev_action = action_arr

        # --- ONE terminal penalty/bonus per step, not several stacked ones --
        # (fixes both the v1 double-crash-penalty bug and the "-20 fires
        # every step while lingering off-road" blow-up: exactly one branch
        # of this elif chain can fire on any given step.)
        crashed = bool(info.get("crash_vehicle", False) or info.get("crash_object", False))
        out_of_road = bool(info.get("out_of_road", False))
        hard_speeding = speed_kmh >= self.hard_speed_limit_kmh
        arrived = bool(info.get("arrive_dest", False))

        if crashed or hard_speeding:
            r_terminal = self.crash_penalty
        elif out_of_road:
            r_terminal = self.out_of_road_penalty
        elif arrived:
            r_terminal = self.arrival_bonus
        else:
            r_terminal = 0.0

        # base_reward carries MetaDrive's dense lateral-centering + route-
        # progress signal (see CUSTOM_REWARD_BASE_OVERRIDES) -- this is what
        # actually teaches the network to steer through curves.
        reward = base_reward + r_speed + r_smooth + r_terminal

        info["reward_breakdown"] = {
            "base": base_reward,
            "speed": r_speed,
            "smooth": r_smooth,
            "terminal": r_terminal,
        }
        return obs, reward, terminated, truncated, info

