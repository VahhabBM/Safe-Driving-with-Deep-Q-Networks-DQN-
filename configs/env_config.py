"""
Task 1: Create the MetaDrive Environment
==========================================

This module is responsible ONLY for building and configuring a
`MetaDriveEnv` instance for single-agent DQN training.

Design goals (per assignment):
    - Reproducible training  -> fixed `start_seed` / `num_scenarios`
    - Single-agent driving   -> default MetaDriveEnv already is single-agent
    - Crash info available   -> MetaDrive exposes crash flags in `info` by
      default (`crash_vehicle`, `crash_object`, `out_of_road`, `arrive_dest`),
      so no extra config flag is required, but we keep `crash_vehicle_done`
      explicit for clarity / reproducibility.
    - Rendering configurable -> off during training (speed), can be turned on
      for evaluation / visualization (Task 8 later).
"""

from typing import Optional, Dict, Any

from metadrive.envs.metadrive_env import MetaDriveEnv


# ---------------------------------------------------------------------------
# Default configuration
# ---------------------------------------------------------------------------
# NOTE: keys below match MetaDrive's BaseEnv / MetaDriveEnv config schema.
#
# Debugging finding (was previously opt-in, now baked in as the default --
# see LEGACY_UNSAFE_REWARD_OVERRIDES below for why it changed):
#   MetaDrive's default reward uses
#     driving_reward * (progress along the route) * lateral_factor * positive_road
#   with `use_lateral_reward=False` by default -- meaning `lateral_factor`
#   does NOT depend on how close the car is to the lane center. The agent
#   gets the exact same dense reward whether it drives dead-center or
#   hugging the edge of the lane, right up until it fully exits the road (a
#   sparse, delayed `out_of_road_penalty`). This makes "just drive straight
#   and fast" the reward-maximizing behavior for a plain DQN -- exactly the
#   "reward goes up, crash rate goes up too" pattern from the reference
#   training run in the README (0% -> 38% crash rate over 800 episodes).
#   `use_lateral_reward=True` gives a continuous, every-step incentive to
#   stay lane-centered instead, and the *_penalty values below make
#   crashing/leaving the road cost more relative to the speed reward, so
#   risky "speed over safety" trajectories stop being reward-optimal.
DEFAULT_CONFIG: Dict[str, Any] = dict(
    # --- Reproducibility -----------------------------------------------
    num_scenarios=100,          # how many procedurally generated maps to draw from
    start_seed=0,                # seed of the first generated map
    random_lane_width=False,
    random_agent_model=False,

    # --- Episode / physics behavior -------------------------------------
    horizon=2000,                  # max steps per episode before truncation -- raised from
                                    # 1000. Evaluation showed average episode length already
                                    # reaching ~660-820 steps out of the old 1000-step horizon,
                                    # meaning many episodes may have been truncated (timed out)
                                    # before the car could actually reach the destination, not
                                    # because it failed -- it just ran out of steps. Doubling
                                    # this gives genuinely capable driving enough room to finish
                                    # a route instead of being cut off.
    crash_vehicle_done=True,      # episode ends on vehicle-vehicle crash
    crash_object_done=False,      # do not end episode on hitting static objects
    out_of_route_done=True,       # episode ends if the car drives off the route
    traffic_density=0.1,          # light traffic so the agent isn't overwhelmed early

    # --- Safety reward shaping (ON by default) ---------------------------
    #
    # IMPORTANT CORRECTION vs earlier versions of this comment: raising
    # out_of_road_penalty / crash_vehicle_penalty / crash_object_penalty
    # any further than they already are (50.0) has been INERT since
    # dqn_agent.py's reward_clip_min=-10.0 was added. Any terminal reward
    # of roughly -15 or more (a -50 penalty plus a small per-step driving
    # component) already saturates the clip floor at exactly -10 -- so
    # 20.0 vs 50.0 vs 80.0 here all produce the IDENTICAL clipped learning
    # signal. This is very likely why repeated attempts to fix a high
    # off-road rate by raising this number never actually worked: the
    # lever was disconnected. Leaving these at 50.0 (no further point
    # raising them) and instead tuning `driving_reward` below, which is
    # NOT clipped away (typical per-step values are well under 10).
    use_lateral_reward=True,      # dense per-step incentive to stay lane-centered
    out_of_road_penalty=50.0,     # MetaDrive default is 5.0 (see note above: raising this
                                   # further than 50 has no effect given reward_clip_min=-10)
    crash_vehicle_penalty=50.0,   # MetaDrive default is 5.0 (same clip-saturation note)
    crash_object_penalty=50.0,    # MetaDrive default is 5.0 (same clip-saturation note)
    driving_reward=0.5,           # MetaDrive default is 1.0 -- this is the weight on the
                                   # dense per-step "progress along the route" reward
                                   # (which use_lateral_reward's centering factor also
                                   # multiplies into). Halving it shrinks EVERY per-step
                                   # reward, which -- unlike the penalties above -- is NOT
                                   # clipped away, so this still meaningfully changes the
                                   # Bellman objective: the same -10 clipped terminal
                                   # penalty now represents a proportionally BIGGER hit
                                   # relative to the smaller per-step gains the agent
                                   # accumulates on the way there, pushing the optimal
                                   # policy toward more cautious driving instead of racing
                                   # through curves/traffic for progress reward that a
                                   # single failure can no longer "outrun" as easily.
    speed_reward=0.0,             # MetaDrive default is 0.1 -- this is the "reward for
                                   # going fast" term. Zeroing it removes the incentive
                                   # to rush into curves faster than the (coarse,
                                   # discrete) steering actions can actually handle;
                                   # `driving_reward` (route progress) still rewards
                                   # forward motion, just not raw speed for its own sake.
                                   # NOTE: verify this key name (and driving_reward above)
                                   # is valid for your installed metadrive-simulator==0.4.3
                                   # by running the 10-episode `--smoke-test` first -- if
                                   # it's not a recognized config key, that quick run will
                                   # fail fast instead of an hour into a full training run.

    # --- Goal-reaching reward (explicitly reward arriving, not just "not
    #     failing") -----------------------------------------------------
    success_reward=50.0,          # MetaDrive default is ~10.0. Nothing in the earlier
                                   # config asked the agent for anything beyond "don't
                                   # crash / don't leave the road" -- an episode that
                                   # survives to the horizon without ever reaching the
                                   # destination scored fine on every metric we were
                                   # tracking. Raising this (paired with the WIDER
                                   # positive reward-clip ceiling in dqn_agent.py's
                                   # reward_clip_max=50.0, vs. reward_clip_min=-10.0 on
                                   # the negative side) makes "actually arrive" register
                                   # as a clearly bigger one-time payoff than routine
                                   # per-step driving reward, instead of being clipped
                                   # down to the same size as everything else. Same
                                   # smoke-test caveat as speed_reward above applies.

    # --- Rendering --------------------------------------------------------
    use_render=False,             # no 3D window during training (much faster)
    manual_control=False,

    # --- Observation --------------------------------------------------
    # Use MetaDrive's default vectorized (state-only) observation, i.e. no
    # camera / image observation, so Task 2's flatten_obs stays simple and
    # fast. This is the default when `image_observation` is left as False.
    image_observation=False,
)


# ---------------------------------------------------------------------------
# LEGACY reward config (the ORIGINAL, spec-default, safety-blind reward)
# ---------------------------------------------------------------------------
# Kept only for comparison / ablation -- e.g. to reproduce the exact
# behavior documented in the README's reference run (reward 17->87 but
# crash rate 0%->38%). Pass `--legacy-reward` to `train.py` to use this
# instead of the (now-default) safety-shaped reward above.
LEGACY_UNSAFE_REWARD_OVERRIDES: Dict[str, Any] = dict(
    use_lateral_reward=False,
    out_of_road_penalty=5.0,
    crash_vehicle_penalty=5.0,
    crash_object_penalty=5.0,
    driving_reward=1.0,   # MetaDrive's actual original default
    speed_reward=0.1,     # MetaDrive's actual original default
    success_reward=10.0,  # MetaDrive's actual original default
)

# Backward-compatible alias: earlier versions of this file exposed the
# safety-shaped values under this name as an opt-in override. That
# shaping is now the default (folded into DEFAULT_CONFIG above), so this
# alias just points at the same values for any old code/notebooks that
# still import it by this name.
REWARD_SHAPING_OVERRIDES: Dict[str, Any] = dict(
    use_lateral_reward=True,
    out_of_road_penalty=50.0,
    crash_vehicle_penalty=50.0,
    crash_object_penalty=50.0,
    driving_reward=0.5,
    speed_reward=0.0,
    success_reward=50.0,
)


# ---------------------------------------------------------------------------
# Overrides to pair with reward_wrapper.SpeedCapRewardWrapper
# ---------------------------------------------------------------------------
# The wrapper ADDS its own speed-cap/smoothness/terminal shaping on top of
# whatever `env.step()` returns as `reward`. If out_of_road_penalty /
# crash_*_penalty / speed_reward / success_reward below were left at
# DEFAULT_CONFIG's values, the SAME failure/success event would be counted
# TWICE: once by MetaDrive's own config-driven reward, once by the wrapper's
# terminal penalty/bonus. Zero those four out here, but KEEP
# use_lateral_reward=True and a positive driving_reward -- that pairing is
# the only DENSE, continuous per-step signal (scales with lane-centering and
# route progress), which is what actually teaches steering through curves.
# A binary "on-road / off-road" flag alone (which is all the wrapper adds)
# has no gradient to learn fine steering from.
CUSTOM_REWARD_BASE_OVERRIDES: Dict[str, Any] = dict(
    use_lateral_reward=True,
    driving_reward=1.0,
    speed_reward=0.0,
    out_of_road_penalty=0.0,
    crash_vehicle_penalty=0.0,
    crash_object_penalty=0.0,
    success_reward=0.0,
)


def create_metadrive_env(
    config_overrides: Optional[Dict[str, Any]] = None,
    render: bool = False,
) -> MetaDriveEnv:
    """Create and configure a MetaDriveEnv instance.

    Args:
        config_overrides: dict of MetaDrive config keys to override/extend
            the defaults above (e.g. {"num_scenarios": 20} for quick tests).
        render: convenience flag to turn on the 3D render window
            (overrides `use_render` in config_overrides if both given).

    Returns:
        A ready-to-use MetaDriveEnv instance (not yet reset).
    """
    config = dict(DEFAULT_CONFIG)
    if config_overrides:
        config.update(config_overrides)
    if render:
        config["use_render"] = True

    env = MetaDriveEnv(config)
    return env


def sanity_check_env(env: MetaDriveEnv, n_steps: int = 20) -> None:
    """Verify that the environment can be reset and stepped through.

    Required by Task 1: "You should verify that the environment can be
    reset and stepped through before training the DQN agent."

    Raises an AssertionError if anything looks wrong; otherwise prints a
    short OK summary.
    """
    obs, info = env.reset()
    assert obs is not None, "reset() returned no observation"

    action_space = env.action_space
    for step in range(n_steps):
        action = action_space.sample()  # random continuous action, just for the smoke test
        obs, reward, terminated, truncated, info = env.step(action)

        assert obs is not None, f"step() returned no observation at step {step}"
        assert isinstance(reward, (int, float)), "reward is not a scalar"
        assert isinstance(info, dict), "info is not a dict"

        if terminated or truncated:
            obs, info = env.reset()

    print(
        f"[sanity_check_env] OK: env reset + {n_steps} steps ran without error. "
        f"Sample info keys: {list(info.keys())[:8]}"
    )


if __name__ == "__main__":
    # Quick manual smoke test: `python -m configs.env_config`
    test_env = create_metadrive_env(config_overrides={"num_scenarios": 5})
    try:
        sanity_check_env(test_env, n_steps=20)
    finally:
        test_env.close()
