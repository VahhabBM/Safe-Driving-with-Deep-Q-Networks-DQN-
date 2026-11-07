"""
Task 6: Implement the Training Loop
Task 7: Train the Agent
=====================================================

Task 6 -- `train_dqn(...)`:
    - resets the env at the start of every episode and flattens the obs
    - each step: epsilon-greedy action -> continuous action -> env.step()
      -> flatten next obs -> store transition in the replay buffer
    - learning only starts once `min_buffer_size` transitions are collected
    - episode ends on `terminated` or `truncated` (MetaDrive's own signals)
    - per episode we record: total reward, number of steps, and whether a
      crash occurred (from `info["crash_vehicle"]` / `info["crash_object"]`)

Task 7 -- `main()`:
    - `--smoke-test` : quick 5-20 episode run to make sure nothing crashes
    - default        : full run (500-1000 episodes) for the real experiment
    - saves the trained online network to `models/dqn_trained.pt`
    - saves the episode history (reward/length/crash/loss/epsilon) to a
      JSON file and, if matplotlib is available, a PNG of the training
      curves -- this is intentionally reused later for Task 8.

Run with (from the `metadrive_dqn/` directory):
    python -m train --smoke-test                 # Task 7, quick check
    python -m train --episodes 1000               # Task 7, full run
"""

import argparse
import json
import os
import shutil
import time
from typing import Dict, List, Optional

import numpy as np

# NOTE on import order: on some Windows setups, importing metadrive/panda3d
# before torch causes torch's native DLLs (c10.dll and friends) to fail to
# load ("OSError: [WinError 1114] A dynamic link library (DLL)
# initialization routine failed"). Importing torch (via dqn_agent) first
# avoids that DLL-loading conflict. Keep `dqn_agent` above `configs.env_config`
# if you ever reorder these.
from dqn_agent import DQNAgent
import torch  # kept right after dqn_agent (which already imports torch) -- see DLL note above
from configs.env_config import create_metadrive_env, CUSTOM_REWARD_BASE_OVERRIDES
from utils.processing import flatten_obs, discrete_to_continuous_action, NUM_DISCRETE_ACTIONS
from reward_wrapper import SpeedCapRewardWrapper


# ---------------------------------------------------------------------------
# Task 6: Training loop
# ---------------------------------------------------------------------------
def train_dqn(
    agent: DQNAgent,
    env,
    num_episodes: int,
    min_buffer_size: int = 1000,
    max_steps_per_episode: Optional[int] = None,
    learn_every: int = 1,
    log_every: int = 10,
    checkpoint_path: Optional[str] = None,
    checkpoint_every: int = 100,
    verbose: bool = True,
) -> Dict[str, List[float]]:
    """Run the DQN training loop described in Task 6.

    Args:
        agent: a `DQNAgent` (Task 5) -- owns the networks + replay buffer.
        env: a `MetaDriveEnv` created via `create_metadrive_env` (Task 1).
        num_episodes: how many episodes to train for.
        min_buffer_size: number of transitions that must be collected
            before `agent.learn()` is ever called (avoids learning from a
            handful of highly correlated early transitions).
        max_steps_per_episode: optional hard cap on steps per episode, on
            top of whatever `horizon` the env config already enforces
            (useful for quick smoke tests).
        learn_every: call `agent.learn()` every N environment steps once
            `min_buffer_size` is reached (1 = every step).
        log_every: print a progress line every N episodes.
        checkpoint_path: if given, `agent.save(...)` is called here every
            `checkpoint_every` episodes (crash-safe long training runs).
        checkpoint_every: see above.
        verbose: whether to print progress lines at all.

    Returns:
        history: dict of per-episode lists --
            "episode_reward", "episode_length", "crashed" (bool),
            "out_of_road" (bool), "arrived" (bool -- reached the destination,
            from info["arrive_dest"]; this is what actually answers "did the
            car get where it was going", separate from just "didn't fail"),
            "avg_loss" (nan if no learn() call happened that episode),
            "epsilon" (value *after* the episode's decay_epsilon() call).
    """
    history: Dict[str, List[float]] = {
        "episode_reward": [],
        "episode_length": [],
        "crashed": [],
        "out_of_road": [],
        "arrived": [],
        "avg_loss": [],
        "epsilon": [],
    }

    for episode in range(1, num_episodes + 1):
        obs, info = env.reset()
        state = flatten_obs(obs)

        episode_reward = 0.0
        episode_steps = 0
        episode_crashed = False
        episode_out_of_road = False
        episode_arrived = False
        episode_losses = []

        done = False
        while not done:
            # --- act -----------------------------------------------------
            action_idx = agent.select_action(state)
            action = discrete_to_continuous_action(action_idx)

            # --- step the real environment --------------------------------
            next_obs, reward, terminated, truncated, info = env.step(action)
            next_state = flatten_obs(next_obs)
            done = bool(terminated or truncated)

            # --- store + learn ---------------------------------------------
            agent.store_transition(state, action_idx, reward, next_state, done)

            if (
                len(agent.replay_buffer) >= min_buffer_size
                and episode_steps % learn_every == 0
            ):
                loss = agent.learn()
                if loss is not None:
                    episode_losses.append(loss)

            # --- bookkeeping ------------------------------------------------
            state = next_state
            episode_reward += reward
            episode_steps += 1
            if info.get("crash_vehicle", False) or info.get("crash_object", False):
                episode_crashed = True
            if info.get("out_of_road", False):
                episode_out_of_road = True
            if info.get("arrive_dest", False):
                episode_arrived = True

            if max_steps_per_episode is not None and episode_steps >= max_steps_per_episode:
                break

        # once per episode, per the standard DQN epsilon schedule
        agent.decay_epsilon()

        history["episode_reward"].append(episode_reward)
        history["episode_length"].append(episode_steps)
        history["crashed"].append(episode_crashed)
        history["out_of_road"].append(episode_out_of_road)
        history["arrived"].append(episode_arrived)
        history["avg_loss"].append(
            float(np.mean(episode_losses)) if episode_losses else float("nan")
        )
        history["epsilon"].append(agent.epsilon)

        if verbose and (episode % log_every == 0 or episode == 1 or episode == num_episodes):
            recent_r = history["episode_reward"][-log_every:]
            recent_c = history["crashed"][-log_every:]
            recent_o = history["out_of_road"][-log_every:]
            recent_a = history["arrived"][-log_every:]
            print(
                f"Ep {episode:5d}/{num_episodes} | "
                f"reward={episode_reward:8.2f} | steps={episode_steps:4d} | "
                f"crash={str(episode_crashed):5s} | off_road={str(episode_out_of_road):5s} | "
                f"arrived={str(episode_arrived):5s} | "
                f"eps={agent.epsilon:.3f} | "
                f"avgR(last {len(recent_r)})={np.mean(recent_r):8.2f} | "
                f"crashRate(last {len(recent_c)})={np.mean(recent_c):.2f} | "
                f"offRoadRate(last {len(recent_o)})={np.mean(recent_o):.2f} | "
                f"arrivedRate(last {len(recent_a)})={np.mean(recent_a):.2f}"
            )

        if checkpoint_path is not None and episode % checkpoint_every == 0:
            os.makedirs(os.path.dirname(checkpoint_path) or ".", exist_ok=True)
            agent.save(checkpoint_path)
            if verbose:
                print(f"  [checkpoint] saved -> {checkpoint_path} (episode {episode})")

    return history


# ---------------------------------------------------------------------------
# Helpers: persisting history + plotting (reused by Task 8 later)
# ---------------------------------------------------------------------------
def save_history(history: Dict[str, List[float]], path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(history, f, indent=2)


def plot_history(history: Dict[str, List[float]], out_path: str) -> None:
    """Quick training-curve plot: reward / episode length / crash rate."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    episodes = np.arange(1, len(history["episode_reward"]) + 1)

    fig, axes = plt.subplots(3, 1, figsize=(9, 10), sharex=True)

    axes[0].plot(episodes, history["episode_reward"], color="tab:blue")
    axes[0].set_ylabel("Episode reward")
    axes[0].set_title("DQN training progress")

    axes[1].plot(episodes, history["episode_length"], color="tab:orange")
    axes[1].set_ylabel("Episode length")

    crashed = np.asarray(history["crashed"], dtype=float)
    out_of_road = np.asarray(history.get("out_of_road", [0] * len(crashed)), dtype=float)
    arrived = np.asarray(history.get("arrived", [0] * len(crashed)), dtype=float)
    window = max(1, len(episodes) // 50)

    def rolling(arr):
        if len(arr) >= window and window > 1:
            return np.convolve(arr, np.ones(window) / window, mode="valid")
        return arr

    crash_rate = rolling(crashed)
    off_road_rate = rolling(out_of_road)
    arrived_rate = rolling(arrived)
    x = np.arange(1, len(crash_rate) + 1)
    axes[2].plot(x, crash_rate, color="tab:red", label="crash rate")
    if history.get("out_of_road") is not None:
        axes[2].plot(x, off_road_rate, color="tab:purple", label="out-of-road rate")
    if history.get("arrived") is not None:
        axes[2].plot(x, arrived_rate, color="tab:green", label="arrival rate")
    axes[2].legend(loc="upper left", fontsize=8)
    axes[2].set_ylabel(f"Rate (rolling w={window})")
    axes[2].set_xlabel("Episode")

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def build_agent(
    state_size: int,
    action_size: int = NUM_DISCRETE_ACTIONS,
    seed: Optional[int] = 0,
    lr: float = 1e-3,
    epsilon_start: float = 1.0,
    epsilon_end: float = 0.05,
    epsilon_decay: float = 0.995,
    hidden_size: int = 256,
    device: Optional[str] = None,
    reward_clip_min: Optional[float] = -10.0,
    reward_clip_max: Optional[float] = 50.0,
) -> DQNAgent:
    """Single place to tune DQNAgent hyperparameters for the real run."""
    return DQNAgent(
        state_size=state_size,
        action_size=action_size,
        hidden_size=hidden_size,
        buffer_capacity=100_000,
        batch_size=64,
        gamma=0.99,
        lr=lr,
        epsilon_start=epsilon_start,
        epsilon_end=epsilon_end,
        epsilon_decay=epsilon_decay,
        target_update_freq=1000,
        seed=seed,
        device=device,
        reward_clip_min=reward_clip_min,
        reward_clip_max=reward_clip_max,
    )


# ---------------------------------------------------------------------------
# Task 7: Train the Agent
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Task 6/7: train the DQN agent in MetaDrive.")
    parser.add_argument(
        "--smoke-test", action="store_true",
        help="Quick 10-episode debug run instead of the full training run.",
    )
    parser.add_argument("--episodes", type=int, default=None,
                         help="Number of training episodes (default: 10 for --smoke-test, else 800).")
    parser.add_argument("--num-scenarios", type=int, default=100,
                         help="Number of procedurally generated maps to train on.")
    parser.add_argument("--horizon", type=int, default=1000, help="Max steps per episode.")
    parser.add_argument("--min-buffer-size", type=int, default=1000,
                         help="Transitions collected before learning starts.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model-out", type=str, default="models/dqn_trained.pt",
                         help="Where to save the final trained Q-network.")
    parser.add_argument("--history-out", type=str, default="models/training_history.json")
    parser.add_argument("--plot-out", type=str, default="models/training_curves.png")
    parser.add_argument(
        "--legacy-reward", action="store_true",
        help="Use the OLD, spec-default, safety-blind reward (LEGACY_UNSAFE_REWARD_OVERRIDES "
             "in configs/env_config.py) instead of the safety-shaped reward that is now the "
             "default. Use this only to reproduce/compare against the old crash-prone "
             "behavior documented in the README (reward went up, crash rate went up too).",
    )
    parser.add_argument(
        "--reward-shaping", action="store_true",
        help="Deprecated / no-op: the safety-shaped reward this used to opt into is now "
             "applied by default (see configs/env_config.py DEFAULT_CONFIG). Kept only so "
             "old commands with this flag don't error out. Use --legacy-reward if you want "
             "the OLD safety-blind behavior instead.",
    )
    parser.add_argument(
        "--traffic-density", type=float, default=None,
        help="Override traffic_density (default from env_config.py is 0.1). "
             "Use 0.0 for a curriculum 'Phase 1' run (learn lane-following/steering "
             "with no other cars to crash into, isolating the out-of-road failure mode).",
    )
    parser.add_argument("--lr", type=float, default=1e-3, help="Adam learning rate.")
    parser.add_argument("--hidden-size", type=int, default=256,
                         help="Width of the DQNNetwork hidden layers. Increase (e.g. 512) "
                              "if inspect_dead_relu.py shows a very high dead-unit fraction, "
                              "to give the network more effective capacity margin.")
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-decay", type=float, default=0.995,
                         help="Per-episode epsilon decay factor. Use a value closer to 1 "
                              "(e.g. 0.998) for slower decay / more exploration over more episodes.")
    parser.add_argument(
        "--reward-clip-min", type=float, default=-10.0,
        help="Clip each raw reward's negative side at this floor before storing it in the "
             "replay buffer (Atari-DQN-style). Guards against Q-value divergence when "
             "*_penalty values in configs/env_config.py are large relative to per-step dense "
             "reward -- see inspect_policy.py output for what unstable/exploded Q-values look "
             "like (values in the 1000s, well beyond max_reward/(1-gamma)). Pass e.g. -1e9 to "
             "effectively disable the floor.",
    )
    parser.add_argument(
        "--reward-clip-max", type=float, default=50.0,
        help="Clip each raw reward's positive side at this ceiling. Deliberately HIGHER than "
             "the (negative) --reward-clip-min floor by default: this is what lets a big "
             "one-time success_reward (configs/env_config.py, for reaching the destination) "
             "still register as clearly larger than routine per-step reward, instead of being "
             "clipped down to the same magnitude as an ordinary crash/off-road penalty. Pass "
             "e.g. 1e9 to effectively disable the ceiling.",
    )
    parser.add_argument(
        "--device", type=str, default=None, choices=["cuda", "cpu"],
        help="Force the training device. Default (omit this flag): auto-detect -- "
             "uses 'cuda' automatically if a GPU is available, else 'cpu'. Pass "
             "--device cuda to require the GPU (errors out if none is visible) or "
             "--device cpu to force CPU even if a GPU is present.",
    )
    parser.add_argument(
        "--resume-from", type=str, default=None,
        help="Path to an existing dqn_trained.pt to load BEFORE training (e.g. to continue "
             "a curriculum 'Phase 1' run into a 'Phase 2' run with traffic turned back on). "
             "The replay buffer is NOT restored (starts fresh), only the network weights + "
             "observation normalizer stats. IMPORTANT: --hidden-size must match the value "
             "used to train the checkpoint being resumed, or loading will fail with a "
             "shape-mismatch error. NOTE: for a single-command curriculum run, prefer "
             "--curriculum-phase1-episodes below instead -- it keeps the SAME replay buffer "
             "and epsilon schedule across both phases (no cold restart).",
    )
    parser.add_argument(
        "--curriculum-phase1-episodes", type=int, default=0,
        help="If > 0, run this many episodes FIRST with traffic_density="
             "--curriculum-traffic-density (default 0.0, no traffic at all) so the agent can "
             "master basic lane-keeping/steering before any cars are around to crash into or "
             "distract it. The SAME agent (network weights, replay buffer, epsilon schedule) "
             "then continues straight into phase 2, which runs the remaining "
             "(--episodes minus this value) episodes at the normal --traffic-density. Use this "
             "if crash rate looks fine but out-of-road rate is high/near 1.0 -- that pattern "
             "means the bottleneck is steering/lane-keeping capability, not the crash-vs-speed "
             "reward balance, and curriculum learning targets that directly.",
    )
    parser.add_argument(
        "--curriculum-traffic-density", type=float, default=0.0,
        help="traffic_density used during curriculum phase 1 (see "
             "--curriculum-phase1-episodes above). Default 0.0.",
    )
    parser.add_argument(
        "--custom-reward", action="store_true",
        help="Wrap the env with reward_wrapper.SpeedCapRewardWrapper and apply "
             "configs/env_config.py's CUSTOM_REWARD_BASE_OVERRIDES automatically. The "
             "wrapper ADDS speed-cap/smoothness/terminal shaping ON TOP OF MetaDrive's own "
             "dense lateral-centering + route-progress reward (kept via use_lateral_reward "
             "+ driving_reward) -- it does NOT replace it. CUSTOM_REWARD_BASE_OVERRIDES "
             "zeroes out MetaDrive's own out_of_road_penalty/crash_*_penalty/speed_reward/"
             "success_reward so those don't get double-counted alongside the wrapper's own "
             "versions. If --reward-clip-min/--reward-clip-max are left at their defaults "
             "(-10.0 / 50.0) when this flag is set, they are auto-widened to -60.0 / 60.0 "
             "-- otherwise the wrapper's -20/-50 terminal penalties clip down to the same "
             "-10 before the network ever sees them.",
    )
    args = parser.parse_args()

    if args.custom_reward:
        if args.reward_clip_min == -10.0:
            args.reward_clip_min = -60.0
            print("[--custom-reward] reward_clip_min left at the MetaDrive-reward default "
                  "(-10.0) -- auto-widening to -60.0 so the wrapper's -20/-50 penalties "
                  "stay distinguishable. Pass --reward-clip-min explicitly to override.")
        if args.reward_clip_max == 50.0:
            args.reward_clip_max = 60.0
            print("[--custom-reward] reward_clip_max left at the MetaDrive-reward default "
                  "(50.0) -- auto-widening to 60.0 to comfortably fit the wrapper's "
                  "arrival_bonus=50.0. Pass --reward-clip-max explicitly to override.")

    if args.episodes is None:
        args.episodes = 10 if args.smoke_test else 800

    if args.curriculum_phase1_episodes >= args.episodes:
        raise ValueError(
            f"--curriculum-phase1-episodes ({args.curriculum_phase1_episodes}) must be "
            f"smaller than --episodes ({args.episodes}) -- phase 2 needs at least 1 episode."
        )

    print("=" * 70)
    print(f"Task 7: training for {args.episodes} episodes "
          f"({'SMOKE TEST' if args.smoke_test else 'FULL RUN'})"
          f"{' [LEGACY safety-blind reward]' if args.legacy_reward else ' [safety-shaped reward (default)]'}")
    print("=" * 70)
    if args.reward_shaping:
        print("Note: --reward-shaping is deprecated and has no effect -- the reward "
              "shaping it used to enable is now the default. Use --legacy-reward if "
              "you actually want the OLD safety-blind behavior instead.")

    # --- GPU diagnostics --------------------------------------------------
    # DQNAgent already auto-selects "cuda" whenever it's available (see
    # dqn_agent.py), so no code change was strictly required to train on a
    # GPU -- this block just makes what device will actually be used
    # obvious up front, and lets --device force one explicitly.
    cuda_available = torch.cuda.is_available()
    print(f"torch: {torch.__version__} | CUDA available: {cuda_available}"
          + (f" | GPU: {torch.cuda.get_device_name(0)}" if cuda_available else ""))
    if args.device == "cuda" and not cuda_available:
        raise RuntimeError(
            "--device cuda was requested but torch.cuda.is_available() is False. "
            "This usually means a CPU-only build of torch is installed -- reinstall "
            "torch with a CUDA wheel (see requirements.txt for the install command) "
            "and check your GPU driver, then re-run."
        )
    if args.device is None and not cuda_available:
        print("(No GPU detected -- training will run on CPU. See requirements.txt "
              "if you intended to use a GPU.)")

    env_overrides = {"num_scenarios": args.num_scenarios, "horizon": args.horizon}
    if args.traffic_density is not None:
        env_overrides["traffic_density"] = args.traffic_density
    if args.legacy_reward:
        from configs.env_config import LEGACY_UNSAFE_REWARD_OVERRIDES
        env_overrides.update(LEGACY_UNSAFE_REWARD_OVERRIDES)
        print(f"LEGACY safety-blind reward overrides applied: {LEGACY_UNSAFE_REWARD_OVERRIDES}")
    print(f"Env overrides: {env_overrides}")

    if args.custom_reward:
        env_overrides.update(CUSTOM_REWARD_BASE_OVERRIDES)
        print(f"[--custom-reward] applied CUSTOM_REWARD_BASE_OVERRIDES (keeps MetaDrive's "
              f"dense lateral/progress signal, zeroes the components reward_wrapper.py "
              f"supplies itself): {CUSTOM_REWARD_BASE_OVERRIDES}")

    env = create_metadrive_env(config_overrides=env_overrides)
    if args.custom_reward:
        env = SpeedCapRewardWrapper(env)
        print("[--custom-reward] env wrapped with SpeedCapRewardWrapper -- reward is now "
              "MetaDrive's dense lateral/progress signal PLUS speed-cap/smoothness/terminal "
              "shaping (crash_vehicle_done / out_of_route_done still control termination "
              "either way).")

    try:
        obs, info = env.reset()
        state_size = flatten_obs(obs).shape[0]
        agent = build_agent(
            state_size=state_size,
            action_size=NUM_DISCRETE_ACTIONS,
            seed=args.seed,
            lr=args.lr,
            epsilon_start=args.epsilon_start,
            epsilon_decay=args.epsilon_decay,
            hidden_size=args.hidden_size,
            device=args.device,
            reward_clip_min=args.reward_clip_min,
            reward_clip_max=args.reward_clip_max,
        )
        if args.resume_from:
            agent.load(args.resume_from)
            print(f"Resumed network weights from {args.resume_from} "
                  f"(replay buffer starts empty, epsilon starts at {args.epsilon_start})")
            if args.epsilon_start < 0.6:
                print(
                    f"*** WARNING: --epsilon-start={args.epsilon_start} is low for a resumed "
                    f"run. The resumed weights encode a policy that was learned under whatever "
                    f"reward function/config was active in the ORIGINAL run that produced "
                    f"{args.resume_from} -- if you've changed anything in configs/env_config.py "
                    f"or dqn_agent.py's reward clipping since then (horizon, success_reward, "
                    f"*_penalty, reward_clip_min/max, etc.), that old policy is no longer "
                    f"guaranteed to be good under the NEW reward function, and a low epsilon "
                    f"means the agent will mostly EXPLOIT that possibly-stale policy instead of "
                    f"exploring to find what's actually optimal now. This exact pattern (resume "
                    f"+ low epsilon + changed reward) has previously caused a training "
                    f"regression in this project (crash/off-road rates got WORSE, not better). "
                    f"Consider --epsilon-start 1.0 (or at least 0.7) whenever you resume after "
                    f"changing the reward function, even though it means re-exploring for a "
                    f"while before the higher-quality weights pay off again. ***"
                )
        print(f"state_size={state_size}, device={agent.device}, lr={args.lr}, "
              f"epsilon_decay={args.epsilon_decay}, hidden_size={args.hidden_size}, "
              f"reward_clip=[{args.reward_clip_min}, {args.reward_clip_max}]")

        # Back up any existing checkpoint at --model-out BEFORE this run's periodic
        # checkpointing or final save can overwrite it. A good checkpoint has already
        # been silently lost this way once in this project (a run that made things
        # worse overwrote the previous good models/dqn_trained.pt with no copy left).
        # This makes that mistake structurally impossible going forward.
        if os.path.exists(args.model_out):
            backup_path = (
                f"{os.path.splitext(args.model_out)[0]}"
                f".backup_{time.strftime('%Y%m%d_%H%M%S')}"
                f"{os.path.splitext(args.model_out)[1]}"
            )
            shutil.copy2(args.model_out, backup_path)
            print(f"Existing checkpoint at {args.model_out} backed up -> {backup_path} "
                  f"(in case this run turns out worse -- compare with inspect_policy.py / "
                  f"evaluate.py before deleting old backups).")

        min_buffer_size = args.min_buffer_size if not args.smoke_test else min(200, args.min_buffer_size)

        start = time.time()

        phase1_history = None
        phase2_episodes = args.episodes

        if args.curriculum_phase1_episodes > 0:
            phase1_overrides = dict(env_overrides)
            phase1_overrides["traffic_density"] = args.curriculum_traffic_density
            print(f"\n--- Curriculum phase 1: {args.curriculum_phase1_episodes} episodes at "
                  f"traffic_density={args.curriculum_traffic_density} (isolating lane-keeping/"
                  f"steering, no traffic to crash into or be distracted by) ---")
            phase1_env = create_metadrive_env(config_overrides=phase1_overrides)
            if args.custom_reward:
                phase1_env = SpeedCapRewardWrapper(phase1_env)
            try:
                phase1_history = train_dqn(
                    agent=agent,
                    env=phase1_env,
                    num_episodes=args.curriculum_phase1_episodes,
                    min_buffer_size=min_buffer_size,
                    checkpoint_path=args.model_out,
                    checkpoint_every=max(1, args.curriculum_phase1_episodes // 5),
                    log_every=1 if args.smoke_test else 10,
                )
            finally:
                phase1_env.close()
            print(
                f"--- Curriculum phase 1 finished. Last-10 crash rate: "
                f"{np.mean(phase1_history['crashed'][-10:]):.2f}, off-road rate: "
                f"{np.mean(phase1_history['out_of_road'][-10:]):.2f} -- if off-road rate is "
                f"still high here, the agent hasn't learned basic steering yet; consider "
                f"raising --curriculum-phase1-episodes before moving on to phase 2. ---\n"
                f"--- Curriculum phase 2: {args.episodes - args.curriculum_phase1_episodes} "
                f"episodes at traffic_density={env_overrides.get('traffic_density', 0.1)} "
                f"(same agent -- weights, replay buffer, and epsilon schedule all carry over "
                f"from phase 1) ---"
            )
            phase2_episodes = args.episodes - args.curriculum_phase1_episodes

        phase2_history = train_dqn(
            agent=agent,
            env=env,
            num_episodes=phase2_episodes,
            min_buffer_size=min_buffer_size,
            checkpoint_path=args.model_out,
            checkpoint_every=max(1, phase2_episodes // 5),
            log_every=1 if args.smoke_test else 10,
        )

        history = (
            {k: phase1_history[k] + phase2_history[k] for k in phase2_history}
            if phase1_history is not None
            else phase2_history
        )

        elapsed = time.time() - start
        print(f"\nTraining finished in {elapsed / 60:.1f} min "
              f"({elapsed / max(1, args.episodes):.2f} s/episode)")

        # Task 7: save the trained model exactly at models/dqn_trained.pt
        agent.save(args.model_out)
        print(f"Saved trained model -> {args.model_out}")

        save_history(history, args.history_out)
        print(f"Saved training history -> {args.history_out}")

        try:
            plot_history(history, args.plot_out)
            print(f"Saved training curves -> {args.plot_out}")
        except ImportError:
            print("matplotlib not available -- skipped plotting (history JSON still saved).")

        print("\nSummary:")
        print(f"  mean reward (last 10 eps): {np.mean(history['episode_reward'][-10:]):.2f}")
        print(f"  mean length (last 10 eps): {np.mean(history['episode_length'][-10:]):.1f}")
        print(f"  crash rate  (last 10 eps): {np.mean(history['crashed'][-10:]):.2f}")
        print(f"  out-of-road rate (last 10 eps): {np.mean(history['out_of_road'][-10:]):.2f}")
        if "arrived" in history:
            print(f"  arrival rate (last 10 eps): {np.mean(history['arrived'][-10:]):.2f}")

    finally:
        env.close()


if __name__ == "__main__":
    main()
