"""
Task 8: Evaluation and Visualization
=====================================================

This script assumes you already have:
  - models/dqn_trained.pt              (Task 7 output)
  - models/training_history.json       (saved by train.py during Task 7)

It provides three independent pieces, each runnable on its own:

1. `evaluate_policy(...)`
   Runs the trained agent GREEDILY (epsilon=0) for a number of episodes and
   reports average reward, average episode length, and crash rate --
   exactly what the assignment asks to "report" from evaluation episodes.

2. `analyze_training_curves(history)`
   Reuses `train.plot_history` to draw the reward/length/crash-rate plots,
   and additionally prints a short, DATA-DRIVEN discussion comparing the
   first vs. last chunk of training episodes (reward, length, crash rate),
   so you can see numerically whether training improved things -- this is
   the "discuss whether reward improves, length changes, crash rate
   decreases" part of Task 8. You still write the final reflection in your
   own words; this only gives you the numbers to back it up.

3. `record_topdown_video(...)` and `run_3d_visualization(...)`
   Visualize the trained policy in MetaDrive:
     - top-down: renders each step as an image via env.render(mode="topdown")
       and saves the sequence as a video/gif (no on-screen window needed).
     - 3D: opens MetaDrive's interactive 3D render window
       (use_render=True) so you can watch/record the agent driving; also
       grabs periodic in-window screenshots to save as evidence for your
       submission.

Run examples (from the `metadrive_dqn/` directory):
    python -m evaluate --eval-episodes 20
    python -m evaluate --plot-only
    python -m evaluate --topdown-video --video-steps 500
    python -m evaluate --show-3d
    python -m evaluate --eval-episodes 20 --plot-only --topdown-video --show-3d
"""

import argparse
import json
import os
from typing import Dict, List, Optional

import numpy as np

# Same DLL-safe import order as train.py: torch (via dqn_agent) before
# metadrive/panda3d (via configs.env_config).
from dqn_agent import DQNAgent
import torch  # kept right after dqn_agent (which already imports torch) -- see DLL note above
from configs.env_config import create_metadrive_env, CUSTOM_REWARD_BASE_OVERRIDES
from utils.processing import flatten_obs, discrete_to_continuous_action, NUM_DISCRETE_ACTIONS
from train import plot_history
from reward_wrapper import SpeedCapRewardWrapper


# ---------------------------------------------------------------------------
# 1) Evaluation: run the trained policy greedily and report statistics
# ---------------------------------------------------------------------------
def evaluate_policy(
    agent: DQNAgent,
    env,
    num_episodes: int = 20,
    max_steps_per_episode: Optional[int] = None,
    verbose: bool = True,
) -> Dict[str, List[float]]:
    """Run `num_episodes` fully greedy (epsilon=0) episodes and record stats.

    Returns a dict with per-episode "episode_reward", "episode_length",
    "crashed", "out_of_road", "arrived" (reached the destination, from
    info["arrive_dest"]) -- same shape as train_dqn's history, so it can
    be printed / plotted with the same helpers. "arrived" matters
    separately from "not crashed and not off-road": an episode can survive
    to the horizon (neither failure) without ever actually reaching the
    destination, and that's not success either.
    """
    results: Dict[str, List[float]] = {
        "episode_reward": [],
        "episode_length": [],
        "crashed": [],
        "out_of_road": [],
        "arrived": [],
    }

    for ep in range(1, num_episodes + 1):
        obs, info = env.reset()
        state = flatten_obs(obs)

        ep_reward = 0.0
        ep_steps = 0
        ep_crashed = False
        ep_out_of_road = False
        ep_arrived = False
        done = False

        while not done:
            action_idx = agent.select_action(state, greedy=True)
            action = discrete_to_continuous_action(action_idx)
            next_obs, reward, terminated, truncated, info = env.step(action)
            state = flatten_obs(next_obs)
            done = bool(terminated or truncated)

            ep_reward += reward
            ep_steps += 1
            if info.get("crash_vehicle", False) or info.get("crash_object", False):
                ep_crashed = True
            if info.get("out_of_road", False):
                ep_out_of_road = True
            if info.get("arrive_dest", False):
                ep_arrived = True

            if max_steps_per_episode is not None and ep_steps >= max_steps_per_episode:
                break

        results["episode_reward"].append(ep_reward)
        results["episode_length"].append(ep_steps)
        results["crashed"].append(ep_crashed)
        results["out_of_road"].append(ep_out_of_road)
        results["arrived"].append(ep_arrived)

        if verbose:
            print(f"[eval] ep {ep:3d}/{num_episodes} | reward={ep_reward:8.2f} | "
                  f"steps={ep_steps:4d} | crash={ep_crashed} | off_road={ep_out_of_road} | "
                  f"arrived={ep_arrived}")

    if verbose:
        print("\n--- Evaluation summary ---")
        print(f"episodes:      {num_episodes}")
        print(f"avg reward:    {np.mean(results['episode_reward']):.2f} "
              f"(std {np.std(results['episode_reward']):.2f})")
        print(f"avg length:    {np.mean(results['episode_length']):.1f} steps")
        print(f"crash rate:    {np.mean(results['crashed']):.2%}")
        print(f"out-of-road rate: {np.mean(results['out_of_road']):.2%}")
        print(f"arrival rate:  {np.mean(results['arrived']):.2%}  <- reached the destination")
        print(f"overall failure rate (crash OR off-road): "
              f"{np.mean(np.logical_or(results['crashed'], results['out_of_road'])):.2%}")
        # neither failed nor arrived: survived to the horizon (timed out) without
        # actually finishing the route -- worth knowing about separately from failure.
        neither = np.logical_and(
            np.logical_not(np.logical_or(results['crashed'], results['out_of_road'])),
            np.logical_not(results['arrived']),
        )
        print(f"timed-out-without-arriving rate: {np.mean(neither):.2%}")

    return results


def save_evaluation(results: Dict[str, List[float]], path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(results, f, indent=2)


# ---------------------------------------------------------------------------
# 2) Discussion of the training curves (reward / length / crash rate trend)
# ---------------------------------------------------------------------------
def analyze_training_curves(history: Dict[str, List[float]], chunk_frac: float = 0.1) -> str:
    """Compare the first vs. last `chunk_frac` of training episodes.

    Returns a short human-readable summary string (also printed) that you
    can quote / paraphrase in your reflection answers.
    """
    n = len(history["episode_reward"])
    k = max(1, int(n * chunk_frac))

    first_reward = np.mean(history["episode_reward"][:k])
    last_reward = np.mean(history["episode_reward"][-k:])

    first_len = np.mean(history["episode_length"][:k])
    last_len = np.mean(history["episode_length"][-k:])

    first_crash = np.mean(history["crashed"][:k])
    last_crash = np.mean(history["crashed"][-k:])

    has_off_road = "out_of_road" in history
    if has_off_road:
        first_off_road = np.mean(history["out_of_road"][:k])
        last_off_road = np.mean(history["out_of_road"][-k:])

    has_arrived = "arrived" in history
    if has_arrived:
        first_arrived = np.mean(history["arrived"][:k])
        last_arrived = np.mean(history["arrived"][-k:])

    reward_delta = last_reward - first_reward
    len_delta = last_len - first_len
    crash_delta = last_crash - first_crash

    lines = [
        f"Training curve summary (first {k} vs. last {k} of {n} episodes):",
        f"  reward:     {first_reward:8.2f} -> {last_reward:8.2f}  "
        f"({'+' if reward_delta >= 0 else ''}{reward_delta:.2f})",
        f"  length:     {first_len:8.1f} -> {last_len:8.1f}  "
        f"({'+' if len_delta >= 0 else ''}{len_delta:.1f} steps)",
        f"  crash rate: {first_crash:8.2%} -> {last_crash:8.2%}  "
        f"({'+' if crash_delta >= 0 else ''}{crash_delta:.2%})",
    ]
    if has_off_road:
        off_road_delta = last_off_road - first_off_road
        lines.append(
            f"  out-of-road rate: {first_off_road:8.2%} -> {last_off_road:8.2%}  "
            f"({'+' if off_road_delta >= 0 else ''}{off_road_delta:.2%})"
        )
    else:
        lines.append(
            "  out-of-road rate: not available in this history file "
            "(re-run training with the updated train.py to track it)"
        )
    if has_arrived:
        arrived_delta = last_arrived - first_arrived
        lines.append(
            f"  arrival rate:     {first_arrived:8.2%} -> {last_arrived:8.2%}  "
            f"({'+' if arrived_delta >= 0 else ''}{arrived_delta:.2%})  <- reached the destination"
        )
    else:
        lines.append(
            "  arrival rate: not available in this history file "
            "(re-run training with the updated train.py to track it)"
        )

    lines += [
        "",
        "Interpretation (numbers only -- write your own reflection wording):",
        f"  - Reward {'increased' if reward_delta > 0 else 'decreased' if reward_delta < 0 else 'stayed flat'} "
        f"from early to late training.",
        f"  - Episode length {'increased' if len_delta > 0 else 'decreased' if len_delta < 0 else 'stayed flat'} "
        f"(longer episodes usually mean the agent survives longer before a crash/timeout).",
        f"  - Crash rate {'increased' if crash_delta > 0 else 'decreased' if crash_delta < 0 else 'stayed flat'} "
        f"from early to late training.",
    ]
    if has_arrived:
        lines.append(
            f"  - Arrival rate {'increased' if arrived_delta > 0 else 'decreased' if arrived_delta < 0 else 'stayed flat'} "
            f"from early to late training -- this is the metric that actually answers "
            f"'does the car get where it's going', separate from just 'didn't crash/leave the road'."
        )
    summary = "\n".join(lines)
    print(summary)
    return summary


# ---------------------------------------------------------------------------
# 3) Visualization: top-down video (headless) and 3D interactive window
# ---------------------------------------------------------------------------
def record_topdown_video(
    agent: DQNAgent,
    out_path: str = "models/eval_topdown.mp4",
    num_scenarios: int = 5,
    max_steps: int = 500,
    fps: int = 30,
) -> None:
    """Run one greedy episode and save a top-down view video.

    Does not need an on-screen window -- MetaDrive's top-down renderer
    returns a numpy image per frame that we accumulate and save.
    """
    try:
        import imageio
    except ImportError:
        raise RuntimeError(
            "imageio is required for video export. Install with: pip install imageio imageio-ffmpeg"
        )

    env = create_metadrive_env(config_overrides={"num_scenarios": num_scenarios, "horizon": max_steps})
    frames = []
    try:
        obs, info = env.reset()
        state = flatten_obs(obs)
        for _ in range(max_steps):
            frame = env.render(mode="topdown", screen_size=(700, 700))
            if frame is not None:
                frames.append(frame)

            action_idx = agent.select_action(state, greedy=True)
            action = discrete_to_continuous_action(action_idx)
            obs, reward, terminated, truncated, info = env.step(action)
            state = flatten_obs(obs)

            if terminated or truncated:
                break
    finally:
        env.close()

    if not frames:
        print("No frames captured -- check that this MetaDrive version supports "
              "env.render(mode='topdown') with the given arguments.")
        return

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    imageio.mimsave(out_path, frames, fps=fps)
    print(f"Saved top-down video ({len(frames)} frames) -> {out_path}")


def run_3d_visualization(
    agent: DQNAgent,
    num_episodes: int = 1,
    max_steps: int = 1000,
    screenshot_every: Optional[int] = 100,
    screenshot_dir: str = "models/eval_screenshots",
) -> None:
    """Open MetaDrive's interactive 3D window and drive with the trained agent.

    This opens a live window (use_render=True) so you can *watch* the
    agent -- there's no separate video file produced automatically, but a
    few screenshots are grabbed along the way as submission evidence.
    Close the window (or let the episode end) to move to the next episode.
    """
    env = create_metadrive_env(config_overrides={"use_render": True, "horizon": max_steps})
    os.makedirs(screenshot_dir, exist_ok=True)

    try:
        for ep in range(1, num_episodes + 1):
            obs, info = env.reset()
            state = flatten_obs(obs)
            print(f"[3D] episode {ep}/{num_episodes} -- close the window or wait for episode end")

            for step in range(max_steps):
                env.render(mode="human")

                action_idx = agent.select_action(state, greedy=True)
                action = discrete_to_continuous_action(action_idx)
                obs, reward, terminated, truncated, info = env.step(action)
                state = flatten_obs(obs)

                if screenshot_every and step % screenshot_every == 0:
                    try:
                        shot_path = os.path.join(screenshot_dir, f"ep{ep}_step{step}.jpg")
                        env.engine.screenshot(shot_path)
                    except Exception as e:  # pragma: no cover - best effort only
                        print(f"  (screenshot skipped: {e})")

                if terminated or truncated:
                    break
    finally:
        env.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Task 8: evaluate + visualize the trained DQN agent.")
    parser.add_argument("--model-path", type=str, default="models/dqn_trained.pt")
    parser.add_argument("--history-path", type=str, default="models/training_history.json")

    parser.add_argument("--eval-episodes", type=int, default=20,
                         help="Number of greedy evaluation episodes to run (0 to skip).")
    parser.add_argument("--eval-out", type=str, default="models/eval_results.json")

    parser.add_argument("--plot-only", action="store_true",
                         help="Re-plot training curves + print the trend summary from training_history.json.")
    parser.add_argument("--plot-out", type=str, default="models/training_curves.png")

    parser.add_argument("--topdown-video", action="store_true", help="Record a top-down evaluation video.")
    parser.add_argument("--video-out", type=str, default="models/eval_topdown.mp4")
    parser.add_argument("--video-steps", type=int, default=500)

    parser.add_argument("--show-3d", action="store_true", help="Open the interactive 3D view.")
    parser.add_argument("--show-3d-episodes", type=int, default=1)

    parser.add_argument(
        "--device", type=str, default=None, choices=["cuda", "cpu"],
        help="Force the device used to run the loaded agent. Default: auto-detect "
             "('cuda' if available, else 'cpu'), same as during training.",
    )
    parser.add_argument(
        "--traffic-density", type=float, default=None,
        help="MUST match whatever --traffic-density the checkpoint was trained with "
             "(e.g. 0.0 for a Phase-1 no-traffic checkpoint), or 'crash' failures here "
             "reflect traffic the model was never trained to handle, not a real measure "
             "of the checkpoint's driving skill. Default (omit): configs/env_config.py's "
             "default (0.1).",
    )
    parser.add_argument(
        "--custom-reward", action="store_true",
        help="MUST match whatever --custom-reward setting train.py used for this "
             "checkpoint. The episode_reward numbers reported here come from whichever "
             "reward function is active, so evaluating with the wrong one gives numbers "
             "that aren't comparable to what the checkpoint was actually trained/selected "
             "against (crash/off-road/arrival rates are unaffected either way -- those "
             "come from info flags, not the reward).",
    )

    args = parser.parse_args()

    cuda_available = torch.cuda.is_available()
    print(f"torch: {torch.__version__} | CUDA available: {cuda_available}"
          + (f" | GPU: {torch.cuda.get_device_name(0)}" if cuda_available else ""))
    if args.device == "cuda" and not cuda_available:
        raise RuntimeError(
            "--device cuda was requested but torch.cuda.is_available() is False. "
            "Reinstall torch with a CUDA build (see requirements.txt) and check "
            "your GPU driver."
        )

    # --- load the trained agent -------------------------------------------------
    env_overrides = {"num_scenarios": 20, "horizon": 1000}
    if args.traffic_density is not None:
        env_overrides["traffic_density"] = args.traffic_density
    print(f"Env overrides: {env_overrides}")
    if args.custom_reward:
        env_overrides.update(CUSTOM_REWARD_BASE_OVERRIDES)
    env = create_metadrive_env(config_overrides=env_overrides)
    if args.custom_reward:
        env = SpeedCapRewardWrapper(env)
        print("[--custom-reward] env wrapped with SpeedCapRewardWrapper for evaluation, "
              "matching the reward this checkpoint should have been trained with.")
    try:
        obs, info = env.reset()
        state_size = flatten_obs(obs).shape[0]
        agent = DQNAgent(state_size=state_size, action_size=NUM_DISCRETE_ACTIONS, device=args.device)
        agent.load(args.model_path)
        agent.epsilon = 0.0  # purely greedy for evaluation
        print(f"Loaded trained agent from {args.model_path}")

        if args.eval_episodes > 0:
            print("\n" + "=" * 70)
            print("Evaluating trained policy (greedy)")
            print("=" * 70)
            results = evaluate_policy(agent, env, num_episodes=args.eval_episodes)
            save_evaluation(results, args.eval_out)
            print(f"Saved evaluation results -> {args.eval_out}")
    finally:
        env.close()

    # --- training curve plots + discussion --------------------------------------
    if args.plot_only or os.path.exists(args.history_path):
        if os.path.exists(args.history_path):
            print("\n" + "=" * 70)
            print("Training curve analysis")
            print("=" * 70)
            with open(args.history_path) as f:
                history = json.load(f)
            plot_history(history, args.plot_out)
            print(f"Saved training curves -> {args.plot_out}")
            analyze_training_curves(history)
        else:
            print(f"\n(no training history found at {args.history_path}, skipping plots)")

    # --- visualization ------------------------------------------------------------
    if args.topdown_video:
        print("\n" + "=" * 70)
        print("Recording top-down visualization")
        print("=" * 70)
        record_topdown_video(agent, out_path=args.video_out, max_steps=args.video_steps)

    if args.show_3d:
        print("\n" + "=" * 70)
        print("Opening 3D visualization")
        print("=" * 70)
        run_3d_visualization(agent, num_episodes=args.show_3d_episodes)


if __name__ == "__main__":
    main()
