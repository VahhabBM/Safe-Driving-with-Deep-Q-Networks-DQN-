"""
Diagnostic: is the trained policy suffering from "action collapse"
(always picking the same action regardless of state)?

Run with:
    python -m inspect_policy

For a handful of real states sampled from a live episode, this prints the
full Q-value vector (all NUM_DISCRETE_ACTIONS actions) and the chosen greedy
action. If the argmax action is (almost) always the same AND the Q-values
barely change between very different states (e.g. approaching a straight
vs. approaching a sharp curve), that confirms action collapse rather than
a code bug.
"""
import argparse

import numpy as np

from dqn_agent import DQNAgent
from configs.env_config import create_metadrive_env
from utils.processing import flatten_obs, discrete_to_continuous_action, ACTION_MEANINGS, NUM_DISCRETE_ACTIONS


def main():
    parser = argparse.ArgumentParser(description="Inspect the trained policy for action collapse.")
    parser.add_argument("--model-path", type=str, default="models/dqn_trained.pt")
    parser.add_argument(
        "--device", type=str, default=None, choices=["cuda", "cpu"],
        help="Force the device the network runs on. Default: auto-detect.",
    )
    parser.add_argument(
        "--traffic-density", type=float, default=None,
        help="MUST match whatever --traffic-density the checkpoint was trained with "
             "(e.g. 0.0 for a Phase-1 no-traffic checkpoint), or you'll see crashes/"
             "behavior the model was never trained to handle, which is a confound, "
             "not a real measurement of what the checkpoint actually learned. "
             "Default (omit this flag): configs/env_config.py's default (0.1).",
    )
    args = parser.parse_args()

    env_overrides = {"num_scenarios": 20, "horizon": 300}
    if args.traffic_density is not None:
        env_overrides["traffic_density"] = args.traffic_density
    print(f"Env overrides: {env_overrides}")

    env = create_metadrive_env(config_overrides=env_overrides)
    try:
        obs, info = env.reset()
        state_size = flatten_obs(obs).shape[0]
        agent = DQNAgent(state_size=state_size, action_size=NUM_DISCRETE_ACTIONS, device=args.device)
        agent.load(args.model_path)

        action_counts = {i: 0 for i in range(NUM_DISCRETE_ACTIONS)}
        state = flatten_obs(obs)

        # Header + per-row Q-value formatting are generated from
        # ACTION_MEANINGS so this keeps working whatever NUM_DISCRETE_ACTIONS
        # is (10 now, was 6) instead of hardcoding 6 column labels.
        header_cols = " ".join(f"{('Q(' + ACTION_MEANINGS[i] + ')'):>22}" for i in range(NUM_DISCRETE_ACTIONS))
        print(f"{'step':>4} | {header_cols} | chosen")

        import torch
        for step in range(150):
            net_state = state
            if agent.obs_normalizer is not None:
                net_state = agent.obs_normalizer.normalize(state)
            state_t = torch.as_tensor(net_state, dtype=torch.float32, device=agent.device).unsqueeze(0)
            with torch.no_grad():
                q_values = agent.q_network(state_t).cpu().numpy().flatten()
            action_idx = int(np.argmax(q_values))
            action_counts[action_idx] += 1

            if step % 10 == 0:
                q_str = " ".join(f"{q:22.3f}" for q in q_values)
                print(f"{step:4d} | {q_str} | {ACTION_MEANINGS[action_idx]}")

            action = discrete_to_continuous_action(action_idx)
            obs, reward, terminated, truncated, info = env.step(action)
            state = flatten_obs(obs)
            if terminated or truncated:
                print(f"  (episode ended at step {step}, reason: "
                      f"{'crash' if info.get('crash_vehicle') else 'out_of_road' if info.get('out_of_road') else 'other'})")
                obs, info = env.reset()
                state = flatten_obs(obs)

        print("\nAction usage over 150 greedy steps:")
        for idx, count in action_counts.items():
            print(f"  {ACTION_MEANINGS[idx]:20s}: {count:4d} times ({count/150:.1%})")

    finally:
        env.close()


if __name__ == "__main__":
    main()
