"""
Sanity test for Task 1 + Task 2 + Task 3 together.

Run with:
    cd metadrive_dqn
    python -m tests.test_tasks_1_2_3

What it checks:
  1. (Task 1) The MetaDriveEnv can be created, reset, and stepped.
  2. (Task 2) flatten_obs() turns the raw observation into a 1D float32
     vector, and discrete_to_continuous_action() maps every one of the
     6 discrete actions to a valid 2D continuous MetaDrive action.
  3. (Task 3) DQNNetwork takes that flattened state and outputs exactly
     6 Q-values (one per discrete action), for both a single state and a
     batch of states.

If everything prints "OK" / "PASSED" at the end, Task 1-3 are wired up
correctly and it's safe to move on to Task 4 (Replay Buffer).
"""

import numpy as np
import torch

from configs.env_config import create_metadrive_env
from utils.processing import (
    flatten_obs,
    discrete_to_continuous_action,
    NUM_DISCRETE_ACTIONS,
    ACTION_MEANINGS,
)
from models.dqn_network import DQNNetwork


def main():
    print("=" * 70)
    print("TASK 1: creating + resetting + stepping the MetaDrive environment")
    print("=" * 70)
    env = create_metadrive_env(config_overrides={"num_scenarios": 5, "horizon": 200})

    try:
        raw_obs, info = env.reset()
        print(f"Raw obs type: {type(raw_obs)}, "
              f"shape: {getattr(raw_obs, 'shape', 'n/a')}")

        print("\n" + "=" * 70)
        print("TASK 2: flatten_obs + discrete_to_continuous_action")
        print("=" * 70)
        state = flatten_obs(raw_obs)
        state_size = state.shape[0]
        assert state.dtype == np.float32
        assert state.ndim == 1
        print(f"Flattened state shape: {state.shape}, dtype: {state.dtype}")

        for idx in range(NUM_DISCRETE_ACTIONS):
            cont_action = discrete_to_continuous_action(idx)
            assert cont_action.shape == (2,)
            assert cont_action.dtype == np.float32
            assert (-1.0 <= cont_action).all() and (cont_action <= 1.0).all()
            print(f"  action {idx} ({ACTION_MEANINGS[idx]:>18s}) -> {cont_action}")

        print("\n" + "=" * 70)
        print("TASK 3: DQNNetwork forward pass")
        print("=" * 70)
        net = DQNNetwork(state_size=state_size, action_size=NUM_DISCRETE_ACTIONS)

        state_tensor = torch.from_numpy(state)
        q_values = net(state_tensor)
        assert q_values.shape == (NUM_DISCRETE_ACTIONS,)
        print(f"Single-state Q-values shape: {q_values.shape} -> {q_values.detach().numpy()}")

        batch = torch.randn(16, state_size)
        q_batch = net(batch)
        assert q_batch.shape == (16, NUM_DISCRETE_ACTIONS)
        print(f"Batch Q-values shape: {q_batch.shape}")

        print("\n" + "=" * 70)
        print("END-TO-END: step the real env using the network's greedy action")
        print("=" * 70)
        for step in range(10):
            with torch.no_grad():
                q = net(torch.from_numpy(state))
            action_idx = int(torch.argmax(q).item())
            action = discrete_to_continuous_action(action_idx)

            raw_next_obs, reward, terminated, truncated, info = env.step(action)
            state = flatten_obs(raw_next_obs)

            print(f"  step {step}: action_idx={action_idx} reward={reward:.3f} "
                  f"terminated={terminated} truncated={truncated} "
                  f"crash={info.get('crash_vehicle', info.get('crash', 'n/a'))}")

            if terminated or truncated:
                raw_obs, info = env.reset()
                state = flatten_obs(raw_obs)

        print("\nALL CHECKS PASSED for Task 1, 2, 3.")

    finally:
        env.close()


if __name__ == "__main__":
    main()
