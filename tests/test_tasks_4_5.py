"""
Sanity test for Task 4 (ReplayBuffer) + Task 5 (DQNAgent), wired into the
real MetaDrive environment (Tasks 1-3).

Run with:
    cd metadrive_dqn
    python -m tests.test_tasks_4_5

This is NOT a real training run (that's Task 6/7) — it's a short loop that:
  1. Creates the env (Task 1) and the DQNAgent (Task 5, which internally
     builds a ReplayBuffer from Task 4).
  2. Plays a handful of steps with epsilon-greedy actions, storing
     transitions in the replay buffer.
  3. Once enough transitions are collected, calls agent.learn() repeatedly
     and checks that the loss is a finite, sane number and that the target
     network gets synced after `target_update_freq` learn() calls.

If it prints "ALL CHECKS PASSED for Task 4, 5." at the end, you're safe to
move on to Task 6 (the real training loop).
"""

import numpy as np
import torch
import tempfile
import os

from configs.env_config import create_metadrive_env
from utils.processing import flatten_obs, discrete_to_continuous_action
from dqn_agent import DQNAgent


def main():
    print("=" * 70)
    print("Setting up env + agent")
    print("=" * 70)
    env = create_metadrive_env(config_overrides={"num_scenarios": 5, "horizon": 200})

    try:
        obs, info = env.reset()
        state = flatten_obs(obs)
        state_size = state.shape[0]
        action_size = 6

        agent = DQNAgent(
            state_size=state_size,
            action_size=action_size,
            hidden_size=128,
            buffer_capacity=5000,
            batch_size=32,
            gamma=0.99,
            lr=1e-3,
            epsilon_start=1.0,
            epsilon_end=0.05,
            epsilon_decay=0.99,
            target_update_freq=5,  # small on purpose, just to exercise the sync logic here
            seed=0,
        )
        print(f"state_size={state_size}, action_size={action_size}, device={agent.device}")

        print("\n" + "=" * 70)
        print("Collecting transitions with epsilon-greedy actions")
        print("=" * 70)
        n_collect_steps = 200
        for step in range(n_collect_steps):
            action_idx = agent.select_action(state)
            action = discrete_to_continuous_action(action_idx)

            next_obs, reward, terminated, truncated, info = env.step(action)
            next_state = flatten_obs(next_obs)
            done = terminated or truncated

            agent.store_transition(state, action_idx, reward, next_state, done)
            state = next_state

            if done:
                obs, info = env.reset()
                state = flatten_obs(obs)

        print(f"Replay buffer size after {n_collect_steps} steps: {len(agent.replay_buffer)}")
        assert len(agent.replay_buffer) == n_collect_steps

        print("\n" + "=" * 70)
        print("Running learn() steps")
        print("=" * 70)
        target_before = {
            k: v.clone() for k, v in agent.target_network.state_dict().items()
        }

        losses = []
        for i in range(30):
            loss = agent.learn()
            assert loss is not None, "learn() returned None even though buffer is full enough"
            assert np.isfinite(loss), f"loss is not finite: {loss}"
            losses.append(loss)
            print(f"  learn step {i}: loss = {loss:.5f}")

        target_after = agent.target_network.state_dict()
        changed = any(
            not torch.equal(target_before[k], target_after[k]) for k in target_before
        )
        assert changed, "target network was never updated despite target_update_freq being hit"
        print("Target network sync: OK (weights changed after target_update_freq learn() calls)")

        print("\n" + "=" * 70)
        print("Epsilon decay + save/load")
        print("=" * 70)
        eps_before = agent.epsilon
        agent.decay_epsilon()
        assert agent.epsilon <= eps_before
        print(f"epsilon: {eps_before:.3f} -> {agent.epsilon:.3f}")

        save_path = os.path.join(tempfile.gettempdir(), "dqn_agent_test.pt")
        agent.save(save_path)
        agent.load(save_path)
        print(f"save/load OK (checkpoint at {save_path})")

        print("\nALL CHECKS PASSED for Task 4, 5.")

    finally:
        env.close()


if __name__ == "__main__":
    main()
