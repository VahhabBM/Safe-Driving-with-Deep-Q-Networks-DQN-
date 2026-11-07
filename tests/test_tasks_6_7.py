"""
Sanity test for Task 6 (Training Loop) + Task 7 (Train the Agent).

Run with:
    cd metadrive_dqn
    python -m tests.test_tasks_6_7

This is a SMALL, FAST version of the real training run -- just enough to
prove that `train_dqn(...)` (Task 6) is wired correctly end-to-end with the
real MetaDrive env + DQNAgent, and that `agent.save(...)` (Task 7) produces
a checkpoint that can be reloaded into a fresh agent and used for greedy
inference. It is NOT the actual 500-1000 episode training run -- run
`python -m train --episodes 800` for that.

If it prints "ALL CHECKS PASSED for Task 6, 7." at the end, the training
loop + save/load pipeline are correct and you're ready to kick off the
real, long run.
"""

import os
import tempfile

import numpy as np
import torch

# Import order matters on some Windows setups: torch (via dqn_agent) must
# load before metadrive/panda3d, otherwise torch's native DLLs can fail to
# load. See the matching note in train.py.
from dqn_agent import DQNAgent
from configs.env_config import create_metadrive_env
from utils.processing import flatten_obs
from train import train_dqn, save_history, build_agent


def main():
    print("=" * 70)
    print("Setting up env + agent for a short Task 6/7 smoke run")
    print("=" * 70)
    env = create_metadrive_env(config_overrides={"num_scenarios": 5, "horizon": 200})

    try:
        obs, info = env.reset()
        state_size = flatten_obs(obs).shape[0]
        agent = build_agent(state_size=state_size, action_size=6, seed=0)
        # make the smoke test fast: learn almost immediately, sync target often
        agent.target_update_freq = 20
        print(f"state_size={state_size}, device={agent.device}")

        print("\n" + "=" * 70)
        print("TASK 6: running train_dqn() for a handful of episodes")
        print("=" * 70)
        n_episodes = 6
        history = train_dqn(
            agent=agent,
            env=env,
            num_episodes=n_episodes,
            min_buffer_size=50,      # small on purpose so learn() actually fires
            max_steps_per_episode=100,
            checkpoint_path=None,
            log_every=1,
        )

        # --- structural checks on the history dict --------------------------
        for key in ("episode_reward", "episode_length", "crashed", "avg_loss", "epsilon"):
            assert key in history, f"history missing key: {key}"
            assert len(history[key]) == n_episodes, (
                f"history['{key}'] has {len(history[key])} entries, expected {n_episodes}"
            )

        assert all(np.isfinite(r) for r in history["episode_reward"]), "non-finite reward found"
        assert all(isinstance(s, int) and s > 0 for s in history["episode_length"]), \
            "episode_length should be positive ints"
        assert all(isinstance(c, bool) for c in history["crashed"]), "crashed should be bools"
        assert all(0.0 <= e <= 1.0 for e in history["epsilon"]), "epsilon out of [0, 1]"
        # epsilon should have decayed at least once across 6 episodes
        assert history["epsilon"][-1] <= history["epsilon"][0], "epsilon never decreased"
        # buffer should have grown by roughly sum(episode_length)
        assert len(agent.replay_buffer) == sum(history["episode_length"]), (
            "replay buffer size doesn't match total steps collected"
        )
        # at least one learn() call should have produced a finite avg_loss
        assert any(np.isfinite(l) for l in history["avg_loss"]), \
            "learn() never ran (avg_loss all NaN) -- check min_buffer_size wiring"

        print("Task 6 structural checks: OK")
        print(f"  episode_reward: {[f'{r:.2f}' for r in history['episode_reward']]}")
        print(f"  episode_length: {history['episode_length']}")
        print(f"  crashed:        {history['crashed']}")
        print(f"  avg_loss:       {[f'{l:.4f}' if np.isfinite(l) else 'nan' for l in history['avg_loss']]}")
        print(f"  epsilon:        {[f'{e:.3f}' for e in history['epsilon']]}")

        print("\n" + "=" * 70)
        print("TASK 7: save trained model + history, reload into a fresh agent")
        print("=" * 70)
        tmp_dir = tempfile.mkdtemp()
        model_path = os.path.join(tmp_dir, "dqn_trained.pt")
        history_path = os.path.join(tmp_dir, "training_history.json")

        agent.save(model_path)
        assert os.path.exists(model_path), "agent.save() did not create the checkpoint file"
        save_history(history, history_path)
        assert os.path.exists(history_path), "save_history() did not create the JSON file"

        # reload into a brand-new agent and confirm weights match + greedy inference works
        fresh_agent = build_agent(state_size=state_size, action_size=6, seed=1)
        fresh_agent.load(model_path)

        for k, v in agent.q_network.state_dict().items():
            assert torch.equal(v, fresh_agent.q_network.state_dict()[k]), \
                f"reloaded weights differ for {k}"

        raw_obs, info = env.reset()
        state = flatten_obs(raw_obs)
        greedy_action_idx = fresh_agent.select_action(state, greedy=True)
        assert 0 <= greedy_action_idx < 6, "reloaded agent produced an invalid action index"

        print(f"Model checkpoint OK: {model_path}")
        print(f"History JSON OK:     {history_path}")
        print(f"Reloaded agent greedy action on a fresh reset: {greedy_action_idx}")

        print("\nALL CHECKS PASSED for Task 6, 7.")

    finally:
        env.close()


if __name__ == "__main__":
    main()
