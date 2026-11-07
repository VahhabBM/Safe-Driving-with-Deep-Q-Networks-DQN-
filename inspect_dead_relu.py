"""
Diagnostic: is the trained Q-network's hidden layer "dead" (all ReLU
units output 0 regardless of input), causing a constant output vector?

Run with:
    python -m inspect_dead_relu
"""
import argparse

import numpy as np
import torch

from dqn_agent import DQNAgent
from configs.env_config import create_metadrive_env
from utils.processing import flatten_obs, NUM_DISCRETE_ACTIONS


def main():
    parser = argparse.ArgumentParser(description="Inspect the trained Q-network for dead ReLU units.")
    parser.add_argument("--model-path", type=str, default="models/dqn_trained.pt")
    parser.add_argument(
        "--device", type=str, default=None, choices=["cuda", "cpu"],
        help="Force the device the network runs on. Default: auto-detect.",
    )
    parser.add_argument(
        "--traffic-density", type=float, default=None,
        help="MUST match whatever --traffic-density the checkpoint was trained with, "
             "or the sampled 'real states' won't reflect what the checkpoint actually "
             "saw during training. Default (omit this flag): configs/env_config.py's default (0.1).",
    )
    args = parser.parse_args()

    cuda_available = torch.cuda.is_available()
    print(f"torch: {torch.__version__} | CUDA available: {cuda_available}"
          + (f" | GPU: {torch.cuda.get_device_name(0)}" if cuda_available else ""))

    env_overrides = {"num_scenarios": 20, "horizon": 50}
    if args.traffic_density is not None:
        env_overrides["traffic_density"] = args.traffic_density
    print(f"Env overrides: {env_overrides}")

    env = create_metadrive_env(config_overrides=env_overrides)
    try:
        obs, info = env.reset()
        state_size = flatten_obs(obs).shape[0]
        agent = DQNAgent(state_size=state_size, action_size=NUM_DISCRETE_ACTIONS, device=args.device)
        agent.load(args.model_path)
        net = agent.q_network
        net.eval()

        # Collect a batch of REAL, DIFFERENT states from a live episode.
        states = [flatten_obs(obs)]
        for _ in range(31):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            states.append(flatten_obs(obs))
            if terminated or truncated:
                obs, info = env.reset()
        states = np.stack(states)  # (32, state_size)

        # Also throw in pure random noise states, just to compare.
        random_states = np.random.randn(32, state_size).astype(np.float32)

        # Hook into every Linear layer's output (pre- and post-ReLU) to see
        # how many units are "dead" (always <= 0, i.e. always 0 after ReLU).
        activations = {}

        def make_hook(name):
            def hook(module, inp, out):
                activations[name] = out.detach().cpu().numpy()
            return hook

        hooks = []
        linear_layers = [(name, module) for name, module in net.named_modules()
                          if isinstance(module, torch.nn.Linear)]
        # The last Linear layer is the raw Q-value output (no ReLU after it) --
        # "dead unit" only makes sense for the two *hidden* layers before it.
        hidden_linear_layers = linear_layers[:-1]
        for name, module in hidden_linear_layers:
            hooks.append(module.register_forward_hook(make_hook(name)))

        print("=== Forward pass on 32 REAL (different) states from the env ===")
        states_for_net = states
        if agent.obs_normalizer is not None:
            states_for_net = np.stack([agent.obs_normalizer.normalize(s) for s in states])
        with torch.no_grad():
            real_q = net(
                torch.as_tensor(states_for_net, dtype=torch.float32, device=agent.device)
            ).cpu().numpy()
        for name, act in activations.items():
            dead_frac = np.mean(np.all(act <= 0, axis=0))
            print(f"layer '{name}': output shape {act.shape}, "
                  f"fraction of ALWAYS-ZERO units across the batch: {dead_frac:.1%}")

        print(f"\nQ-value range across these 32 real states: "
              f"min={real_q.min():.4f}, max={real_q.max():.4f}, std={real_q.std():.6f}")
        print(f"Are all 32 rows identical? {np.allclose(real_q, real_q[0], atol=1e-4)}")

        print("\n=== Forward pass on 32 RANDOM NOISE states (sanity comparison) ===")
        activations.clear()
        with torch.no_grad():
            noise_q = net(
                torch.as_tensor(random_states, dtype=torch.float32, device=agent.device)
            ).cpu().numpy()
        for name, act in activations.items():
            dead_frac = np.mean(np.all(act <= 0, axis=0))
            print(f"layer '{name}': fraction of ALWAYS-ZERO units: {dead_frac:.1%}")
        print(f"Q-value range on random noise: min={noise_q.min():.4f}, max={noise_q.max():.4f}, "
              f"std={noise_q.std():.6f}")
        print(f"Are all 32 rows identical (even for pure random noise input)? "
              f"{np.allclose(noise_q, noise_q[0], atol=1e-4)}")

        for h in hooks:
            h.remove()

        print("\n--- Interpretation ---")
        print("If 'fraction of ALWAYS-ZERO units' is high (e.g. >90%) in BOTH the real")
        print("and random-noise cases, the hidden layers have collapsed ('dead ReLU'):")
        print("the network ignores its input almost entirely and just outputs learned")
        print("bias values. This explains why the greedy action never changes.")

    finally:
        env.close()


if __name__ == "__main__":
    main()
