"""
Task 4: Implement the Experience Replay Buffer
==================================================

Stores transitions (s_t, a_t, r_t, s_{t+1}, d_t) and supports:
  - adding new transitions                (push)
  - randomly sampling mini-batches         (sample)
  - returning sampled data as PyTorch tensors
  - reporting how many transitions are stored (__len__)
"""

import random
from collections import deque
from typing import Tuple

import numpy as np
import torch


class ReplayBuffer:
    """Fixed-capacity FIFO replay buffer for DQN.

    Once `capacity` is reached, the oldest transitions are dropped
    automatically (via `collections.deque(maxlen=capacity)`).
    """

    def __init__(self, capacity: int = 100_000, seed: int = None):
        self.capacity = capacity
        self.buffer: deque = deque(maxlen=capacity)
        if seed is not None:
            random.seed(seed)

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """Add one transition (s_t, a_t, r_t, s_{t+1}, d_t) to the buffer.

        `state`/`next_state` are expected to already be flattened
        (i.e. the output of `utils.processing.flatten_obs`).
        """
        # store as compact numpy dtypes to keep memory usage down
        transition = (
            np.asarray(state, dtype=np.float32),
            int(action),
            float(reward),
            np.asarray(next_state, dtype=np.float32),
            bool(done),
        )
        self.buffer.append(transition)

    def sample(
        self, batch_size: int, device: str = "cpu"
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Randomly sample a mini-batch of transitions as PyTorch tensors.

        Returns:
            states:      (batch_size, state_size)  float32
            actions:     (batch_size, 1)            int64
            rewards:     (batch_size, 1)            float32
            next_states: (batch_size, state_size)  float32
            dones:       (batch_size, 1)            float32  (1.0 if episode ended, else 0.0)
        """
        if len(self.buffer) < batch_size:
            raise ValueError(
                f"Cannot sample batch_size={batch_size}, only "
                f"{len(self.buffer)} transitions stored."
            )

        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states_t = torch.as_tensor(np.stack(states), dtype=torch.float32, device=device)
        actions_t = torch.as_tensor(actions, dtype=torch.int64, device=device).unsqueeze(1)
        rewards_t = torch.as_tensor(rewards, dtype=torch.float32, device=device).unsqueeze(1)
        next_states_t = torch.as_tensor(
            np.stack(next_states), dtype=torch.float32, device=device
        )
        dones_t = torch.as_tensor(dones, dtype=torch.float32, device=device).unsqueeze(1)

        return states_t, actions_t, rewards_t, next_states_t, dones_t

    def __len__(self) -> int:
        """Number of transitions currently stored."""
        return len(self.buffer)


if __name__ == "__main__":
    # Quick manual test: `python -m replay_buffer`
    buf = ReplayBuffer(capacity=1000, seed=0)
    state_size = 10

    for i in range(50):
        s = np.random.randn(state_size)
        a = np.random.randint(0, 6)
        r = np.random.randn()
        s2 = np.random.randn(state_size)
        d = bool(i % 17 == 0)
        buf.push(s, a, r, s2, d)

    print(f"Buffer size after 50 pushes: {len(buf)}")
    states, actions, rewards, next_states, dones = buf.sample(16)
    print("states:", states.shape, states.dtype)
    print("actions:", actions.shape, actions.dtype)
    print("rewards:", rewards.shape, rewards.dtype)
    print("next_states:", next_states.shape, next_states.dtype)
    print("dones:", dones.shape, dones.dtype)
