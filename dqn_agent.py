"""
Task 5: Implement the DQN Agent
==================================

DQNAgent owns:
  - a Q-network                (models.dqn_network.DQNNetwork)
  - a target network            (same architecture, periodically synced)
  - an optimizer                (Adam)
  - a replay buffer              (replay_buffer.ReplayBuffer)

Behavior:
  - epsilon-greedy action selection during training
  - learning from sampled replay-buffer mini-batches using a DOUBLE DQN
    Bellman target:  y = r + (1 - d) * gamma * Q_target(s', argmax_a' Q_online(s', a'))
    (decouples action-selection from action-evaluation to curb the
    systematic Q-value overestimation vanilla DQN's max-of-target-net
    target is prone to, especially with large-magnitude shaped rewards)
  - reward clipping (to [reward_clip_min, reward_clip_max], Atari-DQN-style,
    but ASYMMETRIC by default: a much larger positive ceiling than negative
    floor, so a big one-time "reached the destination" bonus isn't squashed
    down to the same size as an ordinary crash/off-road penalty)
    before transitions are stored, as a guard against Q-value divergence
    from large terminal rewards/penalties
  - Huber loss (SmoothL1Loss) between Q(s, a) and y, plus gradient-norm
    clipping -- both are standard DQN stability techniques that prevent
    an occasional large TD-error from producing a gradient large enough
    to push ReLU biases so negative the hidden layers "die" (i.e. the
    network's output collapses to a constant regardless of input state).
  - periodic target-network updates (hard update every N learning steps)
"""

import random
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from models.dqn_network import DQNNetwork
from replay_buffer import ReplayBuffer


class RunningNormalizer:
    """Online mean/variance tracker (Welford's algorithm) for input states.

    Not part of the DQN architecture itself -- just a standard input
    preprocessing step (used in most real RL pipelines, e.g. Stable-
    Baselines' VecNormalize) that keeps every feature on a similar scale.
    This directly targets the "dead ReLU" collapse diagnosed via
    inspect_dead_relu.py: without normalization, a handful of large-scale
    state features can dominate the pre-activation sum, making the same
    (learned) bias push almost every hidden unit negative for the real
    observation distribution while leaving the network free to react
    normally to out-of-distribution inputs (e.g. random noise) -- exactly
    the asymmetry we measured.
    """

    def __init__(self, size: int, epsilon: float = 1e-4, clip: float = 10.0):
        self.mean = np.zeros(size, dtype=np.float64)
        self.var = np.ones(size, dtype=np.float64)
        self.count = epsilon
        self.clip = clip

    def update(self, x: np.ndarray) -> None:
        x = np.asarray(x, dtype=np.float64)
        self.count += 1
        delta = x - self.mean
        self.mean += delta / self.count
        delta2 = x - self.mean
        self.var += (delta * delta2 - self.var) / self.count

    def normalize(self, x: np.ndarray) -> np.ndarray:
        std = np.sqrt(np.maximum(self.var, 1e-8))
        normed = (np.asarray(x, dtype=np.float64) - self.mean) / std
        return np.clip(normed, -self.clip, self.clip).astype(np.float32)

    def state_dict(self) -> dict:
        return {"mean": self.mean, "var": self.var, "count": self.count}

    def load_state_dict(self, d: dict) -> None:
        self.mean = np.asarray(d["mean"], dtype=np.float64)
        self.var = np.asarray(d["var"], dtype=np.float64)
        self.count = float(d["count"])


class DQNAgent:
    def __init__(
        self,
        state_size: int,
        action_size: int,
        hidden_size: int = 256,
        buffer_capacity: int = 100_000,
        batch_size: int = 64,
        gamma: float = 0.99,
        lr: float = 1e-3,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay: float = 0.995,
        target_update_freq: int = 1000,
        device: Optional[str] = None,
        seed: Optional[int] = None,
        grad_clip_norm: float = 10.0,
        normalize_obs: bool = True,
        reward_clip_min: Optional[float] = -10.0,
        reward_clip_max: Optional[float] = 50.0,
    ):
        """
        Args:
            state_size: dimensionality of the flattened observation.
            action_size: number of discrete actions (10, since utils/processing.py
                added slight-left/slight-right steering levels -- was 6 originally).
            hidden_size: width of the DQNNetwork's hidden layers.
            buffer_capacity: max transitions kept in the replay buffer.
            batch_size: mini-batch size sampled for each learning step.
            gamma: discount factor for the Bellman target.
            lr: Adam learning rate.
            epsilon_start / epsilon_end / epsilon_decay: epsilon-greedy
                exploration schedule. `epsilon` is multiplied by
                `epsilon_decay` every time `decay_epsilon()` is called
                (typically once per episode), floored at `epsilon_end`.
            target_update_freq: number of `learn()` calls between hard
                copies of the online network's weights into the target
                network.
            device: "cpu" / "cuda". Auto-detected if not given.
            seed: optional RNG seed for reproducibility.
            reward_clip_min / reward_clip_max: clip each raw reward to
                [reward_clip_min, reward_clip_max] before it's stored in the
                replay buffer (Atari-DQN-style reward clipping, but
                ASYMMETRIC here on purpose). Without a negative floor, a
                large terminal penalty (e.g. out_of_road_penalty=50)
                combined with dense per-step shaped reward can push the
                Bellman target's magnitude into a regime where bootstrapped
                Q-values diverge (grow to physically implausible values
                like 1000s, well beyond max_reward/(1-gamma)) -- this shows
                up as an arbitrary, unstable argmax that can flip between
                training runs rather than a genuinely learned preference.
                The positive ceiling is set much HIGHER than the negative
                floor by default so a big one-time success_reward (for
                reaching the destination, configs/env_config.py) still
                reads as clearly bigger than routine per-step reward,
                instead of being clipped down to the exact same size as an
                ordinary crash/off-road penalty. Set either bound to None
                to disable clipping on that side (not recommended for the
                negative side if you've raised *_penalty above defaults).
        """
        self.state_size = state_size
        self.action_size = action_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.target_update_freq = target_update_freq
        self.device = torch.device(
            device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)

        # --- Networks --------------------------------------------------
        self.q_network = DQNNetwork(state_size, action_size, hidden_size).to(self.device)
        self.target_network = DQNNetwork(state_size, action_size, hidden_size).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()  # target net is never trained directly

        # --- Optimizer & loss -------------------------------------------
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        # Huber loss (SmoothL1Loss) instead of plain MSE: MSE lets a single
        # large TD-error (which happens naturally as reward magnitudes grow
        # over training) produce a huge gradient, which can push ReLU
        # biases so negative that hidden units "die" for the normal-scale
        # real observation distribution (network output collapses to a
        # constant regardless of input). Huber loss caps the gradient
        # contribution of large errors and is what the original DQN paper
        # uses for exactly this reason.
        self.loss_fn = nn.SmoothL1Loss()

        # --- Replay buffer ------------------------------------------------
        self.replay_buffer = ReplayBuffer(capacity=buffer_capacity, seed=seed)

        self._learn_step_counter = 0
        self.grad_clip_norm = grad_clip_norm  # gradient clipping to prevent exploding updates
        self.reward_clip_min = reward_clip_min  # reward clipping to prevent Q-value divergence
        self.reward_clip_max = reward_clip_max

        self.obs_normalizer = RunningNormalizer(state_size) if normalize_obs else None

    # ------------------------------------------------------------------
    # Acting
    # ------------------------------------------------------------------
    def select_action(self, state: np.ndarray, greedy: bool = False) -> int:
        """Epsilon-greedy action selection.

        Args:
            state: flattened state, shape (state_size,).
            greedy: if True, always pick argmax_a Q(s, a) (used for
                evaluation, ignoring epsilon).

        Returns:
            Integer action index in [0, action_size - 1].
        """
        if (not greedy) and (random.random() < self.epsilon):
            if self.obs_normalizer is not None:
                self.obs_normalizer.update(state)
            return random.randrange(self.action_size)

        if self.obs_normalizer is not None:
            if not greedy:
                self.obs_normalizer.update(state)
            state = self.obs_normalizer.normalize(state)

        state_t = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        self.q_network.eval()
        with torch.no_grad():
            q_values = self.q_network(state_t)
        self.q_network.train()
        return int(torch.argmax(q_values, dim=1).item())

    def decay_epsilon(self) -> None:
        """Decay epsilon multiplicatively, floored at `epsilon_end`.
        Call this once per episode (typical usage)."""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    # ------------------------------------------------------------------
    # Replay buffer interaction
    # ------------------------------------------------------------------
    def store_transition(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        if self.reward_clip_min is not None or self.reward_clip_max is not None:
            lo = self.reward_clip_min if self.reward_clip_min is not None else -np.inf
            hi = self.reward_clip_max if self.reward_clip_max is not None else np.inf
            reward = float(np.clip(reward, lo, hi))
        self.replay_buffer.push(state, action, reward, next_state, done)

    # ------------------------------------------------------------------
    # Learning
    # ------------------------------------------------------------------
    def learn(self) -> Optional[float]:
        """Sample a mini-batch and perform one gradient step.

        Returns:
            The scalar loss value, or None if the buffer doesn't yet
            contain enough transitions for a full batch.
        """
        if len(self.replay_buffer) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(
            self.batch_size, device=self.device
        )

        if self.obs_normalizer is not None:
            mean = torch.as_tensor(self.obs_normalizer.mean, dtype=torch.float32, device=self.device)
            std = torch.as_tensor(
                np.sqrt(np.maximum(self.obs_normalizer.var, 1e-8)), dtype=torch.float32, device=self.device
            )
            clip = self.obs_normalizer.clip
            states = torch.clamp((states - mean) / std, -clip, clip)
            next_states = torch.clamp((next_states - mean) / std, -clip, clip)

        # Q(s, a) for the actions that were actually taken
        q_values = self.q_network(states).gather(1, actions)  # (batch, 1)

        # Bellman target: y = r + (1 - d) * gamma * Q_target(s', argmax_a' Q_online(s', a'))
        #
        # This is Double DQN (Van Hasselt et al., 2016), not vanilla DQN. Vanilla DQN's
        # `target_network(next_states).max(...)` uses the SAME network to both pick the
        # best next action and estimate its value -- any noisy overestimate for some
        # action gets both selected AND used as the value, and this bias compounds over
        # thousands of bootstrapped updates, especially with large-magnitude shaped
        # rewards (this is exactly what produced the ~1500-3000 Q-values seen via
        # inspect_policy.py, well beyond the ~max_reward/(1-gamma) sanity ceiling).
        # Double DQN decouples the two: the ONLINE network picks which next action
        # looks best, the TARGET network estimates that specific action's value --
        # this removes the self-reinforcing max-of-noisy-estimates bias.
        with torch.no_grad():
            next_actions = self.q_network(next_states).argmax(dim=1, keepdim=True)
            next_q_values = self.target_network(next_states).gather(1, next_actions)
            targets = rewards + (1.0 - dones) * self.gamma * next_q_values

        loss = self.loss_fn(q_values, targets)

        self.optimizer.zero_grad()
        loss.backward()
        # Clip gradients: without this, an occasional large TD-error (e.g.
        # right after a scenario with an unusually high/low reward) can
        # produce a huge weight update that pushes ReLU biases so negative
        # the hidden layers "die" for the normal-scale real observation
        # distribution -- exactly the collapse diagnosed via
        # inspect_dead_relu.py (network output becomes constant regardless
        # of input). Clipping the gradient norm keeps every update bounded.
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), self.grad_clip_norm)
        self.optimizer.step()

        self._learn_step_counter += 1
        if self._learn_step_counter % self.target_update_freq == 0:
            self.update_target_network()

        return loss.item()

    def update_target_network(self) -> None:
        """Hard-copy online network weights into the target network."""
        self.target_network.load_state_dict(self.q_network.state_dict())

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save(self, path: str) -> None:
        checkpoint = {"q_network": self.q_network.state_dict()}
        if self.obs_normalizer is not None:
            checkpoint["obs_normalizer"] = self.obs_normalizer.state_dict()
        torch.save(checkpoint, path)

    def load(self, path: str) -> None:
        # weights_only=False: our checkpoint stores the obs_normalizer's
        # numpy arrays alongside the network weights. PyTorch >= 2.6
        # defaults torch.load to weights_only=True, which refuses to
        # unpickle anything but raw tensors (as a security precaution for
        # untrusted files). This is our own checkpoint, so it's safe.
        loaded = torch.load(path, map_location=self.device, weights_only=False)
        if isinstance(loaded, dict) and "q_network" in loaded:
            # new-format checkpoint (network weights + normalizer stats)
            state_dict = loaded["q_network"]
            if self.obs_normalizer is not None and "obs_normalizer" in loaded:
                self.obs_normalizer.load_state_dict(loaded["obs_normalizer"])
        else:
            # old-format checkpoint: a bare network state_dict (pre-normalizer)
            state_dict = loaded

        # Auto-detect hidden_size from the checkpoint's first layer weight
        # shape (net.0.weight: [hidden_size, state_size]) and rebuild the
        # networks if it doesn't match this agent's current hidden_size --
        # avoids a confusing shape-mismatch error if a checkpoint was
        # trained with e.g. --hidden-size 512 but this agent was
        # constructed with the default 256 (as evaluate.py / the
        # inspect_*.py scripts do).
        first_linear_key = next(k for k in state_dict if k.endswith("net.0.weight"))
        checkpoint_hidden_size = state_dict[first_linear_key].shape[0]
        current_hidden_size = self.q_network.net[0].out_features
        if checkpoint_hidden_size != current_hidden_size:
            print(f"[DQNAgent.load] checkpoint hidden_size={checkpoint_hidden_size} != "
                  f"current hidden_size={current_hidden_size}; rebuilding networks to match.")
            self.q_network = DQNNetwork(self.state_size, self.action_size, checkpoint_hidden_size).to(self.device)
            self.target_network = DQNNetwork(self.state_size, self.action_size, checkpoint_hidden_size).to(self.device)

        self.q_network.load_state_dict(state_dict)
        self.target_network.load_state_dict(state_dict)


if __name__ == "__main__":
    # Quick manual test (no MetaDrive needed): `python -m dqn_agent`
    state_size, action_size = 20, 6
    agent = DQNAgent(state_size, action_size, buffer_capacity=1000, batch_size=8, seed=0)

    # Fill the buffer with fake transitions
    for _ in range(100):
        s = np.random.randn(state_size).astype(np.float32)
        a = agent.select_action(s)
        r = np.random.randn()
        s2 = np.random.randn(state_size).astype(np.float32)
        d = random.random() < 0.05
        agent.store_transition(s, a, r, s2, d)

    losses = [agent.learn() for _ in range(20)]
    losses = [l for l in losses if l is not None]
    print(f"Ran {len(losses)} learn() steps, last loss = {losses[-1]:.4f}")
    print(f"epsilon before decay: {agent.epsilon:.3f}")
    agent.decay_epsilon()
    print(f"epsilon after decay:  {agent.epsilon:.3f}")

    agent.save("dqn_test_checkpoint.pt")
    agent.load("dqn_test_checkpoint.pt")
    print("save/load OK")