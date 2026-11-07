"""
Task 3: Implement the DQN Neural Network
==========================================

DQNNetwork takes a flattened MetaDrive observation (state vector) as input
and outputs one Q-value per discrete action:

    state_size -> hidden_size -> hidden_size -> action_size

Two hidden fully-connected layers with ReLU activations, exactly as
required by the assignment. Output is raw (no activation) since Q-values
are unbounded real numbers.
"""

import torch
import torch.nn as nn


class DQNNetwork(nn.Module):
    """Simple feed-forward Q-network.

    Args:
        state_size: dimensionality of the flattened observation.
        action_size: number of discrete actions (6 for this assignment).
        hidden_size: width of both hidden layers (default 256).
    """

    def __init__(self, state_size: int, action_size: int, hidden_size: int = 256):
        super().__init__()
        self.state_size = state_size
        self.action_size = action_size

        self.net = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_size),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Compute Q-values for every action given a batch of states.

        Args:
            state: tensor of shape (batch_size, state_size) or (state_size,).

        Returns:
            tensor of shape (batch_size, action_size) or (action_size,)
            containing Q(s, a0), ..., Q(s, a5).
        """
        if state.dim() == 1:
            state = state.unsqueeze(0)
            return self.net(state).squeeze(0)
        return self.net(state)


if __name__ == "__main__":
    # Quick manual shape check: `python -m models.dqn_network`
    dummy_state_size = 259  # example MetaDrive default state-obs size
    net = DQNNetwork(state_size=dummy_state_size, action_size=6)

    single = torch.randn(dummy_state_size)
    batch = torch.randn(32, dummy_state_size)

    print("single ->", net(single).shape)  # expected: torch.Size([6])
    print("batch  ->", net(batch).shape)   # expected: torch.Size([32, 6])
