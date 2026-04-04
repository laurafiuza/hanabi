"""
Small MLP policy network for Hanabi.
Input: 124 features -> Hidden: 2x128 ReLU -> Output: 68 action logits
"""
import torch
import torch.nn as nn
from features import NUM_FEATURES
from actions import NUM_ACTIONS


class HanabiNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(NUM_FEATURES, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, NUM_ACTIONS),
        )

    def forward(self, x, mask=None):
        """Forward pass. Returns logits (pre-softmax).

        Args:
            x: (batch, NUM_FEATURES) float tensor
            mask: (batch, NUM_ACTIONS) bool tensor — True for valid actions
        Returns:
            logits: (batch, NUM_ACTIONS) with -inf for invalid actions if mask provided
        """
        logits = self.net(x)
        if mask is not None:
            logits = logits.masked_fill(~mask, float('-inf'))
        return logits

    def predict_action(self, features, valid_mask):
        """Single-sample greedy prediction for inference.

        Args:
            features: np.array of shape (NUM_FEATURES,)
            valid_mask: list of bool, length NUM_ACTIONS
        Returns:
            action_idx: int
        """
        self.eval()
        with torch.no_grad():
            x = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
            m = torch.tensor(valid_mask, dtype=torch.bool).unsqueeze(0)
            logits = self.forward(x, m)
            return logits.argmax(dim=1).item()
