"""
Minimal numpy feedforward network for REINFORCE policy gradient.
Input(414) -> Dense(256, ReLU) -> Dense(128, ReLU) -> Dense(48, masked softmax)
"""
import numpy as np
from featurize import OBS_SIZE, ACTION_SIZE


def init_weights():
    """Xavier initialization for all layers."""
    def xavier(fan_in, fan_out):
        scale = np.sqrt(2.0 / (fan_in + fan_out))
        return np.random.randn(fan_in, fan_out).astype(np.float32) * scale

    return {
        'W1': xavier(OBS_SIZE, 256),
        'b1': np.zeros(256, dtype=np.float32),
        'W2': xavier(256, 128),
        'b2': np.zeros(128, dtype=np.float32),
        'W3': xavier(128, ACTION_SIZE),
        'b3': np.zeros(ACTION_SIZE, dtype=np.float32),
    }


def forward(obs, weights, mask):
    """
    Forward pass. Returns (action_probs, cache).
    obs: (OBS_SIZE,)
    mask: (ACTION_SIZE,) boolean
    """
    # Layer 1
    z1 = obs @ weights['W1'] + weights['b1']
    h1 = np.maximum(z1, 0)  # ReLU

    # Layer 2
    z2 = h1 @ weights['W2'] + weights['b2']
    h2 = np.maximum(z2, 0)  # ReLU

    # Layer 3 (logits)
    logits = h2 @ weights['W3'] + weights['b3']

    # Masked softmax
    logits_masked = np.where(mask, logits, -1e9)
    logits_shifted = logits_masked - logits_masked.max()
    exp_logits = np.exp(logits_shifted)
    probs = exp_logits / exp_logits.sum()

    cache = {'obs': obs, 'z1': z1, 'h1': h1, 'z2': z2, 'h2': h2,
             'logits': logits, 'probs': probs, 'mask': mask}
    return probs, cache


def sample_action(probs):
    """Sample an action index from the probability distribution."""
    return np.random.choice(len(probs), p=probs)


def compute_grad(weights, cache, action_idx, advantage, entropy_coeff=0.01):
    """
    Compute gradient of the REINFORCE loss:
      loss = -advantage * log(pi(a|s)) - entropy_coeff * H(pi)
    Returns a dict of gradients matching weights keys.
    """
    probs = cache['probs']
    mask = cache['mask']

    # d_loss/d_logits for policy gradient: -advantage * (one_hot - probs) for masked
    one_hot = np.zeros(ACTION_SIZE, dtype=np.float32)
    one_hot[action_idx] = 1.0

    # Policy gradient component
    d_logits = -advantage * (one_hot - probs)

    # Entropy bonus gradient: encourages exploration
    # H = -sum(p * log(p)), d_H/d_logits = -sum(dp/dlogits * (log(p) + 1))
    # For softmax: dp_i/dlogits_j = p_i(delta_ij - p_j)
    # d_H/d_logits_j = sum_i p_i * (delta_ij - p_j) * (log(p_i) + 1)
    # Simplification: d_H/d_logits = p * (log(p) + 1) - p * sum(p * (log(p) + 1))
    # = p * ((log(p) + 1) - sum(p * (log(p) + 1)))
    safe_log = np.log(probs + 1e-10)
    entropy_per_logit = probs * (safe_log + 1)
    d_entropy = entropy_per_logit - probs * entropy_per_logit.sum()
    d_logits += entropy_coeff * d_entropy

    # Zero out invalid actions
    d_logits = np.where(mask, d_logits, 0.0)

    # Backprop through layer 3
    dW3 = np.outer(cache['h2'], d_logits)
    db3 = d_logits
    dh2 = d_logits @ weights['W3'].T

    # Backprop through ReLU + layer 2
    dz2 = dh2 * (cache['z2'] > 0).astype(np.float32)
    dW2 = np.outer(cache['h1'], dz2)
    db2 = dz2
    dh1 = dz2 @ weights['W2'].T

    # Backprop through ReLU + layer 1
    dz1 = dh1 * (cache['z1'] > 0).astype(np.float32)
    dW1 = np.outer(cache['obs'], dz1)
    db1 = dz1

    return {'W1': dW1, 'b1': db1, 'W2': dW2, 'b2': db2, 'W3': dW3, 'b3': db3}


class AdamOptimizer:
    def __init__(self, weights, lr=3e-4, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 0
        self.m = {k: np.zeros_like(v) for k, v in weights.items()}
        self.v = {k: np.zeros_like(v) for k, v in weights.items()}

    def step(self, weights, grads):
        self.t += 1
        for k in weights:
            self.m[k] = self.beta1 * self.m[k] + (1 - self.beta1) * grads[k]
            self.v[k] = self.beta2 * self.v[k] + (1 - self.beta2) * grads[k] ** 2
            m_hat = self.m[k] / (1 - self.beta1 ** self.t)
            v_hat = self.v[k] / (1 - self.beta2 ** self.t)
            weights[k] -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
        return weights


if __name__ == '__main__':
    weights = init_weights()
    obs = np.random.randn(OBS_SIZE).astype(np.float32)
    mask = np.ones(ACTION_SIZE, dtype=bool)
    mask[10:] = False  # only first 10 actions valid

    probs, cache = forward(obs, weights, mask)
    print(f"Probs shape: {probs.shape}, sum: {probs.sum():.6f}")
    print(f"Valid probs sum: {probs[:10].sum():.6f}")

    action = sample_action(probs)
    grad = compute_grad(weights, cache, action, advantage=1.0)
    print(f"Grad keys: {list(grad.keys())}")
    print(f"W1 grad norm: {np.linalg.norm(grad['W1']):.4f}")
    print("Network test passed!")
