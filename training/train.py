"""
Supervised training of the Hanabi policy network.

Phase 1: Train on heuristic data (cross-entropy on hard labels).
Phase 2: Expert Iteration — MCTS generates soft targets (visit distributions),
         network trains on KL divergence to match them.
"""
import os
import sys
import pickle
import time
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import multiprocessing as mp

sys.path.insert(0, os.path.dirname(__file__))

from network import HanabiNet
from features import featurize, NUM_FEATURES
from actions import (
    NUM_ACTIONS, encode_action, decode_action,
    get_valid_action_mask
)
from engine import create_game, apply_action, get_score, clone_state
from heuristic_bot import choose_heuristic_action
from mcts import run_mcts, action_key


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(model, num_games=1000):
    """Evaluate model by playing games greedily. Returns avg score."""
    model.eval()
    scores = []
    for _ in range(num_games):
        state = create_game()
        while state['status'] == 'playing':
            player_idx = state['current_player']
            feat = featurize(state, player_idx)
            mask = get_valid_action_mask(state, player_idx)
            action_idx = model.predict_action(feat, mask)
            action = decode_action(action_idx, player_idx)
            apply_action(state, action)
        scores.append(get_score(state))
    return sum(scores) / len(scores), min(scores), max(scores)


def evaluate_heuristic(num_games=1000):
    """Baseline: play games with heuristic bot."""
    scores = []
    for _ in range(num_games):
        state = create_game()
        while state['status'] == 'playing':
            action = choose_heuristic_action(state, state['current_player'])
            apply_action(state, action)
        scores.append(get_score(state))
    return sum(scores) / len(scores)


def evaluate_mcts_with_model(model, num_games=200, mcts_budget_ms=200):
    """Evaluate MCTS using the neural network as rollout policy."""
    model.eval()

    def neural_policy(state, player_idx):
        feat = featurize(state, player_idx)
        mask = get_valid_action_mask(state, player_idx)
        action_idx = model.predict_action(feat, mask)
        return decode_action(action_idx, player_idx)

    scores = []
    for _ in range(num_games):
        state = create_game()
        while state['status'] == 'playing':
            player_idx = state['current_player']
            visits = run_mcts(state, player_idx, neural_policy, time_budget_ms=mcts_budget_ms)
            if visits:
                best_action = max(visits, key=lambda x: x[1])[0]
                apply_action(state, best_action)
            else:
                action = choose_heuristic_action(state, player_idx)
                apply_action(state, action)
        scores.append(get_score(state))
    return sum(scores) / len(scores), min(scores), max(scores)


# ---------------------------------------------------------------------------
# Phase 1: Supervised training on heuristic data
# ---------------------------------------------------------------------------

def train_supervised(data_path, epochs=30, batch_size=512, lr=1e-3, model=None):
    """Train network on heuristic data (hard labels)."""
    print("Loading data...")
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    features, labels = data['features'], data['labels']
    print(f"  {len(features):,} samples")

    # Split 95/5
    n = len(features)
    perm = np.random.permutation(n)
    val_size = n // 20
    val_idx, train_idx = perm[:val_size], perm[val_size:]

    train_ds = TensorDataset(torch.tensor(features[train_idx]), torch.tensor(labels[train_idx]))
    val_ds = TensorDataset(torch.tensor(features[val_idx]), torch.tensor(labels[val_idx]))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    if model is None:
        model = HanabiNet()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    print(f"Training for {epochs} epochs...")
    best_val_loss = float('inf')
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(xb)
            train_correct += (logits.argmax(1) == yb).sum().item()
            train_total += len(xb)
        scheduler.step()

        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                logits = model(xb)
                loss = criterion(logits, yb)
                val_loss += loss.item() * len(xb)
                val_correct += (logits.argmax(1) == yb).sum().item()
                val_total += len(xb)

        tl = train_loss / train_total
        ta = train_correct / train_total * 100
        vl = val_loss / val_total
        va = val_correct / val_total * 100

        if vl < best_val_loss:
            best_val_loss = vl
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}  train_loss={tl:.4f} acc={ta:.1f}%  "
                  f"val_loss={vl:.4f} acc={va:.1f}%")

    model.load_state_dict(best_state)
    return model


# ---------------------------------------------------------------------------
# Phase 2: Expert Iteration — generate MCTS data, train on soft targets
# ---------------------------------------------------------------------------

def _generate_one_game_mcts(args):
    """Worker function: play one game with MCTS, collect visit distributions.

    Uses the heuristic as the rollout policy for stable MCTS quality.
    The neural net is not used during data generation to avoid the
    degradation spiral where a weakening net produces worse MCTS data.
    """
    game_idx, mcts_budget_ms = args

    samples = []
    state = create_game()

    while state['status'] == 'playing':
        player_idx = state['current_player']
        feat = featurize(state, player_idx)

        # Run MCTS with heuristic rollout (stable quality)
        visits = run_mcts(state, player_idx, choose_heuristic_action, time_budget_ms=mcts_budget_ms)

        if visits:
            # Build soft target from visit distribution
            target = np.zeros(NUM_ACTIONS, dtype=np.float32)
            total_visits = sum(v for _, v in visits)
            for action, v in visits:
                idx = encode_action(action, player_idx)
                target[idx] = v / total_visits

            # Also build validity mask
            mask = np.array(get_valid_action_mask(state, player_idx), dtype=np.float32)

            samples.append((feat, target, mask))

            # Play the best action
            best_action = max(visits, key=lambda x: x[1])[0]
            apply_action(state, best_action)
        else:
            # Fallback to heuristic
            action = choose_heuristic_action(state, player_idx)
            apply_action(state, action)

    return samples


def generate_mcts_data(num_games=5000, mcts_budget_ms=200, num_workers=None):
    """Generate training data using MCTS with heuristic rollout."""
    if num_workers is None:
        num_workers = min(mp.cpu_count(), 8)

    print(f"  Generating {num_games:,} MCTS games ({mcts_budget_ms}ms/move, {num_workers} workers)...")
    t0 = time.time()

    args = [(i, mcts_budget_ms) for i in range(num_games)]

    with mp.Pool(num_workers) as pool:
        all_games = pool.map(_generate_one_game_mcts, args, chunksize=10)

    # Flatten
    all_samples = []
    for game_samples in all_games:
        all_samples.extend(game_samples)

    elapsed = time.time() - t0
    games_per_sec = num_games / elapsed
    print(f"  {len(all_samples):,} samples from {num_games:,} games "
          f"in {elapsed:.0f}s ({games_per_sec:.1f} games/s)")

    # Convert to numpy
    features = np.zeros((len(all_samples), NUM_FEATURES), dtype=np.float32)
    targets = np.zeros((len(all_samples), NUM_ACTIONS), dtype=np.float32)
    masks = np.zeros((len(all_samples), NUM_ACTIONS), dtype=np.float32)
    for i, (feat, target, mask) in enumerate(all_samples):
        features[i] = feat
        targets[i] = target
        masks[i] = mask

    return features, targets, masks


def train_mixed(heuristic_features, heuristic_labels, mcts_features, mcts_targets, mcts_masks,
                 epochs=30, batch_size=512, lr=1e-3):
    """Train a fresh network on mixed heuristic (hard) + MCTS (soft) targets.

    Strategy: convert heuristic hard labels to one-hot soft targets, then
    train on the combined dataset with the same soft-target loss.
    This prevents catastrophic forgetting by keeping the heuristic knowledge.
    """
    # Convert heuristic hard labels to one-hot soft targets
    n_heuristic = len(heuristic_features)
    h_targets = np.zeros((n_heuristic, NUM_ACTIONS), dtype=np.float32)
    h_targets[np.arange(n_heuristic), heuristic_labels] = 1.0
    # Build validity masks for heuristic data (the chosen action is always valid)
    h_masks = np.zeros((n_heuristic, NUM_ACTIONS), dtype=np.float32)
    h_masks[np.arange(n_heuristic), heuristic_labels] = 1.0
    # Actually we need full masks — but we don't have them for heuristic data.
    # Set all to 1.0 and let the soft target handle it (only the chosen action has weight)
    h_masks[:] = 1.0

    # Combine datasets
    all_features = np.concatenate([heuristic_features, mcts_features])
    all_targets = np.concatenate([h_targets, mcts_targets])
    all_masks = np.concatenate([h_masks, mcts_masks])

    print(f"  Combined dataset: {n_heuristic:,} heuristic + {len(mcts_features):,} MCTS "
          f"= {len(all_features):,} total")

    # Split 95/5
    n = len(all_features)
    perm = np.random.permutation(n)
    val_size = n // 20
    val_idx, train_idx = perm[:val_size], perm[val_size:]

    def make_loader(idx):
        return DataLoader(
            TensorDataset(
                torch.tensor(all_features[idx]),
                torch.tensor(all_targets[idx]),
                torch.tensor(all_masks[idx]),
            ),
            batch_size=batch_size, shuffle=True
        )

    train_loader = make_loader(train_idx)
    val_loader = make_loader(val_idx)

    # Train from scratch
    model = HanabiNet()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_loss = float('inf')
    best_state = {k: v.clone() for k, v in model.state_dict().items()}

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        train_total = 0
        for xb, yb, mb in train_loader:
            optimizer.zero_grad()
            logits = model(xb)
            logits = logits.masked_fill(mb == 0, -1e9)
            log_probs = F.log_softmax(logits, dim=1)
            loss = -(yb * log_probs).sum(dim=1).mean()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(xb)
            train_total += len(xb)
        scheduler.step()

        model.eval()
        val_loss = 0.0
        val_total = 0
        with torch.no_grad():
            for xb, yb, mb in val_loader:
                logits = model(xb)
                logits = logits.masked_fill(mb == 0, -1e9)
                log_probs = F.log_softmax(logits, dim=1)
                loss = -(yb * log_probs).sum(dim=1).mean()
                val_loss += loss.item() * len(xb)
                val_total += len(xb)

        tl = train_loss / train_total
        vl = val_loss / val_total

        if vl < best_val_loss:
            best_val_loss = vl
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if epoch % 5 == 0 or epoch == 1:
            print(f"    Epoch {epoch:3d}  train_loss={tl:.4f}  val_loss={vl:.4f}")

    model.load_state_dict(best_state)
    return model


def run_exit(num_iterations=3, games_per_iter=5000, mcts_budget_ms=200):
    """Expert Iteration loop.

    Strategy:
    1. Generate MCTS data with heuristic rollout (stable, consistent quality)
    2. Combine with original heuristic data to prevent forgetting
    3. Train from scratch on the mixed dataset each iteration
    4. Accumulate MCTS data across iterations
    """
    print("\n" + "=" * 60)
    print("EXPERT ITERATION")
    print("=" * 60)

    ckpt_dir = os.path.join(os.path.dirname(__file__), 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)

    # Load heuristic data (subsample to avoid drowning out MCTS data)
    data_path = os.path.join(os.path.dirname(__file__), 'data.pkl')
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    heuristic_features = data['features']
    heuristic_labels = data['labels']
    # Subsample: keep at most 5x the expected MCTS data size per iteration
    max_heuristic = games_per_iter * 25 * 5  # ~25 moves/game, 5x ratio
    if len(heuristic_features) > max_heuristic:
        idx = np.random.choice(len(heuristic_features), max_heuristic, replace=False)
        heuristic_features = heuristic_features[idx]
        heuristic_labels = heuristic_labels[idx]
    print(f"  Heuristic data: {len(heuristic_features):,} samples (subsampled)")

    # Accumulate MCTS data across iterations
    all_mcts_features = []
    all_mcts_targets = []
    all_mcts_masks = []

    best_model = None
    best_score = 0.0

    for iteration in range(1, num_iterations + 1):
        print(f"\n--- Iteration {iteration}/{num_iterations} ---")

        # Generate MCTS data
        features, targets, masks = generate_mcts_data(
            num_games=games_per_iter, mcts_budget_ms=mcts_budget_ms
        )

        all_mcts_features.append(features)
        all_mcts_targets.append(targets)
        all_mcts_masks.append(masks)

        # Combine all MCTS data so far
        mcts_f = np.concatenate(all_mcts_features)
        mcts_t = np.concatenate(all_mcts_targets)
        mcts_m = np.concatenate(all_mcts_masks)

        # Train from scratch on mixed data
        model = train_mixed(heuristic_features, heuristic_labels,
                           mcts_f, mcts_t, mcts_m, epochs=30)

        # Save checkpoint
        ckpt_path = os.path.join(ckpt_dir, f'policy_net_exit{iteration}.pt')
        torch.save(model.state_dict(), ckpt_path)

        # Evaluate
        print(f"  Evaluating standalone neural net (1000 games)...")
        nn_avg, nn_min, nn_max = evaluate(model, 1000)
        print(f"    Standalone NN: avg={nn_avg:.2f}, min={nn_min}, max={nn_max}")

        if nn_avg > best_score:
            best_score = nn_avg
            best_model = model
            best_path = os.path.join(ckpt_dir, 'policy_net.pt')
            torch.save(model.state_dict(), best_path)
            print(f"    New best! Saved to {best_path}")

    return best_model if best_model is not None else model


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    data_path = os.path.join(os.path.dirname(__file__), 'data.pkl')

    if not os.path.exists(data_path):
        print(f"No data found at {data_path}. Run generate_data.py first.")
        sys.exit(1)

    # Baseline eval
    print("Baseline evaluation...")
    heuristic_avg = evaluate_heuristic(1000)
    print(f"  Heuristic: avg={heuristic_avg:.2f}")

    # Expert Iteration
    num_iters = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    games = int(sys.argv[2]) if len(sys.argv) > 2 else 5000
    budget = int(sys.argv[3]) if len(sys.argv) > 3 else 200

    model = run_exit(num_iterations=num_iters, games_per_iter=games, mcts_budget_ms=budget)

    # Final evaluation
    print("\n" + "=" * 60)
    print("FINAL EVALUATION")
    print("=" * 60)
    heuristic_avg = evaluate_heuristic(1000)
    print(f"  Heuristic: avg={heuristic_avg:.2f}")
    nn_avg, nn_min, nn_max = evaluate(model, 1000)
    print(f"  Neural net (standalone): avg={nn_avg:.2f}, min={nn_min}, max={nn_max}")


if __name__ == '__main__':
    main()
