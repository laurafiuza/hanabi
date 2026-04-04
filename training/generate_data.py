"""
Generate training data by playing games with the heuristic bot.
Uses multiprocessing for speed.
"""
import os
import sys
import time
import pickle
import multiprocessing as mp
import numpy as np

# Ensure training/ is on the path
sys.path.insert(0, os.path.dirname(__file__))

from engine import create_game, apply_action
from heuristic_bot import choose_heuristic_action
from features import featurize, NUM_FEATURES
from actions import encode_action, NUM_ACTIONS


def play_one_game(_seed_ignored):
    """Play one game, return list of (features, action_idx) tuples."""
    samples = []
    state = create_game()
    while state['status'] == 'playing':
        player_idx = state['current_player']
        feat = featurize(state, player_idx)
        action = choose_heuristic_action(state, player_idx)
        action_idx = encode_action(action, player_idx)
        samples.append((feat, action_idx))
        apply_action(state, action)
    return samples


def generate_dataset(num_games=100_000, num_workers=None):
    """Generate training data from heuristic games.

    Returns (features, labels) as numpy arrays.
    """
    if num_workers is None:
        num_workers = min(mp.cpu_count(), 8)

    print(f"Generating {num_games:,} games using {num_workers} workers...")
    t0 = time.time()

    with mp.Pool(num_workers) as pool:
        all_games = pool.map(play_one_game, range(num_games), chunksize=500)

    # Flatten
    all_samples = []
    for game_samples in all_games:
        all_samples.extend(game_samples)

    elapsed = time.time() - t0
    print(f"Generated {len(all_samples):,} samples from {num_games:,} games in {elapsed:.1f}s")

    # Convert to numpy
    features = np.zeros((len(all_samples), NUM_FEATURES), dtype=np.float32)
    labels = np.zeros(len(all_samples), dtype=np.int64)
    for i, (feat, act) in enumerate(all_samples):
        features[i] = feat
        labels[i] = act

    return features, labels


def main():
    num_games = int(sys.argv[1]) if len(sys.argv) > 1 else 100_000
    features, labels = generate_dataset(num_games)

    out_path = os.path.join(os.path.dirname(__file__), 'data.pkl')
    with open(out_path, 'wb') as f:
        pickle.dump({'features': features, 'labels': labels}, f)

    print(f"Saved to {out_path}")
    print(f"  Features shape: {features.shape}")
    print(f"  Labels shape:   {labels.shape}")
    print(f"  Action distribution: {np.bincount(labels, minlength=NUM_ACTIONS)[:8]}... (first 8)")


if __name__ == '__main__':
    main()
