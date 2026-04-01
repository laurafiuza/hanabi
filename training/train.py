"""
REINFORCE with baseline training loop for Hanabi bot.
Phase 1: Behavioral cloning from heuristic bot (warm start).
Phase 2: REINFORCE fine-tuning, fuse=1 only, terminal reward.

v3: Terminal reward only (no per-step shaping), 10K epochs, batch=128.
The agent must learn conservative play: only play when certain, hint often.
"""
import os
import time
import json
import numpy as np

from engine import create_game, apply_action, get_score
from featurize import observe, action_to_index, index_to_action, get_action_mask, OBS_SIZE, ACTION_SIZE
from network import init_weights, forward, sample_action, compute_grad, AdamOptimizer
from heuristic_bot import choose_heuristic_action

# Training hyperparameters
BATCH_SIZE = 128
BC_EPOCHS = 300
RL_EPOCHS = 10000
LR = 3e-4
ENTROPY_COEFF = 0.001
BASELINE_DECAY = 0.99
CHECKPOINT_EVERY = 1000
LOG_EVERY = 100
GAMMA = 0.99
DEATH_PENALTY = -10  # Harsh penalty for fuse loss / unwinnable


def play_one_game(weights, greedy_teammates=True):
    """
    Play a full game at fuse=1. Only one player samples stochastically per step;
    rest act greedily. Terminal reward only: score if survived, DEATH_PENALTY if died.
    """
    state = create_game(fuse_tokens=1)
    trajectory = []

    while state['status'] == 'playing':
        player_idx = state['current_player']
        obs = observe(state, player_idx)
        mask = get_action_mask(state, player_idx)

        if not mask.any():
            break

        probs, cache = forward(obs, weights, mask)

        if greedy_teammates:
            is_training_player = len(trajectory) % 5 == player_idx
            if is_training_player:
                action_idx = sample_action(probs)
            else:
                action_idx = int(np.argmax(probs))
        else:
            action_idx = sample_action(probs)

        action = index_to_action(action_idx, player_idx)
        trajectory.append((obs, action_idx, mask, cache))
        apply_action(state, action)

    score = get_score(state)

    # Terminal reward only
    if state['status'] == 'lost':
        reward = DEATH_PENALTY
    else:
        reward = score

    return trajectory, reward, score


def compute_discounted_returns(trajectory_len, terminal_reward, gamma=GAMMA):
    """All steps get the discounted terminal reward. R_t = gamma^(T-t) * terminal_reward."""
    returns = np.zeros(trajectory_len, dtype=np.float32)
    for t in range(trajectory_len):
        returns[t] = terminal_reward * (gamma ** (trajectory_len - 1 - t))
    return returns


def collect_bc_data(num_games):
    """Play games with heuristic bot, collect (observation, action_index, mask) pairs."""
    data = []
    for _ in range(num_games):
        state = create_game(fuse_tokens=1)
        while state['status'] == 'playing':
            player_idx = state['current_player']
            obs = observe(state, player_idx)
            mask = get_action_mask(state, player_idx)
            action = choose_heuristic_action(state, player_idx)
            action_idx = action_to_index(action, player_idx)
            data.append((obs, action_idx, mask))
            apply_action(state, action)
    return data


def bc_grad(weights, obs, target_idx, mask):
    """Supervised cross-entropy gradient."""
    probs, cache = forward(obs, weights, mask)
    return compute_grad(weights, cache, target_idx, advantage=1.0, entropy_coeff=0.0)


def behavioral_cloning(weights, optimizer, bc_data, num_epochs, batch_size=256):
    """Pre-train weights to mimic heuristic bot."""
    print(f"\n=== Phase 1: Behavioral Cloning ({num_epochs} epochs, {len(bc_data)} samples) ===")
    indices = np.arange(len(bc_data))

    for epoch in range(num_epochs):
        np.random.shuffle(indices)
        total_loss = 0.0

        for start in range(0, len(bc_data), batch_size):
            batch_idx = indices[start:start + batch_size]
            total_grad = {k: np.zeros_like(v) for k, v in weights.items()}

            for i in batch_idx:
                obs, target_idx, mask = bc_data[i]
                grad = bc_grad(weights, obs, target_idx, mask)
                for k in total_grad:
                    total_grad[k] += grad[k]
                probs, _ = forward(obs, weights, mask)
                total_loss -= np.log(probs[target_idx] + 1e-10)

            for k in total_grad:
                total_grad[k] /= len(batch_idx)

            grad_norm = np.sqrt(sum(np.sum(g ** 2) for g in total_grad.values()))
            if grad_norm > 1.0:
                for k in total_grad:
                    total_grad[k] *= 1.0 / grad_norm

            weights = optimizer.step(weights, total_grad)

        if (epoch + 1) % 50 == 0:
            avg_loss = total_loss / len(bc_data)
            scores = []
            for _ in range(100):
                _, _, score = play_one_game(weights, greedy_teammates=False)
                scores.append(score)
            print(f"  BC epoch {epoch+1:4d} | loss={avg_loss:.3f} | eval_score={np.mean(scores):.2f}")

    return weights


def train():
    np.random.seed(42)
    weights = init_weights()
    optimizer = AdamOptimizer(weights, lr=LR)

    os.makedirs('checkpoints', exist_ok=True)
    start_time = time.time()

    print(f"Network: {OBS_SIZE} -> 256 -> 128 -> {ACTION_SIZE}")
    print(f"LR={LR}, entropy={ENTROPY_COEFF}, gamma={GAMMA}, batch={BATCH_SIZE}")
    print(f"RL epochs: {RL_EPOCHS}, death_penalty={DEATH_PENALTY}")

    # Phase 1: Behavioral cloning
    print("Collecting heuristic bot demonstrations...")
    bc_data = collect_bc_data(2000)
    print(f"Collected {len(bc_data)} state-action pairs from 2000 games")
    weights = behavioral_cloning(weights, optimizer, bc_data, BC_EPOCHS)
    save_weights(weights, 'checkpoints/weights_after_bc.json')

    # Phase 2: REINFORCE (fuse=1 only, terminal reward)
    print(f"\n=== Phase 2: REINFORCE (fuse=1, {RL_EPOCHS} epochs x {BATCH_SIZE} games) ===")
    baseline = 0.0
    best_avg = -float('inf')
    best_weights = None

    for epoch in range(RL_EPOCHS):
        batch_trajectories = []
        batch_scores = []

        for _ in range(BATCH_SIZE):
            traj, reward, score = play_one_game(weights, greedy_teammates=True)
            returns = compute_discounted_returns(len(traj), reward)
            batch_trajectories.append((traj, returns))
            batch_scores.append(score)

        avg_score = np.mean(batch_scores)
        baseline = BASELINE_DECAY * baseline + (1 - BASELINE_DECAY) * avg_score

        total_grad = {k: np.zeros_like(v) for k, v in weights.items()}
        total_steps = 0

        for traj, returns in batch_trajectories:
            for t, (obs, action_idx, mask, cache) in enumerate(traj):
                advantage = returns[t] - baseline
                grad = compute_grad(weights, cache, action_idx, advantage, ENTROPY_COEFF)
                for k in total_grad:
                    total_grad[k] += grad[k]
                total_steps += 1

        if total_steps > 0:
            for k in total_grad:
                total_grad[k] /= total_steps

        grad_norm = np.sqrt(sum(np.sum(g ** 2) for g in total_grad.values()))
        if grad_norm > 1.0:
            for k in total_grad:
                total_grad[k] *= 1.0 / grad_norm

        weights = optimizer.step(weights, total_grad)

        if avg_score > best_avg:
            best_avg = avg_score
            best_weights = {k: v.copy() for k, v in weights.items()}

        if (epoch + 1) % LOG_EVERY == 0:
            elapsed = time.time() - start_time
            print(f"  epoch {epoch+1:6d} | avg_score={avg_score:.2f} | "
                  f"best={best_avg:.2f} | baseline={baseline:.2f} | "
                  f"grad_norm={grad_norm:.4f} | time={elapsed:.0f}s")

        if (epoch + 1) % CHECKPOINT_EVERY == 0:
            save_weights(weights, f'checkpoints/weights_epoch_{epoch+1}.json')

    # Save best weights (not just latest — in case of late-stage regression)
    if best_weights:
        save_weights(best_weights, 'checkpoints/weights_best.json')
    save_weights(weights, 'checkpoints/weights_final.json')

    # Final eval
    print("\n=== Final Evaluation (fuse=1, greedy, 500 games) ===")
    for label, w in [("latest", weights), ("best", best_weights or weights)]:
        scores = []
        for _ in range(500):
            state = create_game(fuse_tokens=1)
            while state['status'] == 'playing':
                player_idx = state['current_player']
                obs = observe(state, player_idx)
                mask = get_action_mask(state, player_idx)
                if not mask.any():
                    break
                probs, _ = forward(obs, w, mask)
                action_idx = int(np.argmax(probs))
                action = index_to_action(action_idx, player_idx)
                apply_action(state, action)
            scores.append(get_score(state))
        print(f"  {label:8s}: avg={np.mean(scores):.2f}, median={np.median(scores):.0f}, max={max(scores)}")

    print("-" * 60)
    print(f"Training complete. Best avg score: {best_avg:.2f}")
    print(f"Total time: {time.time() - start_time:.0f}s")

    return best_weights or weights


def save_weights(weights, path):
    data = {k: v.tolist() for k, v in weights.items()}
    with open(path, 'w') as f:
        json.dump(data, f)


def load_weights(path):
    with open(path) as f:
        data = json.load(f)
    return {k: np.array(v, dtype=np.float32) for k, v in data.items()}


if __name__ == '__main__':
    train()
