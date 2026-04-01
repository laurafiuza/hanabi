"""
REINFORCE with baseline training loop for Hanabi bot.
Phase 1: Behavioral cloning from heuristic bot (warm start).
Phase 2: REINFORCE fine-tuning via self-play.
"""
import sys
import os
import time
import json
import numpy as np

from engine import create_game, apply_action, get_valid_actions, get_score
from featurize import observe, action_to_index, index_to_action, get_action_mask, OBS_SIZE, ACTION_SIZE
from network import init_weights, forward, sample_action, compute_grad, AdamOptimizer
from heuristic_bot import choose_heuristic_action

# Training hyperparameters
BATCH_SIZE = 64
BC_EPOCHS = 300       # Behavioral cloning epochs
RL_EPOCHS = 1200      # REINFORCE epochs
LR = 3e-4
ENTROPY_COEFF = 0.01
BASELINE_DECAY = 0.99
CHECKPOINT_EVERY = 200
LOG_EVERY = 50
def play_one_game(weights):
    """
    Play a full game with all players using the policy network.
    Returns (trajectory, reward, score).
    trajectory = list of (obs, action_idx, mask, cache) per step.
    Reward = per-step shaped: +2 for successful play, -1 for failed play, 0 otherwise.
    Plus final score bonus.
    """
    state = create_game()
    trajectory = []
    step_rewards = []

    while state['status'] == 'playing':
        player_idx = state['current_player']
        obs = observe(state, player_idx)
        mask = get_action_mask(state, player_idx)

        if not mask.any():
            break

        probs, cache = forward(obs, weights, mask)
        action_idx = sample_action(probs)
        trajectory.append((obs, action_idx, mask, cache))

        old_score = get_score(state)
        action = index_to_action(action_idx, player_idx)
        apply_action(state, action)
        new_score = get_score(state)

        # Per-step reward shaping
        if new_score > old_score:
            step_rewards.append(2.0)  # successful play
        elif action['type'] == 'play' and new_score == old_score:
            step_rewards.append(-1.0)  # failed play (wrong card)
        elif action['type'] == 'discard':
            # Reward safe discards (card already surpassed by play area)
            card = state['discard_pile'][-1] if state['discard_pile'] else None
            if card and state['play_area'][card['suit']] >= card['rank']:
                step_rewards.append(0.5)  # good discard — card was useless
            else:
                step_rewards.append(0.0)
        else:
            step_rewards.append(0.0)

    score = get_score(state)
    # Total reward = sum of step rewards + final score bonus
    total_reward = sum(step_rewards) + score

    return trajectory, total_reward, score


def collect_bc_data(num_games):
    """Play games with heuristic bot, collect (observation, action_index, mask) pairs."""
    data = []
    for _ in range(num_games):
        state = create_game()
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
    """Supervised cross-entropy gradient: maximize log(pi(target|obs))."""
    probs, cache = forward(obs, weights, mask)
    # Gradient = -(one_hot - probs), same as REINFORCE with advantage=1
    return compute_grad(weights, cache, target_idx, advantage=1.0, entropy_coeff=0.0)


def behavioral_cloning(weights, optimizer, bc_data, num_epochs, batch_size=256):
    """Pre-train weights to mimic heuristic bot via supervised learning."""
    print(f"\n=== Phase 1: Behavioral Cloning ({num_epochs} epochs, {len(bc_data)} samples) ===")
    indices = np.arange(len(bc_data))

    for epoch in range(num_epochs):
        np.random.shuffle(indices)
        total_loss = 0.0
        num_batches = 0

        for start in range(0, len(bc_data), batch_size):
            batch_idx = indices[start:start + batch_size]
            total_grad = {k: np.zeros_like(v) for k, v in weights.items()}

            for i in batch_idx:
                obs, target_idx, mask = bc_data[i]
                grad = bc_grad(weights, obs, target_idx, mask)
                for k in total_grad:
                    total_grad[k] += grad[k]

                # Track accuracy
                probs, _ = forward(obs, weights, mask)
                total_loss -= np.log(probs[target_idx] + 1e-10)

            for k in total_grad:
                total_grad[k] /= len(batch_idx)

            # Gradient clipping
            grad_norm = np.sqrt(sum(np.sum(g ** 2) for g in total_grad.values()))
            if grad_norm > 1.0:
                for k in total_grad:
                    total_grad[k] *= 1.0 / grad_norm

            weights = optimizer.step(weights, total_grad)
            num_batches += 1

        if (epoch + 1) % 50 == 0:
            avg_loss = total_loss / len(bc_data)
            # Evaluate
            scores = []
            for _ in range(100):
                traj, reward, score = play_one_game(weights)
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
    print(f"LR={LR}, entropy_coeff={ENTROPY_COEFF}")

    # Phase 1: Behavioral cloning
    print("Collecting heuristic bot demonstrations...")
    bc_data = collect_bc_data(2000)
    print(f"Collected {len(bc_data)} state-action pairs from 2000 games")
    weights = behavioral_cloning(weights, optimizer, bc_data, BC_EPOCHS)
    save_weights(weights, 'checkpoints/weights_after_bc.json')

    # Phase 2: REINFORCE fine-tuning
    print(f"\n=== Phase 2: REINFORCE Fine-tuning ({RL_EPOCHS} epochs x {BATCH_SIZE} games) ===")
    baseline = 0.0
    best_avg = -float('inf')

    for epoch in range(RL_EPOCHS):
        batch_trajectories = []
        batch_rewards = []
        batch_scores = []

        for _ in range(BATCH_SIZE):
            traj, reward, score = play_one_game(weights)
            batch_trajectories.append((traj, reward))
            batch_rewards.append(reward)
            batch_scores.append(score)

        avg_reward = np.mean(batch_rewards)
        avg_score = np.mean(batch_scores)
        baseline = BASELINE_DECAY * baseline + (1 - BASELINE_DECAY) * avg_reward

        total_grad = {k: np.zeros_like(v) for k, v in weights.items()}
        total_steps = 0

        for traj, reward in batch_trajectories:
            advantage = reward - baseline
            for obs, action_idx, mask, cache in traj:
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

        if (epoch + 1) % LOG_EVERY == 0:
            elapsed = time.time() - start_time
            print(f"  RL epoch {epoch+1:5d} | avg_score={avg_score:.2f} | "
                  f"baseline={baseline:.2f} | grad_norm={grad_norm:.4f} | "
                  f"time={elapsed:.0f}s")
            if avg_score > best_avg:
                best_avg = avg_score

        if (epoch + 1) % CHECKPOINT_EVERY == 0:
            save_weights(weights, f'checkpoints/weights_rl_{epoch+1}.json')

    save_weights(weights, 'checkpoints/weights_final.json')
    print("-" * 60)
    print(f"Training complete. Best avg score: {best_avg:.2f}")
    print(f"Total time: {time.time() - start_time:.0f}s")

    return weights


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
