"""
REINFORCE with baseline training loop for Hanabi bot.
Phase 1: Behavioral cloning from heuristic bot (warm start).
Phase 2: REINFORCE fine-tuning with curriculum on fuse tokens (3→2→1).

Key improvements over v1:
- Curriculum: train with 3 fuse tokens first, tighten to 1
- Greedy teammates: only the current player samples, others act greedily
- Per-step discounted returns instead of flat episode return
- Baseline tracks raw score, not shaped reward
- Lower entropy after BC (0.001)
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
BATCH_SIZE = 64
BC_EPOCHS = 300
LR = 3e-4
ENTROPY_COEFF = 0.001  # Lower after BC — we already have a good init
BASELINE_DECAY = 0.99
CHECKPOINT_EVERY = 200
LOG_EVERY = 50
GAMMA = 0.99  # Per-step discount

# Curriculum: (fuse_tokens, num_rl_epochs)
CURRICULUM = [
    (3, 600),
    (2, 600),
    (1, 600),
]


def play_one_game(weights, fuse_tokens=1, greedy_teammates=True):
    """
    Play a full game. Only one player per step samples stochastically;
    all others act greedily (highest-prob valid action).
    Returns (trajectory, step_rewards, score).
    trajectory entries are only for the stochastically-acting steps.
    """
    state = create_game(fuse_tokens=fuse_tokens)
    trajectory = []
    step_rewards = []

    while state['status'] == 'playing':
        player_idx = state['current_player']
        obs = observe(state, player_idx)
        mask = get_action_mask(state, player_idx)

        if not mask.any():
            break

        probs, cache = forward(obs, weights, mask)

        if greedy_teammates:
            # Only one random player per game step samples; rest act greedily
            # The "training player" rotates: use step count mod to pick
            is_training_player = (len(trajectory) + len(step_rewards)) % 5 == player_idx
            if is_training_player:
                action_idx = sample_action(probs)
            else:
                action_idx = int(np.argmax(probs))
        else:
            action_idx = sample_action(probs)

        # Always record trajectory (we need gradients for all players since
        # they share weights), but mark whether this was a sampled action
        old_score = get_score(state)
        action = index_to_action(action_idx, player_idx)
        apply_action(state, action)
        new_score = get_score(state)

        # Per-step reward
        if new_score > old_score:
            r = 2.0
        elif action['type'] == 'play' and new_score == old_score:
            r = -1.0
        elif action['type'] == 'discard':
            card = state['discard_pile'][-1] if state['discard_pile'] else None
            if card and state['play_area'][card['suit']] >= card['rank']:
                r = 0.5
            else:
                r = 0.0
        else:
            r = 0.0

        trajectory.append((obs, action_idx, mask, cache))
        step_rewards.append(r)

    score = get_score(state)
    return trajectory, step_rewards, score


def compute_discounted_returns(step_rewards, final_score, gamma=GAMMA):
    """Compute per-step discounted returns: R_t = r_t + gamma*r_{t+1} + ... + score_bonus."""
    T = len(step_rewards)
    returns = np.zeros(T, dtype=np.float32)
    # The final "bonus" is the game score, added to last step
    running = float(final_score)
    for t in reversed(range(T)):
        running = step_rewards[t] + gamma * running
        returns[t] = running
    return returns


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
                _, _, score = play_one_game(weights, fuse_tokens=1, greedy_teammates=False)
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
    print(f"LR={LR}, entropy_coeff={ENTROPY_COEFF}, gamma={GAMMA}")
    print(f"Curriculum: {[(f, e) for f, e in CURRICULUM]}")

    # Phase 1: Behavioral cloning
    print("Collecting heuristic bot demonstrations...")
    bc_data = collect_bc_data(2000)
    print(f"Collected {len(bc_data)} state-action pairs from 2000 games")
    weights = behavioral_cloning(weights, optimizer, bc_data, BC_EPOCHS)
    save_weights(weights, 'checkpoints/weights_after_bc.json')

    # Phase 2: REINFORCE with curriculum
    for stage_idx, (fuse_tokens, rl_epochs) in enumerate(CURRICULUM):
        print(f"\n=== Phase 2.{stage_idx+1}: REINFORCE (fuse={fuse_tokens}, {rl_epochs} epochs x {BATCH_SIZE} games) ===")
        baseline = 0.0  # Reset baseline for each curriculum stage
        best_avg = -float('inf')

        for epoch in range(rl_epochs):
            batch_trajectories = []
            batch_scores = []

            for _ in range(BATCH_SIZE):
                traj, step_rews, score = play_one_game(weights, fuse_tokens=fuse_tokens, greedy_teammates=True)
                returns = compute_discounted_returns(step_rews, score)
                batch_trajectories.append((traj, returns))
                batch_scores.append(score)

            avg_score = np.mean(batch_scores)
            # Baseline tracks raw score
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

            if (epoch + 1) % LOG_EVERY == 0:
                elapsed = time.time() - start_time
                print(f"  epoch {epoch+1:5d} | fuse={fuse_tokens} | avg_score={avg_score:.2f} | "
                      f"baseline={baseline:.2f} | grad_norm={grad_norm:.4f} | "
                      f"time={elapsed:.0f}s")
                if avg_score > best_avg:
                    best_avg = avg_score

            if (epoch + 1) % CHECKPOINT_EVERY == 0:
                save_weights(weights, f'checkpoints/weights_fuse{fuse_tokens}_epoch{epoch+1}.json')

        save_weights(weights, f'checkpoints/weights_after_fuse{fuse_tokens}.json')
        print(f"  Stage done. Best avg score (fuse={fuse_tokens}): {best_avg:.2f}")

    # Final eval with fuse=1 (production setting)
    print("\n=== Final Evaluation (fuse=1, greedy, 500 games) ===")
    final_scores = []
    for _ in range(500):
        state = create_game(fuse_tokens=1)
        while state['status'] == 'playing':
            player_idx = state['current_player']
            obs = observe(state, player_idx)
            mask = get_action_mask(state, player_idx)
            if not mask.any():
                break
            probs, _ = forward(obs, weights, mask)
            action_idx = int(np.argmax(probs))  # greedy
            action = index_to_action(action_idx, player_idx)
            apply_action(state, action)
        final_scores.append(get_score(state))
    print(f"  Greedy RL bot (fuse=1): avg={np.mean(final_scores):.2f}, "
          f"median={np.median(final_scores):.0f}, max={max(final_scores)}")

    save_weights(weights, 'checkpoints/weights_final.json')
    print("-" * 60)
    print(f"Training complete. Total time: {time.time() - start_time:.0f}s")

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
