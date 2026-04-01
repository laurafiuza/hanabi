"""
Evaluate RL bot vs heuristic bot.
"""
import sys
import numpy as np
from engine import create_game, apply_action, get_valid_actions, get_score
from featurize import observe, get_action_mask, index_to_action
from network import forward, sample_action
from train import load_weights
from heuristic_bot import choose_heuristic_action

NUM_GAMES = 1000


def play_game_with_policy(weights, deterministic=False):
    """All players use RL policy."""
    state = create_game()
    while state['status'] == 'playing':
        player_idx = state['current_player']
        obs = observe(state, player_idx)
        mask = get_action_mask(state, player_idx)
        if not mask.any():
            break
        probs, _ = forward(obs, weights, mask)
        if deterministic:
            action_idx = int(np.argmax(probs))
        else:
            action_idx = sample_action(probs)
        action = index_to_action(action_idx, player_idx)
        apply_action(state, action)
    return get_score(state)


def play_game_heuristic():
    """All players use heuristic bot."""
    state = create_game()
    while state['status'] == 'playing':
        action = choose_heuristic_action(state, state['current_player'])
        apply_action(state, action)
    return get_score(state)


def evaluate(weights_path):
    weights = load_weights(weights_path)

    print(f"Evaluating {NUM_GAMES} games each...")
    print("-" * 50)

    # Heuristic bot
    heuristic_scores = [play_game_heuristic() for _ in range(NUM_GAMES)]
    print(f"Heuristic:       avg={np.mean(heuristic_scores):.2f}, "
          f"median={np.median(heuristic_scores):.0f}, "
          f"std={np.std(heuristic_scores):.2f}, "
          f"max={max(heuristic_scores)}")

    # RL bot (stochastic)
    rl_scores = [play_game_with_policy(weights, deterministic=False) for _ in range(NUM_GAMES)]
    print(f"RL (stochastic): avg={np.mean(rl_scores):.2f}, "
          f"median={np.median(rl_scores):.0f}, "
          f"std={np.std(rl_scores):.2f}, "
          f"max={max(rl_scores)}")

    # RL bot (deterministic / greedy)
    rl_det_scores = [play_game_with_policy(weights, deterministic=True) for _ in range(NUM_GAMES)]
    print(f"RL (greedy):     avg={np.mean(rl_det_scores):.2f}, "
          f"median={np.median(rl_det_scores):.0f}, "
          f"std={np.std(rl_det_scores):.2f}, "
          f"max={max(rl_det_scores)}")


if __name__ == '__main__':
    path = sys.argv[1] if len(sys.argv) > 1 else 'checkpoints/weights_final.json'
    evaluate(path)
