"""
State featurization for the neural network.

Produces a ~124-dim feature vector from the perspective of a given player.
All features are normalized to roughly [0, 1] range.
"""
import numpy as np
from engine import SUITS, RANK_COPIES, NUM_PLAYERS

SUIT_COUNT = len(SUITS)
NUM_FEATURES = 124


def featurize(state, player_idx):
    """Convert game state to a feature vector from player_idx's perspective.

    Returns np.array of shape (NUM_FEATURES,), dtype float32.
    """
    features = np.zeros(NUM_FEATURES, dtype=np.float32)
    idx = 0

    # --- Play area: 5 suits x current rank, normalized /5 ---
    for s in range(SUIT_COUNT):
        features[idx] = state['play_area'][s] / 5.0
        idx += 1

    # --- Info tokens (normalized /8) and fuse tokens ---
    features[idx] = state['info_tokens'] / 8.0
    idx += 1
    features[idx] = float(state['fuse_tokens'])
    idx += 1

    # --- Own hand hint info: 4 cards x 13 features ---
    me = state['players'][player_idx]
    for i in range(4):  # HAND_SIZE = 4
        if i < len(me['hand']):
            ks = me['hint_info']['known_suits'][i]
            kr = me['hint_info']['known_ranks'][i]
            # Suit one-hot (5)
            if ks is not None:
                features[idx + ks] = 1.0
            idx += 5
            # Rank one-hot (5)
            if kr is not None:
                features[idx + (kr - 1)] = 1.0
            idx += 5
            # has_suit, has_rank flags
            features[idx] = 1.0 if ks is not None else 0.0
            idx += 1
            features[idx] = 1.0 if kr is not None else 0.0
            idx += 1
            # Card age normalized
            features[idx] = me['hint_info']['card_ages'][i] / 20.0
            idx += 1
        else:
            idx += 13  # skip empty slot

    # --- Other players (compressed): 4 players x 7 features ---
    for offset in range(1, NUM_PLAYERS):
        other_idx = (player_idx + offset) % NUM_PLAYERS
        other = state['players'][other_idx]
        # Rank counts (5): how many of each rank in hand
        rank_counts = [0] * 5
        playable_count = 0
        critical_count = 0
        for card in other['hand']:
            rank_counts[card['rank'] - 1] += 1
            if state['play_area'][card['suit']] == card['rank'] - 1:
                playable_count += 1
            if _is_critical_fast(card, state):
                critical_count += 1
        for r in range(5):
            features[idx] = rank_counts[r] / 4.0
            idx += 1
        features[idx] = playable_count / 4.0
        idx += 1
        features[idx] = critical_count / 4.0
        idx += 1

    # --- Danger signals ---
    # Remaining copies: 25 dims (5 suits x 5 ranks)
    for s in range(SUIT_COUNT):
        for r in range(1, 6):
            total = RANK_COPIES[r]
            played = 1 if state['play_area'][s] >= r else 0
            discarded = sum(1 for c in state['discard_pile']
                           if c['suit'] == s and c['rank'] == r)
            remaining = total - played - discarded
            features[idx] = max(0, remaining) / 3.0
            idx += 1

    # Dead suit flags: 5 dims
    for s in range(SUIT_COUNT):
        dead = False
        needed = state['play_area'][s] + 1
        if needed <= 5:
            for r in range(needed, 6):
                total = RANK_COPIES[r]
                discarded = sum(1 for c in state['discard_pile']
                               if c['suit'] == s and c['rank'] == r)
                if discarded >= total:
                    dead = True
                    break
        features[idx] = 1.0 if dead else 0.0
        idx += 1

    # Max achievable per suit: 5 dims
    for s in range(SUIT_COUNT):
        max_rank = 5
        for r in range(state['play_area'][s] + 1, 6):
            total = RANK_COPIES[r]
            discarded = sum(1 for c in state['discard_pile']
                           if c['suit'] == s and c['rank'] == r)
            if discarded >= total:
                max_rank = r - 1
                break
        features[idx] = max_rank / 5.0
        idx += 1

    # --- Deck size and turns remaining ---
    features[idx] = len(state['deck']) / 40.0
    idx += 1
    if state['turns_remaining'] is None:
        features[idx] = -1.0
    else:
        features[idx] = state['turns_remaining'] / 5.0
    idx += 1

    assert idx == NUM_FEATURES, f"Expected {NUM_FEATURES} features, got {idx}"
    return features


def _is_critical_fast(card, state):
    """Quick criticality check (no import needed)."""
    suit = card['suit']
    rank = card['rank']
    if state['play_area'][suit] >= rank:
        return False
    total = RANK_COPIES[rank]
    discarded = sum(1 for c in state['discard_pile']
                    if c['suit'] == suit and c['rank'] == rank)
    return discarded >= total - 1
