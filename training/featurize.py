"""
State featurization for RL training.
Converts game state into a fixed-size observation vector and handles action indexing.
"""
import numpy as np
from engine import NUM_PLAYERS, HAND_SIZE, MAX_INFO_TOKENS, INITIAL_FUSE_TOKENS, RANK_COPIES

# Observation vector layout:
# play_area:       25  (5 suits x 5 ranks, one-hot per suit progress)
# info_tokens:      1  (normalized 0-1)
# fuse_tokens:      1  (normalized 0-1)
# deck_fraction:    1  (normalized 0-1)
# my_hand_hints:   40  (4 cards x (5 suit + 5 rank) one-hot)
# others_cards:   160  (4 players x 4 cards x (5 suit + 5 rank))
# others_hints:   160  (4 players x 4 cards x (5 suit + 5 rank))
# discard_summary: 25  (5 suits x 5 ranks, normalized count)
# turns_remaining:  1
# TOTAL:          414

OBS_SIZE = 414

# Action space: 48 fixed indices
# [0..3]   PLAY_CARD(i)
# [4..7]   DISCARD(i)
# [8..47]  GIVE_HINT: 4 target offsets x (5 suits + 5 ranks)
#          index = 8 + (target_offset - 1) * 10 + hint_type
#          hint_type: 0-4 = suit index, 5-9 = rank index (rank-1)
ACTION_SIZE = 48


def observe(state, player_idx):
    """Build observation vector for the given player. Returns ndarray of shape (OBS_SIZE,)."""
    obs = np.zeros(OBS_SIZE, dtype=np.float32)
    offset = 0

    # Play area: 25 features (5 suits x 5 ranks, cumulative one-hot)
    for suit_idx in range(5):
        level = state['play_area'][suit_idx]
        for r in range(level):
            obs[offset + suit_idx * 5 + r] = 1.0
    offset += 25

    # Info tokens (normalized)
    obs[offset] = state['info_tokens'] / MAX_INFO_TOKENS
    offset += 1

    # Fuse tokens (normalized)
    obs[offset] = state['fuse_tokens'] / max(INITIAL_FUSE_TOKENS, 1)
    offset += 1

    # Deck fraction
    obs[offset] = len(state['deck']) / 50.0
    offset += 1

    # My hand hints: 4 cards x (5 suit bits + 5 rank bits) = 40
    me = state['players'][player_idx]
    for i in range(HAND_SIZE):
        card_offset = offset + i * 10
        if i < len(me['hand']):
            ks = me['hint_info']['known_suits'][i]
            kr = me['hint_info']['known_ranks'][i]
            if ks is not None:
                obs[card_offset + ks] = 1.0
            if kr is not None:
                obs[card_offset + 5 + (kr - 1)] = 1.0
    offset += 40

    # Other players' actual cards: 4 players x 4 cards x 10 = 160
    other_indices = [(player_idx + d) % NUM_PLAYERS for d in range(1, NUM_PLAYERS)]
    for pi, oidx in enumerate(other_indices):
        other = state['players'][oidx]
        for i in range(HAND_SIZE):
            card_offset = offset + pi * HAND_SIZE * 10 + i * 10
            if i < len(other['hand']):
                card = other['hand'][i]
                obs[card_offset + card['suit']] = 1.0
                obs[card_offset + 5 + (card['rank'] - 1)] = 1.0
    offset += 160

    # Other players' hint info: 4 players x 4 cards x 10 = 160
    for pi, oidx in enumerate(other_indices):
        other = state['players'][oidx]
        for i in range(HAND_SIZE):
            card_offset = offset + pi * HAND_SIZE * 10 + i * 10
            if i < len(other['hand']):
                ks = other['hint_info']['known_suits'][i]
                kr = other['hint_info']['known_ranks'][i]
                if ks is not None:
                    obs[card_offset + ks] = 1.0
                if kr is not None:
                    obs[card_offset + 5 + (kr - 1)] = 1.0
    offset += 160

    # Discard pile summary: 5 suits x 5 ranks = 25
    for card in state['discard_pile']:
        idx = card['suit'] * 5 + (card['rank'] - 1)
        obs[offset + idx] += 1.0 / RANK_COPIES[card['rank']]
    offset += 25

    # Turns remaining
    if state['turns_remaining'] is not None:
        obs[offset] = state['turns_remaining'] / NUM_PLAYERS
    offset += 1

    assert offset == OBS_SIZE
    return obs


def action_to_index(action, player_idx):
    """Convert an action dict to a fixed action index [0..47]."""
    atype = action['type']
    if atype == 'play':
        return action['card_idx']
    elif atype == 'discard':
        return 4 + action['card_idx']
    elif atype == 'hint':
        # Compute target offset (1..4) relative to player_idx
        target_offset = (action['target'] - player_idx) % NUM_PLAYERS
        hint = action['hint']
        if hint['kind'] == 'suit':
            hint_type = hint['value']  # 0..4
        else:
            hint_type = 5 + (hint['value'] - 1)  # 5..9
        return 8 + (target_offset - 1) * 10 + hint_type
    raise ValueError(f"Unknown action type: {atype}")


def index_to_action(index, player_idx):
    """Convert a fixed action index back to an action dict."""
    if index < 4:
        return {'type': 'play', 'player': player_idx, 'card_idx': index}
    elif index < 8:
        return {'type': 'discard', 'player': player_idx, 'card_idx': index - 4}
    else:
        hint_idx = index - 8
        target_offset = hint_idx // 10 + 1  # 1..4
        hint_type = hint_idx % 10
        target = (player_idx + target_offset) % NUM_PLAYERS
        if hint_type < 5:
            hint = {'kind': 'suit', 'value': hint_type}
        else:
            hint = {'kind': 'rank', 'value': hint_type - 5 + 1}
        return {'type': 'hint', 'player': player_idx, 'target': target, 'hint': hint}


def get_action_mask(state, player_idx):
    """Returns a boolean mask of shape (ACTION_SIZE,) — True for valid actions."""
    mask = np.zeros(ACTION_SIZE, dtype=bool)
    valid_actions = state['_valid_actions'] if '_valid_actions' in state else None

    # Cache valid actions if not already cached
    if valid_actions is None:
        from engine import get_valid_actions
        valid_actions = get_valid_actions(state, player_idx)

    for action in valid_actions:
        idx = action_to_index(action, player_idx)
        mask[idx] = True

    return mask


if __name__ == '__main__':
    from engine import create_game, get_valid_actions
    state = create_game()
    obs = observe(state, 0)
    print(f"Observation shape: {obs.shape}, non-zero: {np.count_nonzero(obs)}")

    actions = get_valid_actions(state, 0)
    mask = get_action_mask(state, 0)
    print(f"Valid actions: {len(actions)}, mask sum: {mask.sum()}")

    # Round-trip test
    for a in actions:
        idx = action_to_index(a, 0)
        a2 = index_to_action(idx, 0)
        assert a['type'] == a2['type'], f"Mismatch: {a} -> {idx} -> {a2}"
    print("Action round-trip test passed!")
