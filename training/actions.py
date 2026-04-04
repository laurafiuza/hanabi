"""
Fixed 68-dim action encoding/decoding for Hanabi.

Layout:
  [0..3]   PLAY_CARD cardIndex 0-3
  [4..7]   DISCARD cardIndex 0-3
  [8..67]  GIVE_HINT: 4 relative targets x (5 suits + 5 ranks)
           target t, suit s: 8 + t*10 + s
           target t, rank r: 8 + t*10 + 5 + (r-1)

Target indices are relative: 0 = next player in turn order, 1 = two ahead, etc.
"""
from engine import NUM_PLAYERS

NUM_ACTIONS = 68


def encode_action(action, player_idx):
    """Convert a game action dict to a fixed action index [0, 67]."""
    atype = action['type']
    if atype == 'play':
        return action['card_idx']
    elif atype == 'discard':
        return 4 + action['card_idx']
    elif atype == 'hint':
        # Convert absolute target to relative offset
        abs_target = action['target']
        rel_target = (abs_target - player_idx - 1) % NUM_PLAYERS
        hint = action['hint']
        base = 8 + rel_target * 10
        if hint['kind'] == 'suit':
            return base + hint['value']  # suit is already 0-4
        else:
            return base + 5 + (hint['value'] - 1)  # rank 1-5 -> 0-4
    raise ValueError(f"Unknown action type: {atype}")


def decode_action(action_idx, player_idx):
    """Convert a fixed action index back to a game action dict."""
    if action_idx < 4:
        return {'type': 'play', 'player': player_idx, 'card_idx': action_idx}
    elif action_idx < 8:
        return {'type': 'discard', 'player': player_idx, 'card_idx': action_idx - 4}
    else:
        hint_idx = action_idx - 8
        rel_target = hint_idx // 10
        within = hint_idx % 10
        abs_target = (player_idx + 1 + rel_target) % NUM_PLAYERS
        if within < 5:
            return {'type': 'hint', 'player': player_idx, 'target': abs_target,
                    'hint': {'kind': 'suit', 'value': within}}
        else:
            return {'type': 'hint', 'player': player_idx, 'target': abs_target,
                    'hint': {'kind': 'rank', 'value': within - 5 + 1}}
    raise ValueError(f"Invalid action index: {action_idx}")


def get_valid_action_mask(state, player_idx):
    """Return a list of 68 booleans indicating valid actions."""
    from engine import get_valid_actions
    mask = [False] * NUM_ACTIONS
    for action in get_valid_actions(state, player_idx):
        idx = encode_action(action, player_idx)
        mask[idx] = True
    return mask
