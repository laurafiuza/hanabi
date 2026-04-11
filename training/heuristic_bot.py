"""
Python port of the heuristic bot from src/engine/ai.ts.
Uses unified action scoring (tier + score) for decision making.
Used as baseline and for ExIt training data generation.
"""
from engine import (
    NUM_PLAYERS, SUITS, RANK_COPIES, MAX_INFO_TOKENS, get_valid_actions
)

SUIT_COUNT = len(SUITS)

TIER_ORDER = {'must': 3, 'strong': 2, 'neutral': 1, 'bad': 0}


# ---------------------------------------------------------------------------
# Convention helpers
# ---------------------------------------------------------------------------

def get_chop_index(player):
    """Return index of the oldest (leftmost) unclued card, or -1 if fully clued."""
    for i in range(len(player['hand'])):
        if (player['hint_info']['known_suits'][i] is None
                and player['hint_info']['known_ranks'][i] is None):
            return i
    return -1


def is_card_trash(card, state):
    """A card is trash if already played or unreachable (gap below in suit)."""
    suit, rank = card['suit'], card['rank']
    if state['play_area'][suit] >= rank:
        return True
    next_needed = state['play_area'][suit] + 1
    for r in range(next_needed, rank):
        total = RANK_COPIES[r]
        discarded = sum(1 for c in state['discard_pile']
                        if c['suit'] == suit and c['rank'] == r)
        if discarded >= total:
            return True
    return False


def hint_touches_trash(target, hint, state):
    """Good Touch Principle: does this hint touch any trash card?"""
    for i, card in enumerate(target['hand']):
        if hint['kind'] == 'suit':
            touches = card['suit'] == hint['value']
        else:
            touches = card['rank'] == hint['value']
        if touches and is_card_trash(card, state):
            return True
    return False


def is_critical(card, state):
    suit = card['suit']
    rank = card['rank']
    if state['play_area'][suit] >= rank:
        return False
    total = RANK_COPIES[rank]
    discarded = sum(1 for c in state['discard_pile']
                    if c['suit'] == suit and c['rank'] == rank)
    return discarded >= total - 1


def is_card_playable(suit, rank, state):
    return state['play_area'][suit] == rank - 1


def is_card_useless(suit, rank, state):
    return state['play_area'][suit] >= rank


def is_rank_useless(rank, state):
    return all(state['play_area'][s] >= rank for s in range(SUIT_COUNT))


def is_suit_complete(suit, state):
    return state['play_area'][suit] >= 5


def is_known_safe_play(ks, kr, state):
    if ks is not None and kr is not None:
        return is_card_playable(ks, kr, state)
    if kr is not None and ks is None:
        all_ready = all(state['play_area'][s] >= kr or state['play_area'][s] == kr - 1
                        for s in range(SUIT_COUNT))
        some_need = any(state['play_area'][s] == kr - 1 for s in range(SUIT_COUNT))
        return all_ready and some_need
    return False


def hint_new_info_count(target, hint):
    count = 0
    for i, card in enumerate(target['hand']):
        if hint['kind'] == 'suit':
            if card['suit'] == hint['value'] and target['hint_info']['known_suits'][i] != hint['value']:
                count += 1
        else:
            if card['rank'] == hint['value'] and target['hint_info']['known_ranks'][i] != hint['value']:
                count += 1
    return count


# ---------------------------------------------------------------------------
# Hint classification helpers
# ---------------------------------------------------------------------------

def is_hint_a_save(state, target_idx, hint):
    target = state['players'][target_idx]
    chop_idx = get_chop_index(target)
    if chop_idx < 0:
        return False
    chop_card = target['hand'][chop_idx]
    if not is_critical(chop_card, state):
        return False
    if hint['kind'] == 'suit':
        return chop_card['suit'] == hint['value']
    return chop_card['rank'] == hint['value']


def is_hint_a_2_save(state, target_idx, hint):
    target = state['players'][target_idx]
    chop_idx = get_chop_index(target)
    if chop_idx < 0:
        return False
    chop_card = target['hand'][chop_idx]
    if chop_card['rank'] != 2 or state['play_area'][chop_card['suit']] >= 2:
        return False
    if is_critical(chop_card, state):
        return False
    if (target['hint_info']['known_suits'][chop_idx] is not None
            or target['hint_info']['known_ranks'][chop_idx] is not None):
        return False
    if hint['kind'] == 'suit':
        return chop_card['suit'] == hint['value']
    return chop_card['rank'] == hint['value']


def is_hint_a_play_clue(state, target_idx, hint):
    target = state['players'][target_idx]
    for i, card in enumerate(target['hand']):
        if hint['kind'] == 'suit':
            touches = card['suit'] == hint['value']
        else:
            touches = card['rank'] == hint['value']
        if not touches:
            continue
        if state['play_area'][card['suit']] == card['rank'] - 1:
            ks = target['hint_info']['known_suits'][i] == card['suit']
            kr = target['hint_info']['known_ranks'][i] == card['rank']
            if not (ks and kr):
                return True
    return False


def is_hint_useful(state, target_idx, hint):
    target = state['players'][target_idx]
    for i, card in enumerate(target['hand']):
        if hint['kind'] == 'suit':
            touches = card['suit'] == hint['value']
        else:
            touches = card['rank'] == hint['value']
        if not touches:
            continue
        needed = state['play_area'][card['suit']] + 1
        if card['rank'] >= needed and card['rank'] <= needed + 1:
            return True
    return False


# ---------------------------------------------------------------------------
# Core scoring function
# ---------------------------------------------------------------------------

def score_action(state, player_idx, action):
    """Returns (tier, score) for an action."""
    me = state['players'][player_idx]

    if action['type'] == 'play':
        idx = action['card_idx']
        ks = me['hint_info']['known_suits'][idx]
        kr = me['hint_info']['known_ranks'][idx]

        if is_known_safe_play(ks, kr, state):
            # Plays are must-tier but score below critical saves (which start at 80+).
            return ('must', 60 - kr)

        if state['turns_remaining'] is not None and (ks is not None or kr is not None):
            return ('neutral', 10)

        if ks is not None and kr is not None:
            return ('bad', -100)

        # Partially known (suit or rank) — risky but not blind
        if ks is not None or kr is not None:
            return ('bad', -30)

        # Truly unknown — no info at all
        return ('bad', -200)

    if action['type'] == 'discard':
        idx = action['card_idx']
        ks = me['hint_info']['known_suits'][idx]
        kr = me['hint_info']['known_ranks'][idx]

        if kr == 5:
            return ('bad', -50)

        if is_known_safe_play(ks, kr, state):
            return ('bad', -100)

        if ks is not None and kr is not None and is_critical(me['hand'][idx], state):
            return ('bad', -80)

        if ks is not None and kr is not None and is_card_useless(ks, kr, state):
            return ('strong', 50)
        if kr is not None and is_rank_useless(kr, state):
            return ('strong', 48)
        if ks is not None and kr is None and is_suit_complete(ks, state):
            return ('strong', 46)

        chop_idx = get_chop_index(me)
        if chop_idx >= 0 and idx == chop_idx:
            token_bonus = 40 if state['info_tokens'] >= MAX_INFO_TOKENS else (10 if state['info_tokens'] >= 4 else 0)
            return ('neutral', 35 + token_bonus)

        if ks is None and kr is None:
            return ('bad', -10)

        return ('bad', -20)

    if action['type'] == 'hint':
        target_idx = action['target']
        target = state['players'][target_idx]
        hint = action['hint']
        info_count = hint_new_info_count(target, hint)
        touches_trash = hint_touches_trash(target, hint, state)

        if info_count == 0:
            return ('bad', 0)

        if touches_trash:
            if is_hint_a_save(state, target_idx, hint):
                return ('strong', 40 + info_count)
            return ('bad', 0)

        if is_hint_a_save(state, target_idx, hint):
            return ('must', 80 + info_count)

        if is_hint_a_2_save(state, target_idx, hint):
            return ('strong', 60 + info_count)

        if is_hint_a_play_clue(state, target_idx, hint):
            return ('strong', 55 + info_count)

        if is_hint_useful(state, target_idx, hint):
            return ('neutral', 30 + info_count)

        return ('neutral', 10 + info_count)

    return ('neutral', 0)


# ---------------------------------------------------------------------------
# Main heuristic (uses score_action)
# ---------------------------------------------------------------------------

def choose_heuristic_action(state, player_idx):
    actions = get_valid_actions(state, player_idx)
    scored = []
    for a in actions:
        tier, sc = score_action(state, player_idx, a)
        scored.append((a, tier, sc))

    # Sort by tier desc, then score desc
    scored.sort(key=lambda x: (TIER_ORDER[x[1]], x[2]), reverse=True)

    # Pick highest non-bad, or least-bad
    for a, tier, sc in scored:
        if tier != 'bad':
            return a
    return scored[0][0]


if __name__ == '__main__':
    from engine import create_game, apply_action, get_score
    scores = []
    for _ in range(1000):
        state = create_game()
        while state['status'] == 'playing':
            action = choose_heuristic_action(state, state['current_player'])
            apply_action(state, action)
        scores.append(get_score(state))
    avg = sum(scores) / len(scores)
    print(f"Heuristic bot: 1000 games, avg score = {avg:.2f}, "
          f"min = {min(scores)}, max = {max(scores)}")
