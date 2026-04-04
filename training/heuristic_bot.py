"""
Python port of the heuristic bot from src/engine/ai.ts.
Used as baseline and for ExIt training data generation.
"""
from engine import (
    NUM_PLAYERS, SUITS, RANK_COPIES, MAX_INFO_TOKENS, get_valid_actions
)

SUIT_COUNT = len(SUITS)


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


def best_hint_by_info_count(candidates, player_idx):
    if not candidates:
        return None
    candidates.sort(key=lambda c: c['count'], reverse=True)
    best = candidates[0]
    return {'type': 'hint', 'player': player_idx, 'target': best['target'],
            'hint': best['hint']}


# ---------------------------------------------------------------------------
# Hint finders
# ---------------------------------------------------------------------------

def find_critical_hint(state, player_idx):
    """Two-tier critical hint:
    Tier 1: complete partial info on a critical card (one hint away from full knowledge)
    Tier 2: hint about a critical card with no info yet (prefer suit for non-5s, rank for 5s)
    """
    completion = []
    new_info = []

    for offset in range(1, NUM_PLAYERS):
        target_idx = (player_idx + offset) % NUM_PLAYERS
        target = state['players'][target_idx]

        for i, card in enumerate(target['hand']):
            if not is_critical(card, state):
                continue
            knows_suit = target['hint_info']['known_suits'][i] == card['suit']
            knows_rank = target['hint_info']['known_ranks'][i] == card['rank']
            if knows_suit and knows_rank:
                continue

            has_partial = knows_suit or knows_rank

            if has_partial:
                # Tier 1: give the missing dimension
                if not knows_rank:
                    hint = {'kind': 'rank', 'value': card['rank']}
                    completion.append({'target': target_idx, 'hint': hint,
                                       'count': hint_new_info_count(target, hint)})
                if not knows_suit:
                    hint = {'kind': 'suit', 'value': card['suit']}
                    completion.append({'target': target_idx, 'hint': hint,
                                       'count': hint_new_info_count(target, hint)})
            else:
                # Tier 2: no info yet
                if card['rank'] == 5:
                    hint = {'kind': 'rank', 'value': card['rank']}
                    new_info.append({'target': target_idx, 'hint': hint,
                                     'count': hint_new_info_count(target, hint)})
                else:
                    # Prefer suit for non-5 critical cards (more disambiguating)
                    suit_hint = {'kind': 'suit', 'value': card['suit']}
                    new_info.append({'target': target_idx, 'hint': suit_hint,
                                     'count': hint_new_info_count(target, suit_hint) + 10})
                    rank_hint = {'kind': 'rank', 'value': card['rank']}
                    new_info.append({'target': target_idx, 'hint': rank_hint,
                                     'count': hint_new_info_count(target, rank_hint)})

    result = best_hint_by_info_count(completion, player_idx)
    if result:
        return result
    return best_hint_by_info_count(new_info, player_idx)


def find_playable_hint(state, player_idx):
    candidates = []
    for offset in range(1, NUM_PLAYERS):
        target_idx = (player_idx + offset) % NUM_PLAYERS
        target = state['players'][target_idx]
        for i, card in enumerate(target['hand']):
            if state['play_area'][card['suit']] != card['rank'] - 1:
                continue
            knows_suit = target['hint_info']['known_suits'][i] == card['suit']
            knows_rank = target['hint_info']['known_ranks'][i] == card['rank']
            if knows_suit and knows_rank:
                continue
            if not knows_rank:
                hint = {'kind': 'rank', 'value': card['rank']}
                candidates.append({'target': target_idx, 'hint': hint,
                                   'count': hint_new_info_count(target, hint)})
            if not knows_suit:
                hint = {'kind': 'suit', 'value': card['suit']}
                candidates.append({'target': target_idx, 'hint': hint,
                                   'count': hint_new_info_count(target, hint)})
    return best_hint_by_info_count(candidates, player_idx)


def find_any_useful_hint(state, player_idx):
    candidates = []
    for offset in range(1, NUM_PLAYERS):
        target_idx = (player_idx + offset) % NUM_PLAYERS
        target = state['players'][target_idx]
        for i, card in enumerate(target['hand']):
            needed = state['play_area'][card['suit']] + 1
            if card['rank'] >= needed and card['rank'] <= needed + 1:
                if target['hint_info']['known_ranks'][i] != card['rank']:
                    hint = {'kind': 'rank', 'value': card['rank']}
                    candidates.append({'target': target_idx, 'hint': hint,
                                       'count': hint_new_info_count(target, hint)})
                if target['hint_info']['known_suits'][i] != card['suit']:
                    hint = {'kind': 'suit', 'value': card['suit']}
                    candidates.append({'target': target_idx, 'hint': hint,
                                       'count': hint_new_info_count(target, hint)})
    return best_hint_by_info_count(candidates, player_idx)


# ---------------------------------------------------------------------------
# Discard helpers
# ---------------------------------------------------------------------------

def find_useless_discard(state, player_idx):
    me = state['players'][player_idx]
    # Full knowledge: suit+rank and already played
    for i in range(len(me['hand'])):
        ks = me['hint_info']['known_suits'][i]
        kr = me['hint_info']['known_ranks'][i]
        if ks is not None and kr is not None and is_card_useless(ks, kr, state):
            return i
    # Rank-only: all suits have played past this rank
    for i in range(len(me['hand'])):
        kr = me['hint_info']['known_ranks'][i]
        if kr is not None and is_rank_useless(kr, state):
            return i
    # Suit-only: suit is complete
    for i in range(len(me['hand'])):
        ks = me['hint_info']['known_suits'][i]
        kr = me['hint_info']['known_ranks'][i]
        if ks is not None and kr is None and is_suit_complete(ks, state):
            return i
    return -1


# ---------------------------------------------------------------------------
# Main heuristic
# ---------------------------------------------------------------------------

def choose_heuristic_action(state, player_idx):
    actions = get_valid_actions(state, player_idx)
    me = state['players'][player_idx]

    # 1-2. Play known-safe, prefer lowest rank
    best_idx = -1
    best_rank = 999
    for i in range(len(me['hand'])):
        ks = me['hint_info']['known_suits'][i]
        kr = me['hint_info']['known_ranks'][i]
        if ks is not None and kr is not None and is_card_playable(ks, kr, state):
            if kr < best_rank:
                best_rank = kr
                best_idx = i
        if kr is not None and ks is None:
            all_ready = all(state['play_area'][s] >= kr or state['play_area'][s] == kr - 1
                           for s in range(SUIT_COUNT))
            some_need = any(state['play_area'][s] == kr - 1 for s in range(SUIT_COUNT))
            if all_ready and some_need and kr < best_rank:
                best_rank = kr
                best_idx = i
    if best_idx >= 0:
        return {'type': 'play', 'player': player_idx, 'card_idx': best_idx}

    # At max info tokens, prefer discarding useless before hinting
    if state['info_tokens'] >= MAX_INFO_TOKENS:
        useless = find_useless_discard(state, player_idx)
        if useless >= 0:
            return {'type': 'discard', 'player': player_idx, 'card_idx': useless}

    # 3. Critical hint (two-tier)
    if state['info_tokens'] > 0:
        hint = find_critical_hint(state, player_idx)
        if hint:
            return hint

    # 4. Playable hint
    if state['info_tokens'] > 0:
        hint = find_playable_hint(state, player_idx)
        if hint:
            return hint

    # 5. Discard known-useless
    useless = find_useless_discard(state, player_idx)
    if useless >= 0:
        return {'type': 'discard', 'player': player_idx, 'card_idx': useless}

    # 6. Any useful hint before discarding
    if state['info_tokens'] > 0:
        hint = find_any_useful_hint(state, player_idx)
        if hint:
            return hint

    # 7. Discard oldest unhinted card (highest age, no hint info), never discard known 5
    best_idx = -1
    best_age = -1
    for i in range(len(me['hand'])):
        ks = me['hint_info']['known_suits'][i]
        kr = me['hint_info']['known_ranks'][i]
        if ks is None and kr is None and me['hint_info']['card_ages'][i] > best_age:
            best_age = me['hint_info']['card_ages'][i]
            best_idx = i
    if best_idx >= 0:
        return {'type': 'discard', 'player': player_idx, 'card_idx': best_idx}

    # 8. Discard least-known card, never discard known 5, prefer older
    best_idx = -1
    best_score = float('-inf')
    for i in range(len(me['hand'])):
        kr = me['hint_info']['known_ranks'][i]
        if kr == 5:
            continue
        ks = me['hint_info']['known_suits'][i]
        score = 0
        if kr is None:
            score += 3
        if ks is None:
            score += 1
        score += me['hint_info']['card_ages'][i] * 0.1
        if score > best_score:
            best_score = score
            best_idx = i
    if best_idx >= 0:
        return {'type': 'discard', 'player': player_idx, 'card_idx': best_idx}

    # 9. Any valid hint
    if state['info_tokens'] > 0:
        hint_actions = [a for a in actions if a['type'] == 'hint']
        if hint_actions:
            return hint_actions[0]

    # 10. Last resort
    return {'type': 'discard', 'player': player_idx, 'card_idx': 0}


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
