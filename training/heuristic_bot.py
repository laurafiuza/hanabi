"""
Python port of the heuristic bot from src/engine/ai.ts.
Used as baseline for comparison with RL bot.
"""
from engine import (
    NUM_PLAYERS, SUITS, RANK_COPIES, get_valid_actions
)


def is_critical(card, state):
    suit = card['suit']
    rank = card['rank']
    if state['play_area'][suit] >= rank:
        return False
    total = RANK_COPIES[rank]
    discarded = sum(1 for c in state['discard_pile']
                    if c['suit'] == suit and c['rank'] == rank)
    return discarded >= total - 1


def find_critical_hint(state, player_idx):
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
            if not knows_rank:
                return {'type': 'hint', 'player': player_idx, 'target': target_idx,
                        'hint': {'kind': 'rank', 'value': card['rank']}}
            if not knows_suit:
                return {'type': 'hint', 'player': player_idx, 'target': target_idx,
                        'hint': {'kind': 'suit', 'value': card['suit']}}
    return None


def find_playable_hint(state, player_idx):
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
                return {'type': 'hint', 'player': player_idx, 'target': target_idx,
                        'hint': {'kind': 'rank', 'value': card['rank']}}
            if not knows_suit:
                return {'type': 'hint', 'player': player_idx, 'target': target_idx,
                        'hint': {'kind': 'suit', 'value': card['suit']}}
    return None


def find_useful_hint(state, player_idx):
    for offset in range(1, NUM_PLAYERS):
        target_idx = (player_idx + offset) % NUM_PLAYERS
        target = state['players'][target_idx]
        for i, card in enumerate(target['hand']):
            if card['rank'] == 5 and target['hint_info']['known_ranks'][i] != 5:
                return {'type': 'hint', 'player': player_idx, 'target': target_idx,
                        'hint': {'kind': 'rank', 'value': 5}}
    return None


def choose_heuristic_action(state, player_idx):
    actions = get_valid_actions(state, player_idx)
    me = state['players'][player_idx]

    # 1. Play known-safe
    for i in range(len(me['hand'])):
        ks = me['hint_info']['known_suits'][i]
        kr = me['hint_info']['known_ranks'][i]
        if ks is not None and kr is not None:
            if state['play_area'][ks] == kr - 1:
                return {'type': 'play', 'player': player_idx, 'card_idx': i}

    # 2. Play known-rank universally safe
    for i in range(len(me['hand'])):
        kr = me['hint_info']['known_ranks'][i]
        ks = me['hint_info']['known_suits'][i]
        if kr is not None and ks is None:
            all_ready = all(state['play_area'][s] >= kr or state['play_area'][s] == kr - 1
                           for s in range(5))
            some_need = any(state['play_area'][s] == kr - 1 for s in range(5))
            if all_ready and some_need:
                return {'type': 'play', 'player': player_idx, 'card_idx': i}

    # 3. Critical hint
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
    for i in range(len(me['hand'])):
        ks = me['hint_info']['known_suits'][i]
        kr = me['hint_info']['known_ranks'][i]
        if ks is not None and kr is not None:
            if state['play_area'][ks] >= kr:
                return {'type': 'discard', 'player': player_idx, 'card_idx': i}

    # 6. Useful hint when tokens low
    if state['info_tokens'] > 0 and state['info_tokens'] <= 3:
        hint = find_useful_hint(state, player_idx)
        if hint:
            return hint

    # 7. Discard unhinted
    for i in range(len(me['hand'])):
        ks = me['hint_info']['known_suits'][i]
        kr = me['hint_info']['known_ranks'][i]
        if ks is None and kr is None:
            return {'type': 'discard', 'player': player_idx, 'card_idx': i}

    # 8. Any hint
    if state['info_tokens'] > 0:
        hint_actions = [a for a in actions if a['type'] == 'hint']
        if hint_actions:
            return hint_actions[0]

    # 9. Discard oldest
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
