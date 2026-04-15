"""
Python port of the Hanabi game engine (training only).
Mirrors src/engine/{types,deck,hints,validation,game}.ts
Uses mutable state for speed.
"""
import random
from copy import deepcopy

# Constants
SUITS = ['white', 'yellow', 'green', 'blue', 'red']
SUIT_IDX = {s: i for i, s in enumerate(SUITS)}
RANKS = [1, 2, 3, 4, 5]
RANK_COPIES = {1: 3, 2: 2, 3: 2, 4: 2, 5: 1}
MAX_INFO_TOKENS = 8
INITIAL_FUSE_TOKENS = 3
HAND_SIZE = 4
NUM_PLAYERS = 5


def create_deck():
    cards = []
    for suit_idx, suit in enumerate(SUITS):
        for rank in RANKS:
            for copy in range(RANK_COPIES[rank]):
                cards.append({'suit': suit_idx, 'rank': rank})
    return cards


def shuffle_deck(deck):
    random.shuffle(deck)
    return deck


def create_empty_hint_info(size):
    return {
        'known_suits': [None] * size,
        'known_ranks': [None] * size,
        'card_ages': [0] * size,
    }


def create_game(fuse_tokens=None):
    deck = shuffle_deck(create_deck())
    players = []
    for i in range(NUM_PLAYERS):
        hand = deck[:HAND_SIZE]
        deck = deck[HAND_SIZE:]
        players.append({
            'id': i,
            'is_human': i == 0,
            'hand': hand,
            'hint_info': create_empty_hint_info(HAND_SIZE),
        })

    return {
        'players': players,
        'deck': deck,
        'play_area': [0] * 5,  # indexed by suit_idx, value = highest rank played
        'discard_pile': [],
        'info_tokens': MAX_INFO_TOKENS,
        'fuse_tokens': fuse_tokens if fuse_tokens is not None else INITIAL_FUSE_TOKENS,
        'current_player': 0,
        'turns_remaining': None,
        'status': 'playing',
    }


def get_score(state):
    return sum(state['play_area'])


def _remove_card_from_hints(hint_info, idx):
    hint_info['known_suits'].pop(idx)
    hint_info['known_ranks'].pop(idx)
    hint_info['card_ages'].pop(idx)


def _add_card_to_hints(hint_info):
    hint_info['known_suits'].append(None)
    hint_info['known_ranks'].append(None)
    hint_info['card_ages'].append(0)


def _apply_hint(player, hint):
    """hint = {'kind': 'suit'|'rank', 'value': int}"""
    for i, card in enumerate(player['hand']):
        if hint['kind'] == 'suit' and card['suit'] == hint['value']:
            player['hint_info']['known_suits'][i] = hint['value']
            player['hint_info']['card_ages'][i] = 0
        elif hint['kind'] == 'rank' and card['rank'] == hint['value']:
            player['hint_info']['known_ranks'][i] = hint['value']
            player['hint_info']['card_ages'][i] = 0


def _check_unwinnable(state):
    for suit_idx in range(5):
        needed = state['play_area'][suit_idx] + 1
        if needed > 5:
            continue
        for rank in range(needed, 6):
            total = RANK_COPIES[rank]
            discarded = sum(1 for c in state['discard_pile']
                           if c['suit'] == suit_idx and c['rank'] == rank)
            if discarded >= total:
                return True
    return False


def is_valid_action(state, action):
    if state['status'] != 'playing':
        return False
    if action['player'] != state['current_player']:
        return False

    player = state['players'][action['player']]
    atype = action['type']

    if atype == 'play' or atype == 'discard':
        return 0 <= action['card_idx'] < len(player['hand'])

    if atype == 'hint':
        if state['info_tokens'] <= 0:
            return False
        if action['target'] == action['player']:
            return False
        target = state['players'][action['target']]
        if len(target['hand']) == 0:
            return False
        hint = action['hint']
        if hint['kind'] == 'suit':
            return any(c['suit'] == hint['value'] for c in target['hand'])
        else:
            return any(c['rank'] == hint['value'] for c in target['hand'])

    return False


def get_valid_actions(state, player_idx):
    if state['status'] != 'playing' or state['current_player'] != player_idx:
        return []

    actions = []
    player = state['players'][player_idx]

    for i in range(len(player['hand'])):
        actions.append({'type': 'play', 'player': player_idx, 'card_idx': i})
        actions.append({'type': 'discard', 'player': player_idx, 'card_idx': i})

    if state['info_tokens'] > 0:
        for t in range(NUM_PLAYERS):
            if t == player_idx:
                continue
            target = state['players'][t]
            suits_in_hand = set(c['suit'] for c in target['hand'])
            ranks_in_hand = set(c['rank'] for c in target['hand'])
            for s in suits_in_hand:
                actions.append({
                    'type': 'hint', 'player': player_idx, 'target': t,
                    'hint': {'kind': 'suit', 'value': s}
                })
            for r in ranks_in_hand:
                actions.append({
                    'type': 'hint', 'player': player_idx, 'target': t,
                    'hint': {'kind': 'rank', 'value': r}
                })

    return actions


def apply_action(state, action):
    """Mutates state in place and returns it. Returns state."""
    if not is_valid_action(state, action):
        return state

    atype = action['type']

    if atype == 'play':
        player = state['players'][action['player']]
        card = player['hand'].pop(action['card_idx'])
        _remove_card_from_hints(player['hint_info'], action['card_idx'])

        expected = state['play_area'][card['suit']] + 1
        success = card['rank'] == expected

        if success:
            state['play_area'][card['suit']] = card['rank']
            if card['rank'] == 5 and state['info_tokens'] < MAX_INFO_TOKENS:
                state['info_tokens'] += 1
        else:
            state['fuse_tokens'] -= 1
            state['discard_pile'].append(card)

        # Draw
        if state['deck']:
            player['hand'].append(state['deck'].pop(0))
            _add_card_to_hints(player['hint_info'])

        if state['fuse_tokens'] <= 0:
            state['status'] = 'lost'
            return state

        if not success and _check_unwinnable(state):
            state['status'] = 'lost'
            return state

        if get_score(state) == 25:
            state['status'] = 'won'
            return state

    elif atype == 'discard':
        player = state['players'][action['player']]
        card = player['hand'].pop(action['card_idx'])
        _remove_card_from_hints(player['hint_info'], action['card_idx'])
        state['discard_pile'].append(card)

        if state['info_tokens'] < MAX_INFO_TOKENS:
            state['info_tokens'] += 1

        if state['deck']:
            player['hand'].append(state['deck'].pop(0))
            _add_card_to_hints(player['hint_info'])

        if _check_unwinnable(state):
            state['status'] = 'lost'
            return state

    elif atype == 'hint':
        _apply_hint(state['players'][action['target']], action['hint'])
        state['info_tokens'] -= 1

    # End-game countdown
    if not state['deck'] and state['turns_remaining'] is None:
        # Each player gets exactly one more turn after deck empties.
        # +1 because we decrement immediately below on this same turn.
        state['turns_remaining'] = NUM_PLAYERS + 1

    if state['turns_remaining'] is not None:
        state['turns_remaining'] -= 1
        if state['turns_remaining'] <= 0:
            state['status'] = 'finished'
            return state

    # Age all cards
    for p in state['players']:
        p['hint_info']['card_ages'] = [a + 1 for a in p['hint_info']['card_ages']]

    # Advance turn
    state['current_player'] = (state['current_player'] + 1) % NUM_PLAYERS
    return state


def clone_state(state):
    """Deep copy for when we need immutable semantics."""
    return deepcopy(state)


if __name__ == '__main__':
    # Quick smoke test: play 1000 random games
    scores = []
    for _ in range(1000):
        s = create_game()
        while s['status'] == 'playing':
            actions = get_valid_actions(s, s['current_player'])
            if not actions:
                break
            action = random.choice(actions)
            apply_action(s, action)
        scores.append(get_score(s))
    avg = sum(scores) / len(scores)
    print(f"Random bot: 1000 games, avg score = {avg:.2f}, "
          f"min = {min(scores)}, max = {max(scores)}")
