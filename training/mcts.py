"""
Information Set Monte Carlo Tree Search (ISMCTS) for Hanabi — Python port.

Used for ExIt data generation: run MCTS with neural rollout, collect
visit distributions at the root as soft training targets.
"""
import math
import time
import random
from copy import deepcopy

from engine import (
    SUITS, RANKS, RANK_COPIES, NUM_PLAYERS, HAND_SIZE,
    create_empty_hint_info, apply_action, get_valid_actions, get_score, clone_state
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

EXPLORATION_C = 1.0


# ---------------------------------------------------------------------------
# Tree node
# ---------------------------------------------------------------------------

class MCTSNode:
    __slots__ = ['parent', 'action_key', 'action', 'children', 'visits', 'total_score']

    def __init__(self, parent=None, action_key=None, action=None):
        self.parent = parent
        self.action_key = action_key
        self.action = action  # the game action dict that led to this node
        self.children = {}    # action_key -> MCTSNode
        self.visits = 0
        self.total_score = 0.0


# ---------------------------------------------------------------------------
# Action key (for shared tree across determinizations)
# ---------------------------------------------------------------------------

def action_key(action):
    atype = action['type']
    if atype == 'play':
        return f"P{action['card_idx']}"
    elif atype == 'discard':
        return f"D{action['card_idx']}"
    elif atype == 'hint':
        h = action['hint']
        hk = f"s{h['value']}" if h['kind'] == 'suit' else f"r{h['value']}"
        return f"H{action['target']}{hk}"


# ---------------------------------------------------------------------------
# Determinization: sample unknown cards for the acting player
# ---------------------------------------------------------------------------

def build_unknown_pool(state, player_idx):
    """Build pool of cards not visible to player_idx."""
    # Start with all 50 cards
    counts = {}
    for s in range(len(SUITS)):
        for r in RANKS:
            counts[(s, r)] = RANK_COPIES[r]

    # Subtract play area
    for s in range(len(SUITS)):
        for r in range(1, state['play_area'][s] + 1):
            counts[(s, r)] -= 1

    # Subtract discard pile
    for card in state['discard_pile']:
        counts[(card['suit'], card['rank'])] -= 1

    # Subtract other players' hands (visible)
    for i in range(NUM_PLAYERS):
        if i == player_idx:
            continue
        for card in state['players'][i]['hand']:
            counts[(card['suit'], card['rank'])] -= 1

    # Build pool
    pool = []
    for (s, r), count in counts.items():
        for _ in range(count):
            pool.append({'suit': s, 'rank': r})

    return pool


def assign_consistent_hand(pool, player):
    """Try to assign cards from pool consistent with player's hints."""
    hand_size = len(player['hand'])
    available = list(pool)
    hand = []

    for i in range(hand_size):
        ks = player['hint_info']['known_suits'][i]
        kr = player['hint_info']['known_ranks'][i]

        candidates = [c for c in available
                      if (ks is None or c['suit'] == ks)
                      and (kr is None or c['rank'] == kr)]

        if not candidates:
            return None  # rejection — retry

        pick = random.choice(candidates)
        hand.append(pick)
        available.remove(pick)

    return hand, available


def sample_determinization(state, player_idx):
    """Create a determinized copy where player_idx's hand is sampled."""
    clone = clone_state(state)
    pool = build_unknown_pool(state, player_idx)
    random.shuffle(pool)

    result = assign_consistent_hand(pool, state['players'][player_idx])
    if result is None:
        return None

    hand, remaining = result
    clone['players'][player_idx]['hand'] = hand
    # Keep hint info from original (it's about what the player knows, not the actual cards)
    random.shuffle(remaining)
    clone['deck'] = remaining

    return clone


# ---------------------------------------------------------------------------
# Action pruning (same logic as TypeScript mcts.ts)
# ---------------------------------------------------------------------------

def filter_and_rank_actions(actions, state, player_idx):
    """Filter out 'bad' tier actions and sort by tier+score (best first)."""
    from heuristic_bot import score_action, TIER_ORDER

    scored = []
    for a in actions:
        tier, sc = score_action(state, player_idx, a)
        scored.append((a, tier, sc))

    # Filter out bad tier
    filtered = [(a, t, s) for a, t, s in scored if t != 'bad']

    # Sort by tier desc, then score desc
    filtered.sort(key=lambda x: (TIER_ORDER[x[1]], x[2]), reverse=True)

    # Safety: never filter all actions
    if not filtered:
        scored.sort(key=lambda x: (TIER_ORDER[x[1]], x[2]), reverse=True)
        return [a for a, _, _ in scored]

    return [a for a, _, _ in filtered]


# ---------------------------------------------------------------------------
# UCT
# ---------------------------------------------------------------------------

def uct_value(child, parent_visits):
    if child.visits == 0:
        return float('inf')
    return (child.total_score / child.visits +
            EXPLORATION_C * math.sqrt(math.log(parent_visits) / child.visits))


def tree_select(root, state, pruned_root_actions):
    node = root
    sim_state = state
    is_root = True

    while sim_state['status'] == 'playing':
        if is_root:
            valid = get_valid_actions(sim_state, sim_state['current_player'])
            valid_keys = {action_key(a) for a in valid}
            actions = [a for a in pruned_root_actions if action_key(a) in valid_keys]
        else:
            actions = get_valid_actions(sim_state, sim_state['current_player'])

        if not actions:
            break

        untried = [a for a in actions if action_key(a) not in node.children]
        if untried:
            break

        best_child = None
        best_value = -float('inf')
        for a in actions:
            key = action_key(a)
            child = node.children.get(key)
            if child is None:
                continue
            val = uct_value(child, node.visits)
            if val > best_value:
                best_value = val
                best_child = child
                best_action = a

        if best_child is None:
            break

        node = best_child
        apply_action(sim_state, best_action)
        is_root = False

    return node, sim_state, is_root


def expand(node, state, ranked_root_actions):
    if state['status'] != 'playing':
        return node, state

    all_actions = get_valid_actions(state, state['current_player'])
    if ranked_root_actions is not None:
        # Root level: use ranked actions (already sorted by tier+score)
        valid_keys = {action_key(a) for a in all_actions}
        actions = [a for a in ranked_root_actions if action_key(a) in valid_keys]
    else:
        actions = all_actions

    untried = [a for a in actions if action_key(a) not in node.children]
    if not untried:
        return node, state

    # At root: pick first untried (biased toward best tier+score)
    # At deeper nodes: pick randomly
    action = untried[0] if ranked_root_actions is not None else random.choice(untried)
    key = action_key(action)
    child = MCTSNode(parent=node, action_key=key, action=action)
    node.children[key] = child

    apply_action(state, action)
    return child, state


def backpropagate(node, score):
    while node is not None:
        node.visits += 1
        node.total_score += score
        node = node.parent


# ---------------------------------------------------------------------------
# Rollout
# ---------------------------------------------------------------------------

def rollout_with_policy(state, policy_fn):
    """Play to completion using policy_fn(state, player_idx) -> action."""
    sim_state = state
    while sim_state['status'] == 'playing':
        action = policy_fn(sim_state, sim_state['current_player'])
        apply_action(sim_state, action)
    return get_score(sim_state) / 25.0


# ---------------------------------------------------------------------------
# Main MCTS
# ---------------------------------------------------------------------------

def run_mcts(state, player_idx, policy_fn, time_budget_ms=200):
    """Run MCTS and return visit distribution over actions.

    Args:
        state: game state (not mutated)
        player_idx: acting player
        policy_fn: rollout policy function(state, player_idx) -> action
        time_budget_ms: search budget in milliseconds

    Returns:
        List of (action, visit_count) tuples for root children.
    """
    all_actions = get_valid_actions(state, player_idx)
    if not all_actions:
        return []

    ranked = filter_and_rank_actions(all_actions, state, player_idx)
    if len(ranked) == 1:
        return [(ranked[0], 1)]

    root = MCTSNode()
    deadline = time.monotonic() + time_budget_ms / 1000.0
    sim_count = 0

    while time.monotonic() < deadline:
        # 1. Determinize
        det = sample_determinization(state, player_idx)
        if det is None:
            continue

        # 2. Select
        node, sim_state, is_root = tree_select(root, det, ranked)

        # 3. Expand (biased at root, random at deeper nodes)
        node, sim_state = expand(node, sim_state, ranked if is_root else None)

        # 4. Rollout
        score = rollout_with_policy(sim_state, policy_fn)

        # 5. Backpropagate
        backpropagate(node, score)
        sim_count += 1

    # Collect visit distribution
    results = []
    for key, child in root.children.items():
        if hasattr(child, 'action'):
            results.append((child.action, child.visits))

    return results
