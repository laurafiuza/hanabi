/**
 * Information Set Monte Carlo Tree Search (ISMCTS) for Hanabi.
 *
 * Handles partial observability via determinization: sample possible hands
 * for the acting player consistent with their hint info, then run UCT search
 * with heuristic rollouts. One shared tree across all determinizations.
 */
import type { GameState, Action, Card, Suit, Rank, Player } from './types';
import { SUITS, RANKS } from './types';
import { RANK_COUNTS } from './deck';
import { applyAction, getValidActions, getScore } from './game';
import { heuristicBotAction } from './ai';

const TIME_BUDGET_MS = 5000;
const EXPLORATION_C = 1.0;

// ---------------------------------------------------------------------------
// Tree node
// ---------------------------------------------------------------------------

class MCTSNode {
  parent: MCTSNode | null;
  action: Action | null;
  children: Map<string, MCTSNode> = new Map();
  visits = 0;
  totalScore = 0;

  constructor(parent: MCTSNode | null, action: Action | null) {
    this.parent = parent;
    this.action = action;
  }
}

// ---------------------------------------------------------------------------
// Action identity key (for shared tree across determinizations)
// ---------------------------------------------------------------------------

function actionKey(action: Action): string {
  switch (action.type) {
    case 'PLAY_CARD':
      return `P${action.cardIndex}`;
    case 'DISCARD':
      return `D${action.cardIndex}`;
    case 'GIVE_HINT': {
      const h = action.hint;
      const hk = h.kind === 'suit' ? `s${h.suit}` : `r${h.rank}`;
      return `H${action.targetPlayerIndex}${hk}`;
    }
  }
}

// ---------------------------------------------------------------------------
// State cloning (deep enough for simulation — cards are never mutated)
// ---------------------------------------------------------------------------

function cloneGameState(state: GameState): GameState {
  return {
    ...state,
    players: state.players.map(p => ({
      ...p,
      hand: [...p.hand],
      hintInfo: {
        knownSuits: [...p.hintInfo.knownSuits],
        knownRanks: [...p.hintInfo.knownRanks],
      },
    })),
    deck: [...state.deck],
    playArea: { ...state.playArea },
    discardPile: [...state.discardPile],
    history: [...state.history],
  };
}

// ---------------------------------------------------------------------------
// Determinization: sample unknown cards for the acting player
// ---------------------------------------------------------------------------

function shuffle<T>(arr: T[]): T[] {
  for (let i = arr.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [arr[i], arr[j]] = [arr[j], arr[i]];
  }
  return arr;
}

function buildUnknownPool(state: GameState, playerIndex: number): Card[] {
  // Count all 50 cards by suit+rank
  const counts = new Map<string, number>();
  for (const suit of SUITS) {
    for (const rank of RANKS) {
      counts.set(`${suit}-${rank}`, RANK_COUNTS[rank]);
    }
  }

  // Subtract play area
  for (const suit of SUITS) {
    for (let r = 1; r <= state.playArea[suit]; r++) {
      const key = `${suit}-${r}`;
      counts.set(key, counts.get(key)! - 1);
    }
  }

  // Subtract discard pile
  for (const card of state.discardPile) {
    const key = `${card.suit}-${card.rank}`;
    counts.set(key, counts.get(key)! - 1);
  }

  // Subtract other players' hands (visible to us)
  for (let i = 0; i < state.players.length; i++) {
    if (i === playerIndex) continue;
    for (const card of state.players[i].hand) {
      const key = `${card.suit}-${card.rank}`;
      counts.set(key, counts.get(key)! - 1);
    }
  }

  // Build pool from remaining counts
  const pool: Card[] = [];
  let idCounter = 0;
  for (const [key, count] of counts) {
    const [suit, rankStr] = key.split('-');
    for (let i = 0; i < count; i++) {
      pool.push({
        id: `unk-${idCounter++}`,
        suit: suit as Suit,
        rank: Number(rankStr) as Rank,
      });
    }
  }

  return pool;
}

function assignConsistentHand(
  pool: Card[],
  player: Player,
): { hand: Card[]; remaining: Card[] } | null {
  const handSize = player.hand.length;
  const available = [...pool];
  const hand: Card[] = [];

  for (let i = 0; i < handSize; i++) {
    const knownSuit = player.hintInfo.knownSuits[i];
    const knownRank = player.hintInfo.knownRanks[i];

    const candidates = available.filter(
      c =>
        (knownSuit === null || c.suit === knownSuit) &&
        (knownRank === null || c.rank === knownRank),
    );

    if (candidates.length === 0) return null; // rejection — retry

    const pick = candidates[Math.floor(Math.random() * candidates.length)];
    hand.push(pick);
    // Remove the picked card from available (by index, to handle duplicates)
    const idx = available.indexOf(pick);
    available.splice(idx, 1);
  }

  return { hand, remaining: available };
}

function sampleDeterminization(state: GameState, playerIndex: number): GameState | null {
  const clone = cloneGameState(state);
  const pool = buildUnknownPool(state, playerIndex);
  shuffle(pool);

  const result = assignConsistentHand(pool, state.players[playerIndex]);
  if (!result) return null;

  clone.players[playerIndex] = {
    ...clone.players[playerIndex],
    hand: result.hand,
  };
  clone.deck = shuffle(result.remaining);

  return clone;
}

// ---------------------------------------------------------------------------
// Fast path: skip MCTS when the heuristic answer is obviously correct
// ---------------------------------------------------------------------------

function hasObviousPlay(state: GameState, playerIndex: number): boolean {
  const me = state.players[playerIndex];
  for (let i = 0; i < me.hand.length; i++) {
    const knownSuit = me.hintInfo.knownSuits[i];
    const knownRank = me.hintInfo.knownRanks[i];
    // Known suit+rank and playable
    if (knownSuit && knownRank && state.playArea[knownSuit] === knownRank - 1) {
      return true;
    }
    // Known rank, universally playable (e.g., known 1 when no 1s played)
    if (knownRank && !knownSuit) {
      const allReady = SUITS.every(s => state.playArea[s] >= knownRank || state.playArea[s] === knownRank - 1);
      const someNeed = SUITS.some(s => state.playArea[s] === knownRank - 1);
      if (allReady && someNeed) return true;
    }
  }
  return false;
}

// ---------------------------------------------------------------------------
// Action pruning: remove provably bad actions from MCTS search space
// ---------------------------------------------------------------------------

const RANK_COPIES: Record<number, number> = { 1: 3, 2: 2, 3: 2, 4: 2, 5: 1 };

function pruneActions(actions: Action[], state: GameState, playerIndex: number): Action[] {
  const me = state.players[playerIndex];

  const dominated = new Set<string>();

  for (const a of actions) {
    const key = actionKey(a);

    if (a.type === 'DISCARD') {
      const knownRank = me.hintInfo.knownRanks[a.cardIndex];
      const knownSuit = me.hintInfo.knownSuits[a.cardIndex];

      // Never discard a known 5
      if (knownRank === 5) { dominated.add(key); continue; }

      // Never discard a known-playable card
      if (knownSuit && knownRank && state.playArea[knownSuit] === knownRank - 1) {
        dominated.add(key); continue;
      }

      // Never discard a universally-playable rank-only card
      if (knownRank && !knownSuit) {
        const allReady = SUITS.every(s => state.playArea[s] >= knownRank || state.playArea[s] === knownRank - 1);
        const someNeed = SUITS.some(s => state.playArea[s] === knownRank - 1);
        if (allReady && someNeed) { dominated.add(key); continue; }
      }

      // Never discard a card known to be critical (last copy, still needed)
      if (knownSuit && knownRank) {
        if (state.playArea[knownSuit] < knownRank) {
          const discarded = state.discardPile.filter(
            c => c.suit === knownSuit && c.rank === knownRank
          ).length;
          if (discarded >= RANK_COPIES[knownRank] - 1) {
            dominated.add(key); continue;
          }
        }
      }
    }

    if (a.type === 'PLAY_CARD') {
      const knownRank = me.hintInfo.knownRanks[a.cardIndex];
      const knownSuit = me.hintInfo.knownSuits[a.cardIndex];

      // Never play a card known to be useless (already played)
      if (knownSuit && knownRank && state.playArea[knownSuit] >= knownRank) {
        dominated.add(key); continue;
      }

      // Never play a card known to be unplayable (wrong rank for its suit)
      if (knownSuit && knownRank && state.playArea[knownSuit] !== knownRank - 1) {
        dominated.add(key); continue;
      }

      // Never play a card with known rank that can't be playable in any suit
      if (knownRank && !knownSuit) {
        const anyPlayable = SUITS.some(s => state.playArea[s] === knownRank - 1);
        if (!anyPlayable) { dominated.add(key); continue; }
      }

      // Don't play a card with no rank info (pure gamble) — unless endgame
      if (!knownRank && state.turnsRemaining === null) {
        dominated.add(key); continue;
      }
    }

    if (a.type === 'GIVE_HINT') {
      // Don't hint about a rank/suit where every matching card in target's hand
      // is already fully known to them
      const target = state.players[a.targetPlayerIndex];
      const hint = a.hint;
      let anyNewInfo = false;
      for (let i = 0; i < target.hand.length; i++) {
        const card = target.hand[i];
        const matches = hint.kind === 'suit'
          ? card.suit === hint.suit
          : card.rank === hint.rank;
        if (matches) {
          const alreadyKnown = hint.kind === 'suit'
            ? target.hintInfo.knownSuits[i] === hint.suit
            : target.hintInfo.knownRanks[i] === hint.rank;
          if (!alreadyKnown) { anyNewInfo = true; break; }
        }
      }
      if (!anyNewInfo) { dominated.add(key); continue; }
    }
  }

  const filtered = actions.filter(a => !dominated.has(actionKey(a)));
  // Safety: never prune all actions
  return filtered.length > 0 ? filtered : actions;
}

// ---------------------------------------------------------------------------
// UCT tree policy
// ---------------------------------------------------------------------------

function uctValue(child: MCTSNode, parentVisits: number): number {
  if (child.visits === 0) return Infinity;
  return (
    child.totalScore / child.visits +
    EXPLORATION_C * Math.sqrt(Math.log(parentVisits) / child.visits)
  );
}

function treeSelectPruned(
  root: MCTSNode,
  state: GameState,
  prunedRootActions: Action[],
  _rootPlayer: number,
): { node: MCTSNode; state: GameState; isRoot: boolean } {
  let node = root;
  let simState = state;
  let isRoot = true;

  while (simState.status === 'playing') {
    // At root, use pruned actions; deeper nodes use full valid actions
    const actions = isRoot
      ? prunedRootActions.filter(a => getValidActions(simState, simState.currentPlayerIndex).some(
          va => actionKey(va) === actionKey(a)))
      : getValidActions(simState, simState.currentPlayerIndex);

    if (actions.length === 0) break;

    const untriedActions = actions.filter(a => !node.children.has(actionKey(a)));
    if (untriedActions.length > 0) break; // expand phase will handle

    let bestChild: MCTSNode | null = null;
    let bestValue = -Infinity;

    for (const a of actions) {
      const key = actionKey(a);
      const child = node.children.get(key);
      if (!child) continue;
      const value = uctValue(child, node.visits);
      if (value > bestValue) {
        bestValue = value;
        bestChild = child;
      }
    }

    if (!bestChild) break;

    node = bestChild;
    simState = applyAction(simState, bestChild.action!);
    isRoot = false;
  }

  return { node, state: simState, isRoot };
}

// ---------------------------------------------------------------------------
// Expand: add one untried child (pruned at root level)
// ---------------------------------------------------------------------------

function expandPruned(
  node: MCTSNode,
  state: GameState,
  prunedActions: Action[] | null,
  _rootPlayer: number,
): { node: MCTSNode; state: GameState } {
  if (state.status !== 'playing') return { node, state };

  const allActions = getValidActions(state, state.currentPlayerIndex);
  const actions = prunedActions
    ? prunedActions.filter(a => allActions.some(va => actionKey(va) === actionKey(a)))
    : allActions;
  const untriedActions = actions.filter(a => !node.children.has(actionKey(a)));

  if (untriedActions.length === 0) return { node, state };

  const action = untriedActions[Math.floor(Math.random() * untriedActions.length)];
  const child = new MCTSNode(node, action);
  node.children.set(actionKey(action), child);

  return { node: child, state: applyAction(state, action) };
}

// ---------------------------------------------------------------------------
// Rollout: play to completion using heuristic
// ---------------------------------------------------------------------------

function rollout(state: GameState): number {
  let simState = state;
  while (simState.status === 'playing') {
    const action = heuristicBotAction(simState, simState.currentPlayerIndex);
    simState = applyAction(simState, action);
  }
  return getScore(simState.playArea) / 25; // normalize to [0,1]
}

// ---------------------------------------------------------------------------
// Backpropagate
// ---------------------------------------------------------------------------

function backpropagate(node: MCTSNode | null, score: number): void {
  while (node !== null) {
    node.visits++;
    node.totalScore += score;
    node = node.parent;
  }
}

// ---------------------------------------------------------------------------
// Main entry point
// ---------------------------------------------------------------------------

const YIELD_INTERVAL_MS = 50; // yield to event loop every 50ms for UI responsiveness

function yieldToEventLoop(): Promise<void> {
  return new Promise(resolve => setTimeout(resolve, 0));
}

export async function chooseMCTSAction(state: GameState, playerIndex: number): Promise<Action> {
  // Fast path: obvious plays don't need search
  if (hasObviousPlay(state, playerIndex)) {
    return heuristicBotAction(state, playerIndex);
  }

  // Pre-compute pruned action set for the root player
  const allActions = getValidActions(state, playerIndex);
  const prunedActions = pruneActions(allActions, state, playerIndex);

  // If only one action survives pruning, just do it
  if (prunedActions.length === 1) {
    return prunedActions[0];
  }

  const root = new MCTSNode(null, null);
  const deadline = performance.now() + TIME_BUDGET_MS;
  let simCount = 0;
  let lastYield = performance.now();

  while (performance.now() < deadline) {
    // Yield to event loop periodically so UI stays responsive
    if (performance.now() - lastYield > YIELD_INTERVAL_MS) {
      await yieldToEventLoop();
      lastYield = performance.now();
    }

    // 1. Determinize
    const det = sampleDeterminization(state, playerIndex);
    if (!det) continue; // rejection — retry

    // 2. Select (using pruned actions at root level)
    const selected = treeSelectPruned(root, det, prunedActions, playerIndex);

    // 3. Expand (using pruned actions at root level)
    const expanded = expandPruned(selected.node, selected.state, selected.isRoot ? prunedActions : null, playerIndex);

    // 4. Rollout
    const score = rollout(expanded.state);

    // 5. Backpropagate
    backpropagate(expanded.node, score);

    simCount++;
  }

  // Pick action with highest visit count
  let bestAction: Action | null = null;
  let bestVisits = -1;

  for (const [, child] of root.children) {
    if (child.visits > bestVisits) {
      bestVisits = child.visits;
      bestAction = child.action;
    }
  }

  if (!bestAction) {
    return heuristicBotAction(state, playerIndex);
  }

  // Safety: verify the chosen action is still valid in the real state
  if (!allActions.some(a => actionKey(a) === actionKey(bestAction!))) {
    return heuristicBotAction(state, playerIndex);
  }

  return bestAction;
}
