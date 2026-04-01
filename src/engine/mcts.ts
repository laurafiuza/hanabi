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

const TIME_BUDGET_MS = 1200;
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
// Fast path: skip MCTS when heuristic has a certain play
// ---------------------------------------------------------------------------

function hasCertainPlay(state: GameState, playerIndex: number): boolean {
  const me = state.players[playerIndex];
  for (let i = 0; i < me.hand.length; i++) {
    const knownSuit = me.hintInfo.knownSuits[i];
    const knownRank = me.hintInfo.knownRanks[i];
    if (knownSuit && knownRank && state.playArea[knownSuit] === knownRank - 1) {
      return true;
    }
  }
  return false;
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

function treeSelect(
  root: MCTSNode,
  state: GameState,
): { node: MCTSNode; state: GameState } {
  let node = root;
  let simState = state;

  while (simState.status === 'playing') {
    const actions = getValidActions(simState, simState.currentPlayerIndex);
    if (actions.length === 0) break;

    // Find untried actions in this determinization
    const untriedActions = actions.filter(a => !node.children.has(actionKey(a)));
    if (untriedActions.length > 0) break; // expand phase will handle

    // All actions tried — select best via UCT among those legal here
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
  }

  return { node, state: simState };
}

// ---------------------------------------------------------------------------
// Expand: add one untried child
// ---------------------------------------------------------------------------

function expand(
  node: MCTSNode,
  state: GameState,
): { node: MCTSNode; state: GameState } {
  if (state.status !== 'playing') return { node, state };

  const actions = getValidActions(state, state.currentPlayerIndex);
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

export function chooseMCTSAction(state: GameState, playerIndex: number): Action {
  // Fast path: if we have a guaranteed safe play, just do it
  if (hasCertainPlay(state, playerIndex)) {
    return heuristicBotAction(state, playerIndex);
  }

  const root = new MCTSNode(null, null);
  const deadline = performance.now() + TIME_BUDGET_MS;
  let simCount = 0;

  while (performance.now() < deadline) {
    // 1. Determinize
    const det = sampleDeterminization(state, playerIndex);
    if (!det) continue; // rejection — retry

    // 2. Select
    const selected = treeSelect(root, det);

    // 3. Expand
    const expanded = expand(selected.node, selected.state);

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
    // Fallback if no simulations completed (shouldn't happen)
    return heuristicBotAction(state, playerIndex);
  }

  return bestAction;
}
