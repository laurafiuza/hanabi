/**
 * Information Set Monte Carlo Tree Search (ISMCTS) for Hanabi.
 *
 * Handles partial observability via determinization: sample possible hands
 * for the acting player consistent with their hint info, then run UCT search
 * with heuristic rollouts. One shared tree across all determinizations.
 *
 * Uses the tier-based action scoring from ai.ts:
 * - "bad" actions are filtered out
 * - "must" and "strong" actions are explored before "neutral"
 */
import type { GameState, Action, Card, Suit, Rank, Player } from './types';
import { SUITS, RANKS } from './types';
import { RANK_COUNTS } from './deck';
import { applyAction, getValidActions, getScore } from './game';
import { heuristicBotAction, scoreAction, TIER_ORDER } from './ai';
import type { ActionTier } from './ai';

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
        cardAges: [...p.hintInfo.cardAges],
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
  const counts = new Map<string, number>();
  for (const suit of SUITS) {
    for (const rank of RANKS) {
      counts.set(`${suit}-${rank}`, RANK_COUNTS[rank]);
    }
  }

  for (const suit of SUITS) {
    for (let r = 1; r <= state.playArea[suit]; r++) {
      const key = `${suit}-${r}`;
      counts.set(key, counts.get(key)! - 1);
    }
  }

  for (const card of state.discardPile) {
    const key = `${card.suit}-${card.rank}`;
    counts.set(key, counts.get(key)! - 1);
  }

  for (let i = 0; i < state.players.length; i++) {
    if (i === playerIndex) continue;
    for (const card of state.players[i].hand) {
      const key = `${card.suit}-${card.rank}`;
      counts.set(key, counts.get(key)! - 1);
    }
  }

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

    if (candidates.length === 0) return null;

    const pick = candidates[Math.floor(Math.random() * candidates.length)];
    hand.push(pick);
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
// Tier-based action filtering (replaces pruneActions)
// ---------------------------------------------------------------------------

interface RankedAction {
  action: Action;
  tier: ActionTier;
  score: number;
}

function filterAndRankActions(actions: Action[], state: GameState, playerIndex: number): RankedAction[] {
  const scored: RankedAction[] = actions.map(a => {
    const s = scoreAction(state, playerIndex, a);
    return { action: a, tier: s.tier, score: s.score };
  });

  // Filter out "bad" tier
  const filtered = scored.filter(s => s.tier !== 'bad');

  // Sort by tier (must > strong > neutral) then score descending
  filtered.sort((a, b) => {
    const tierDiff = TIER_ORDER[b.tier] - TIER_ORDER[a.tier];
    if (tierDiff !== 0) return tierDiff;
    return b.score - a.score;
  });

  // Safety: never filter all actions
  return filtered.length > 0 ? filtered : scored;
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
  rankedRootActions: RankedAction[],
  _rootPlayer: number,
): { node: MCTSNode; state: GameState; isRoot: boolean } {
  let node = root;
  let simState = state;
  let isRoot = true;

  while (simState.status === 'playing') {
    const actions = isRoot
      ? rankedRootActions
          .map(r => r.action)
          .filter(a => getValidActions(simState, simState.currentPlayerIndex).some(
            va => actionKey(va) === actionKey(a)))
      : getValidActions(simState, simState.currentPlayerIndex);

    if (actions.length === 0) break;

    const untriedActions = actions.filter(a => !node.children.has(actionKey(a)));
    if (untriedActions.length > 0) break;

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
// Expand: add one untried child
// At root: biased expansion (try highest-tier actions first)
// At deeper nodes: random expansion
// ---------------------------------------------------------------------------

function expand(
  node: MCTSNode,
  state: GameState,
  rankedRootActions: RankedAction[] | null,
  _rootPlayer: number,
): { node: MCTSNode; state: GameState } {
  if (state.status !== 'playing') return { node, state };

  const allActions = getValidActions(state, state.currentPlayerIndex);

  let untriedActions: Action[];

  if (rankedRootActions) {
    // Root level: use ranked actions, biased order (already sorted by tier+score)
    const validRanked = rankedRootActions
      .map(r => r.action)
      .filter(a => allActions.some(va => actionKey(va) === actionKey(a)));
    untriedActions = validRanked.filter(a => !node.children.has(actionKey(a)));
  } else {
    // Deeper nodes: random selection
    untriedActions = allActions.filter(a => !node.children.has(actionKey(a)));
  }

  if (untriedActions.length === 0) return { node, state };

  // At root: pick the first untried (highest-tier/score due to sorting)
  // At deeper nodes: pick randomly
  const action = rankedRootActions
    ? untriedActions[0]
    : untriedActions[Math.floor(Math.random() * untriedActions.length)];

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
  return getScore(simState.playArea) / 25;
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

const YIELD_INTERVAL_MS = 50;

function yieldToEventLoop(): Promise<void> {
  return new Promise(resolve => setTimeout(resolve, 0));
}

export async function chooseMCTSAction(state: GameState, playerIndex: number): Promise<Action> {
  const allActions = getValidActions(state, playerIndex);
  const ranked = filterAndRankActions(allActions, state, playerIndex);

  // Fast path: if exactly one "must" action, take it immediately
  const mustActions = ranked.filter(r => r.tier === 'must');
  if (mustActions.length === 1) return mustActions[0].action;

  // If only one action survives filtering, just do it
  if (ranked.length === 1) return ranked[0].action;

  const root = new MCTSNode(null, null);
  const deadline = performance.now() + TIME_BUDGET_MS;
  let lastYield = performance.now();

  while (performance.now() < deadline) {
    if (performance.now() - lastYield > YIELD_INTERVAL_MS) {
      await yieldToEventLoop();
      lastYield = performance.now();
    }

    // 1. Determinize
    const det = sampleDeterminization(state, playerIndex);
    if (!det) continue;

    // 2. Select (using ranked actions at root level)
    const selected = treeSelect(root, det, ranked, playerIndex);

    // 3. Expand (biased at root, random at deeper nodes)
    const expanded = expand(selected.node, selected.state, selected.isRoot ? ranked : null, playerIndex);

    // 4. Rollout
    const score = rollout(expanded.state);

    // 5. Backpropagate
    backpropagate(expanded.node, score);
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
