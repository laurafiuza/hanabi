import type { GameState, Action, Suit, Rank } from './types';
import { SUITS, HAND_SIZE, NUM_PLAYERS, MAX_INFO_TOKENS, INITIAL_FUSE_TOKENS } from './types';
import { getValidActions } from './game';

// Must match training/featurize.py exactly
const OBS_SIZE = 414;
const ACTION_SIZE = 48;
const RANK_COPIES: Record<number, number> = { 1: 3, 2: 2, 3: 2, 4: 2, 5: 1 };
const SUIT_INDEX: Record<Suit, number> = { white: 0, yellow: 1, green: 2, blue: 3, red: 4 };

interface ModelWeights {
  W1: number[][];
  b1: number[];
  W2: number[][];
  b2: number[];
  W3: number[][];
  b3: number[];
}

let cachedWeights: ModelWeights | null = null;

export async function loadWeights(): Promise<ModelWeights> {
  if (cachedWeights) return cachedWeights;
  const resp = await fetch('/model_weights.json');
  cachedWeights = await resp.json();
  return cachedWeights!;
}

export function setWeights(w: ModelWeights) {
  cachedWeights = w;
}

function observe(state: GameState, playerIndex: number): number[] {
  const obs = new Array(OBS_SIZE).fill(0);
  let offset = 0;

  // Play area: 25 (cumulative one-hot)
  for (const suit of SUITS) {
    const si = SUIT_INDEX[suit];
    const level = state.playArea[suit];
    for (let r = 0; r < level; r++) {
      obs[offset + si * 5 + r] = 1;
    }
  }
  offset += 25;

  // Info tokens
  obs[offset] = state.infoTokens / MAX_INFO_TOKENS;
  offset += 1;

  // Fuse tokens
  obs[offset] = state.fuseTokens / Math.max(INITIAL_FUSE_TOKENS, 1);
  offset += 1;

  // Deck fraction
  obs[offset] = state.deck.length / 50;
  offset += 1;

  // My hand hints: 4 cards x 10
  const me = state.players[playerIndex];
  for (let i = 0; i < HAND_SIZE; i++) {
    const co = offset + i * 10;
    if (i < me.hand.length) {
      const ks = me.hintInfo.knownSuits[i];
      const kr = me.hintInfo.knownRanks[i];
      if (ks) obs[co + SUIT_INDEX[ks]] = 1;
      if (kr) obs[co + 5 + (kr - 1)] = 1;
    }
  }
  offset += 40;

  // Other players' actual cards: 4 x 4 x 10 = 160
  for (let d = 1; d < NUM_PLAYERS; d++) {
    const oidx = (playerIndex + d) % NUM_PLAYERS;
    const other = state.players[oidx];
    const pi = d - 1;
    for (let i = 0; i < HAND_SIZE; i++) {
      const co = offset + pi * HAND_SIZE * 10 + i * 10;
      if (i < other.hand.length) {
        const card = other.hand[i];
        obs[co + SUIT_INDEX[card.suit]] = 1;
        obs[co + 5 + (card.rank - 1)] = 1;
      }
    }
  }
  offset += 160;

  // Other players' hint info: 4 x 4 x 10 = 160
  for (let d = 1; d < NUM_PLAYERS; d++) {
    const oidx = (playerIndex + d) % NUM_PLAYERS;
    const other = state.players[oidx];
    const pi = d - 1;
    for (let i = 0; i < HAND_SIZE; i++) {
      const co = offset + pi * HAND_SIZE * 10 + i * 10;
      if (i < other.hand.length) {
        const ks = other.hintInfo.knownSuits[i];
        const kr = other.hintInfo.knownRanks[i];
        if (ks) obs[co + SUIT_INDEX[ks]] = 1;
        if (kr) obs[co + 5 + (kr - 1)] = 1;
      }
    }
  }
  offset += 160;

  // Discard summary: 25
  for (const card of state.discardPile) {
    const idx = SUIT_INDEX[card.suit] * 5 + (card.rank - 1);
    obs[offset + idx] += 1.0 / RANK_COPIES[card.rank];
  }
  offset += 25;

  // Turns remaining
  if (state.turnsRemaining !== null) {
    obs[offset] = state.turnsRemaining / NUM_PLAYERS;
  }

  return obs;
}

function matmul(vec: number[], mat: number[][]): number[] {
  const out = new Array(mat[0].length).fill(0);
  for (let i = 0; i < vec.length; i++) {
    if (vec[i] === 0) continue;
    for (let j = 0; j < out.length; j++) {
      out[j] += vec[i] * mat[i][j];
    }
  }
  return out;
}

function forwardPass(obs: number[], weights: ModelWeights): number[] {
  // Layer 1: ReLU
  let h = matmul(obs, weights.W1);
  h = h.map((v, i) => Math.max(0, v + weights.b1[i]));

  // Layer 2: ReLU
  let h2 = matmul(h, weights.W2);
  h2 = h2.map((v, i) => Math.max(0, v + weights.b2[i]));

  // Layer 3: logits
  let logits = matmul(h2, weights.W3);
  logits = logits.map((v, i) => v + weights.b3[i]);

  return logits;
}

function indexToAction(index: number, playerIndex: number): Action {
  if (index < 4) {
    return { type: 'PLAY_CARD', playerIndex, cardIndex: index };
  } else if (index < 8) {
    return { type: 'DISCARD', playerIndex, cardIndex: index - 4 };
  } else {
    const hintIdx = index - 8;
    const targetOffset = Math.floor(hintIdx / 10) + 1;
    const hintType = hintIdx % 10;
    const targetPlayerIndex = (playerIndex + targetOffset) % NUM_PLAYERS;

    if (hintType < 5) {
      return {
        type: 'GIVE_HINT', playerIndex, targetPlayerIndex,
        hint: { kind: 'suit', suit: SUITS[hintType] },
      };
    } else {
      return {
        type: 'GIVE_HINT', playerIndex, targetPlayerIndex,
        hint: { kind: 'rank', rank: (hintType - 4) as Rank },
      };
    }
  }
}

function actionToIndex(action: Action, playerIndex: number): number {
  if (action.type === 'PLAY_CARD') return action.cardIndex;
  if (action.type === 'DISCARD') return 4 + action.cardIndex;
  // GIVE_HINT
  const targetOffset = (action.targetPlayerIndex - playerIndex + NUM_PLAYERS) % NUM_PLAYERS;
  const hint = action.hint;
  const hintType = hint.kind === 'suit'
    ? SUIT_INDEX[hint.suit]
    : 5 + (hint.rank - 1);
  return 8 + (targetOffset - 1) * 10 + hintType;
}

export function chooseRLAction(state: GameState, playerIndex: number): Action | null {
  if (!cachedWeights) return null;

  const obs = observe(state, playerIndex);
  const logits = forwardPass(obs, cachedWeights);

  // Build validity mask
  const validActions = getValidActions(state, playerIndex);
  const mask = new Array(ACTION_SIZE).fill(false);
  for (const a of validActions) {
    mask[actionToIndex(a, playerIndex)] = true;
  }

  // Masked softmax (greedy — pick highest logit among valid)
  let bestIdx = -1;
  let bestLogit = -Infinity;
  for (let i = 0; i < ACTION_SIZE; i++) {
    if (mask[i] && logits[i] > bestLogit) {
      bestLogit = logits[i];
      bestIdx = i;
    }
  }

  if (bestIdx === -1) return null;
  return indexToAction(bestIdx, playerIndex);
}
