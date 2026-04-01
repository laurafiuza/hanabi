import type { GameState, Action, Card, Rank, Suit } from './types';
import { SUITS, getPlayersAfterInTurnOrder } from './types';
import { getValidActions } from './game';
import { chooseMCTSAction } from './mcts';

const RANK_COPIES: Record<number, number> = { 1: 3, 2: 2, 3: 2, 4: 2, 5: 1 };

// A card is critical if it's the last remaining copy of a card still needed for its suit
function isCritical(card: Card, state: GameState): boolean {
  if (state.playArea[card.suit] >= card.rank) return false; // already played
  const totalCopies = RANK_COPIES[card.rank];
  const discarded = state.discardPile.filter(c => c.suit === card.suit && c.rank === card.rank).length;
  return discarded >= totalCopies - 1; // this is the last copy
}

// Is a card with known rank definitely useless regardless of suit?
function isRankUseless(knownRank: Rank, state: GameState): boolean {
  return SUITS.every(s => state.playArea[s] >= knownRank);
}

// Is a card with known suit+rank definitely useless?
function isCardUseless(suit: Suit, rank: Rank, state: GameState): boolean {
  return state.playArea[suit] >= rank;
}

// Is a card with known suit+rank definitely playable right now?
function isCardPlayable(suit: Suit, rank: Rank, state: GameState): boolean {
  return state.playArea[suit] === rank - 1;
}

export function chooseBotAction(state: GameState, playerIndex: number): Action {
  try {
    return chooseMCTSAction(state, playerIndex);
  } catch {
    return heuristicBotAction(state, playerIndex);
  }
}

export function heuristicBotAction(state: GameState, playerIndex: number): Action {
  const actions = getValidActions(state, playerIndex);
  const me = state.players[playerIndex];

  // 1-2. Find all known-safe plays, pick the lowest rank to open up fronts early
  {
    let bestIdx = -1;
    let bestRank = Infinity;
    for (let i = 0; i < me.hand.length; i++) {
      const knownSuit = me.hintInfo.knownSuits[i];
      const knownRank = me.hintInfo.knownRanks[i];
      // Known suit+rank and playable
      if (knownSuit && knownRank && isCardPlayable(knownSuit, knownRank, state)) {
        if (knownRank < bestRank) { bestRank = knownRank; bestIdx = i; }
      }
      // Known rank only, universally playable
      if (knownRank && !knownSuit) {
        const allSuitsReady = SUITS.every(s => state.playArea[s] >= knownRank || state.playArea[s] === knownRank - 1);
        const someNeedIt = SUITS.some(s => state.playArea[s] === knownRank - 1);
        if (allSuitsReady && someNeedIt && knownRank < bestRank) {
          bestRank = knownRank; bestIdx = i;
        }
      }
    }
    if (bestIdx >= 0) {
      return { type: 'PLAY_CARD', playerIndex, cardIndex: bestIdx };
    }
  }

  // 3. URGENT: Hint about critical cards another player might discard
  if (state.infoTokens > 0) {
    const criticalHint = findCriticalHint(state, playerIndex);
    if (criticalHint) return criticalHint;
  }

  // 4. Hint about 5s — they're unique, always worth protecting early
  if (state.infoTokens > 0) {
    const fiveHint = findFiveHint(state, playerIndex);
    if (fiveHint) return fiveHint;
  }

  // 5. Hint about an immediately playable card in another player's hand
  if (state.infoTokens > 0) {
    const hintAction = findPlayableHint(state, playerIndex);
    if (hintAction) return hintAction;
  }

  // 6. Discard a known-useless card
  //    a) Both suit+rank known, already played past this rank
  for (let i = 0; i < me.hand.length; i++) {
    const knownSuit = me.hintInfo.knownSuits[i];
    const knownRank = me.hintInfo.knownRanks[i];
    if (knownSuit && knownRank && isCardUseless(knownSuit, knownRank, state)) {
      return { type: 'DISCARD', playerIndex, cardIndex: i };
    }
  }
  //    b) Only rank known, but ALL suits have played past it (e.g., known 1 and all 1s played)
  for (let i = 0; i < me.hand.length; i++) {
    const knownRank = me.hintInfo.knownRanks[i];
    if (knownRank && isRankUseless(knownRank, state)) {
      return { type: 'DISCARD', playerIndex, cardIndex: i };
    }
  }

  // 7. If info tokens available, hint about something useful before discarding
  if (state.infoTokens > 0) {
    const anyHint = findAnyUsefulHint(state, playerIndex);
    if (anyHint) return anyHint;
  }

  // 8. Discard the oldest card with no hint info
  //    NEVER discard a known 5 — they're irreplaceable
  for (let i = 0; i < me.hand.length; i++) {
    const knownSuit = me.hintInfo.knownSuits[i];
    const knownRank = me.hintInfo.knownRanks[i];
    if (!knownSuit && !knownRank) {
      return { type: 'DISCARD', playerIndex, cardIndex: i };
    }
  }

  // 9. Discard a card we know the least about, but NEVER a known 5, and prefer
  //    discarding lower-ranked known cards over higher-ranked ones
  {
    let bestIdx = -1;
    let bestPriority = Infinity; // lower = better to discard
    for (let i = 0; i < me.hand.length; i++) {
      const knownRank = me.hintInfo.knownRanks[i];
      // Never discard a known 5
      if (knownRank === 5) continue;
      // Prefer discarding cards where we only know rank (not suit) — less info
      const knownSuit = me.hintInfo.knownSuits[i];
      let priority = knownRank ?? 6; // unknown rank = high priority to keep
      if (knownSuit && knownRank) priority -= 0.5; // full info = slightly prefer keeping
      if (priority < bestPriority) {
        bestPriority = priority;
        bestIdx = i;
      }
    }
    if (bestIdx >= 0) {
      return { type: 'DISCARD', playerIndex, cardIndex: bestIdx };
    }
  }

  // 10. Give any valid hint if we have tokens
  if (state.infoTokens > 0) {
    const hintActions = actions.filter(a => a.type === 'GIVE_HINT');
    if (hintActions.length > 0) {
      return hintActions[0];
    }
  }

  // 11. Last resort: discard oldest card
  return { type: 'DISCARD', playerIndex, cardIndex: 0 };
}

// ---------------------------------------------------------------------------
// Hint finders
// ---------------------------------------------------------------------------

function findCriticalHint(state: GameState, playerIndex: number): Action | null {
  for (const targetIdx of getPlayersAfterInTurnOrder(playerIndex)) {
    const target = state.players[targetIdx];

    for (let i = 0; i < target.hand.length; i++) {
      const card = target.hand[i];
      if (!isCritical(card, state)) continue;

      const knowsSuit = target.hintInfo.knownSuits[i] === card.suit;
      const knowsRank = target.hintInfo.knownRanks[i] === card.rank;

      if (knowsSuit && knowsRank) continue;

      // Prefer rank for critical cards (more distinctive signal)
      if (!knowsRank) {
        return {
          type: 'GIVE_HINT', playerIndex, targetPlayerIndex: targetIdx,
          hint: { kind: 'rank', rank: card.rank },
        };
      }
      if (!knowsSuit) {
        return {
          type: 'GIVE_HINT', playerIndex, targetPlayerIndex: targetIdx,
          hint: { kind: 'suit', suit: card.suit },
        };
      }
    }
  }

  return null;
}

function findFiveHint(state: GameState, playerIndex: number): Action | null {
  for (const targetIdx of getPlayersAfterInTurnOrder(playerIndex)) {
    const target = state.players[targetIdx];

    for (let i = 0; i < target.hand.length; i++) {
      if (target.hand[i].rank === 5 && target.hintInfo.knownRanks[i] !== 5) {
        return {
          type: 'GIVE_HINT', playerIndex, targetPlayerIndex: targetIdx,
          hint: { kind: 'rank', rank: 5 as Rank },
        };
      }
    }
  }

  return null;
}

function findPlayableHint(state: GameState, playerIndex: number): Action | null {
  for (const targetIdx of getPlayersAfterInTurnOrder(playerIndex)) {
    const target = state.players[targetIdx];

    for (let i = 0; i < target.hand.length; i++) {
      const card = target.hand[i];
      if (state.playArea[card.suit] !== card.rank - 1) continue;

      const knowsSuit = target.hintInfo.knownSuits[i] === card.suit;
      const knowsRank = target.hintInfo.knownRanks[i] === card.rank;

      if (knowsSuit && knowsRank) continue;

      if (!knowsRank) {
        return {
          type: 'GIVE_HINT', playerIndex, targetPlayerIndex: targetIdx,
          hint: { kind: 'rank', rank: card.rank },
        };
      }
      if (!knowsSuit) {
        return {
          type: 'GIVE_HINT', playerIndex, targetPlayerIndex: targetIdx,
          hint: { kind: 'suit', suit: card.suit },
        };
      }
    }
  }

  return null;
}

function findAnyUsefulHint(state: GameState, playerIndex: number): Action | null {
  // First: hint about cards that are 1 step away from playable (soon-playable)
  for (const targetIdx of getPlayersAfterInTurnOrder(playerIndex)) {
    const target = state.players[targetIdx];
    for (let i = 0; i < target.hand.length; i++) {
      const card = target.hand[i];
      // Card is "soon playable" if its rank is within 2 of what's needed
      const needed = state.playArea[card.suit] + 1;
      if (card.rank >= needed && card.rank <= needed + 1) {
        if (target.hintInfo.knownRanks[i] !== card.rank) {
          return {
            type: 'GIVE_HINT', playerIndex, targetPlayerIndex: targetIdx,
            hint: { kind: 'rank', rank: card.rank },
          };
        }
        if (target.hintInfo.knownSuits[i] !== card.suit) {
          return {
            type: 'GIVE_HINT', playerIndex, targetPlayerIndex: targetIdx,
            hint: { kind: 'suit', suit: card.suit },
          };
        }
      }
    }
  }

  return null;
}
