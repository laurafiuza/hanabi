import type { GameState, Action, Card, Rank, Suit, HintValue } from './types';
import { SUITS, MAX_INFO_TOKENS, getPlayersAfterInTurnOrder } from './types';
import { getValidActions } from './game';
import { chooseMCTSAction } from './mcts';

const RANK_COPIES: Record<number, number> = { 1: 3, 2: 2, 3: 2, 4: 2, 5: 1 };

// A card is critical if it's the last remaining copy of a card still needed for its suit
function isCritical(card: Card, state: GameState): boolean {
  if (state.playArea[card.suit] >= card.rank) return false;
  const totalCopies = RANK_COPIES[card.rank];
  const discarded = state.discardPile.filter(c => c.suit === card.suit && c.rank === card.rank).length;
  return discarded >= totalCopies - 1;
}

function isRankUseless(knownRank: Rank, state: GameState): boolean {
  return SUITS.every(s => state.playArea[s] >= knownRank);
}

function isCardUseless(suit: Suit, rank: Rank, state: GameState): boolean {
  return state.playArea[suit] >= rank;
}

function isCardPlayable(suit: Suit, rank: Rank, state: GameState): boolean {
  return state.playArea[suit] === rank - 1;
}

function isSuitComplete(suit: Suit, state: GameState): boolean {
  return state.playArea[suit] >= 5;
}

// Find a known-useless card to discard (returns index or -1)
function findUselessDiscard(state: GameState, playerIndex: number): number {
  const me = state.players[playerIndex];
  // Full knowledge: suit+rank and already played
  for (let i = 0; i < me.hand.length; i++) {
    const knownSuit = me.hintInfo.knownSuits[i];
    const knownRank = me.hintInfo.knownRanks[i];
    if (knownSuit && knownRank && isCardUseless(knownSuit, knownRank, state)) return i;
  }
  // Rank-only: all suits have played past this rank
  for (let i = 0; i < me.hand.length; i++) {
    const knownRank = me.hintInfo.knownRanks[i];
    if (knownRank && isRankUseless(knownRank, state)) return i;
  }
  // Suit-only: suit is complete (playArea[suit] === 5)
  for (let i = 0; i < me.hand.length; i++) {
    const knownSuit = me.hintInfo.knownSuits[i];
    if (knownSuit && !me.hintInfo.knownRanks[i] && isSuitComplete(knownSuit, state)) return i;
  }
  return -1;
}

export async function chooseBotAction(state: GameState, playerIndex: number): Promise<Action> {
  try {
    return await chooseMCTSAction(state, playerIndex);
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
      if (knownSuit && knownRank && isCardPlayable(knownSuit, knownRank, state)) {
        if (knownRank < bestRank) { bestRank = knownRank; bestIdx = i; }
      }
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

  // At max info tokens, prefer discarding a useless card before hinting
  // (so we make room for future discard-to-gain-token cycles)
  if (state.infoTokens >= MAX_INFO_TOKENS) {
    const uselessIdx = findUselessDiscard(state, playerIndex);
    if (uselessIdx >= 0) {
      return { type: 'DISCARD', playerIndex, cardIndex: uselessIdx };
    }
  }

  // 3. URGENT: Complete partial info on critical cards, then hint new critical cards
  if (state.infoTokens > 0) {
    const criticalHint = findCriticalHint(state, playerIndex);
    if (criticalHint) return criticalHint;
  }

  // 5. Hint about an immediately playable card in another player's hand
  if (state.infoTokens > 0) {
    const hintAction = findPlayableHint(state, playerIndex);
    if (hintAction) return hintAction;
  }

  // 6. Discard a known-useless card
  {
    const uselessIdx = findUselessDiscard(state, playerIndex);
    if (uselessIdx >= 0) {
      return { type: 'DISCARD', playerIndex, cardIndex: uselessIdx };
    }
  }

  // 7. If info tokens available, hint about something useful before discarding
  if (state.infoTokens > 0) {
    const anyHint = findAnyUsefulHint(state, playerIndex);
    if (anyHint) return anyHint;
  }

  // 8. Discard the oldest unhinted card (highest cardAge, no hint info)
  //    NEVER discard a known 5 — they're irreplaceable
  //    Unhinted cards that have sat around longest are least likely to be important
  {
    let bestIdx = -1;
    let bestAge = -1;
    for (let i = 0; i < me.hand.length; i++) {
      const knownSuit = me.hintInfo.knownSuits[i];
      const knownRank = me.hintInfo.knownRanks[i];
      if (!knownSuit && !knownRank && me.hintInfo.cardAges[i] > bestAge) {
        bestAge = me.hintInfo.cardAges[i];
        bestIdx = i;
      }
    }
    if (bestIdx >= 0) {
      return { type: 'DISCARD', playerIndex, cardIndex: bestIdx };
    }
  }

  // 9. Discard a card we know the least about, but NEVER a known 5
  //    Prefer older cards (higher age = less likely to be important)
  {
    let bestIdx = -1;
    let bestScore = -Infinity;
    for (let i = 0; i < me.hand.length; i++) {
      const knownRank = me.hintInfo.knownRanks[i];
      if (knownRank === 5) continue;
      const knownSuit = me.hintInfo.knownSuits[i];
      // Higher score = better discard candidate
      // Less info known = higher base score, older cards get a bonus
      let score = 0;
      if (!knownRank) score += 3;
      if (!knownSuit) score += 1;
      score += me.hintInfo.cardAges[i] * 0.1; // age tiebreaker
      if (score > bestScore) {
        bestScore = score;
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
    if (hintActions.length > 0) return hintActions[0];
  }

  // 11. Last resort: discard oldest card
  return { type: 'DISCARD', playerIndex, cardIndex: 0 };
}

// ---------------------------------------------------------------------------
// Hint scoring: prefer hints that inform the most cards
// ---------------------------------------------------------------------------

function hintNewInfoCount(target: GameState['players'][number], hint: HintValue): number {
  let count = 0;
  for (let i = 0; i < target.hand.length; i++) {
    const card = target.hand[i];
    if (hint.kind === 'suit') {
      if (card.suit === hint.suit && target.hintInfo.knownSuits[i] !== hint.suit) count++;
    } else {
      if (card.rank === hint.rank && target.hintInfo.knownRanks[i] !== hint.rank) count++;
    }
  }
  return count;
}

function buildHintAction(playerIndex: number, targetIdx: number, hint: HintValue): Action {
  return { type: 'GIVE_HINT', playerIndex, targetPlayerIndex: targetIdx, hint };
}

function bestHintByInfoCount(
  candidates: { targetIdx: number; hint: HintValue; count: number }[],
  playerIndex: number,
): Action | null {
  if (candidates.length === 0) return null;
  candidates.sort((a, b) => b.count - a.count);
  const best = candidates[0];
  return buildHintAction(playerIndex, best.targetIdx, best.hint);
}

// ---------------------------------------------------------------------------
// Hint finders (all collect candidates and pick the one with most new info)
// ---------------------------------------------------------------------------

function findCriticalHint(state: GameState, playerIndex: number): Action | null {
  // Tier 1: complete partial info on a critical card (one hint away from full knowledge)
  const completionCandidates: { targetIdx: number; hint: HintValue; count: number }[] = [];
  // Tier 2: hint about a critical card with no info yet
  const newCandidates: { targetIdx: number; hint: HintValue; count: number }[] = [];

  for (const targetIdx of getPlayersAfterInTurnOrder(playerIndex)) {
    const target = state.players[targetIdx];

    for (let i = 0; i < target.hand.length; i++) {
      const card = target.hand[i];
      if (!isCritical(card, state)) continue;

      const knowsSuit = target.hintInfo.knownSuits[i] === card.suit;
      const knowsRank = target.hintInfo.knownRanks[i] === card.rank;
      if (knowsSuit && knowsRank) continue;

      const hasPartialInfo = knowsSuit || knowsRank;

      if (hasPartialInfo) {
        // Tier 1: give the missing dimension to complete their knowledge
        if (!knowsRank) {
          const hint: HintValue = { kind: 'rank', rank: card.rank };
          completionCandidates.push({ targetIdx, hint, count: hintNewInfoCount(target, hint) });
        }
        if (!knowsSuit) {
          const hint: HintValue = { kind: 'suit', suit: card.suit };
          completionCandidates.push({ targetIdx, hint, count: hintNewInfoCount(target, hint) });
        }
      } else {
        // Tier 2: no info yet — for 5s, rank hint is enough (universally recognized);
        // for non-5s, prefer suit (more disambiguating) but offer both
        if (card.rank === 5) {
          const hint: HintValue = { kind: 'rank', rank: card.rank };
          newCandidates.push({ targetIdx, hint, count: hintNewInfoCount(target, hint) });
        } else {
          const suitHint: HintValue = { kind: 'suit', suit: card.suit };
          // Give suit hint a bonus (+10) so it's preferred over rank for non-5 critical cards
          newCandidates.push({ targetIdx, hint: suitHint, count: hintNewInfoCount(target, suitHint) + 10 });
          const rankHint: HintValue = { kind: 'rank', rank: card.rank };
          newCandidates.push({ targetIdx, hint: rankHint, count: hintNewInfoCount(target, rankHint) });
        }
      }
    }
  }

  // Tier 1 takes priority over tier 2
  return bestHintByInfoCount(completionCandidates, playerIndex)
    ?? bestHintByInfoCount(newCandidates, playerIndex);
}

function findPlayableHint(state: GameState, playerIndex: number): Action | null {
  const candidates: { targetIdx: number; hint: HintValue; count: number }[] = [];

  for (const targetIdx of getPlayersAfterInTurnOrder(playerIndex)) {
    const target = state.players[targetIdx];

    for (let i = 0; i < target.hand.length; i++) {
      const card = target.hand[i];
      if (state.playArea[card.suit] !== card.rank - 1) continue;

      const knowsSuit = target.hintInfo.knownSuits[i] === card.suit;
      const knowsRank = target.hintInfo.knownRanks[i] === card.rank;
      if (knowsSuit && knowsRank) continue;

      if (!knowsRank) {
        const hint: HintValue = { kind: 'rank', rank: card.rank };
        candidates.push({ targetIdx, hint, count: hintNewInfoCount(target, hint) });
      }
      if (!knowsSuit) {
        const hint: HintValue = { kind: 'suit', suit: card.suit };
        candidates.push({ targetIdx, hint, count: hintNewInfoCount(target, hint) });
      }
    }
  }

  return bestHintByInfoCount(candidates, playerIndex);
}

function findAnyUsefulHint(state: GameState, playerIndex: number): Action | null {
  const candidates: { targetIdx: number; hint: HintValue; count: number }[] = [];

  for (const targetIdx of getPlayersAfterInTurnOrder(playerIndex)) {
    const target = state.players[targetIdx];
    for (let i = 0; i < target.hand.length; i++) {
      const card = target.hand[i];
      const needed = state.playArea[card.suit] + 1;
      if (card.rank >= needed && card.rank <= needed + 1) {
        if (target.hintInfo.knownRanks[i] !== card.rank) {
          const hint: HintValue = { kind: 'rank', rank: card.rank };
          candidates.push({ targetIdx, hint, count: hintNewInfoCount(target, hint) });
        }
        if (target.hintInfo.knownSuits[i] !== card.suit) {
          const hint: HintValue = { kind: 'suit', suit: card.suit };
          candidates.push({ targetIdx, hint, count: hintNewInfoCount(target, hint) });
        }
      }
    }
  }

  return bestHintByInfoCount(candidates, playerIndex);
}
