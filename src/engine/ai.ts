import type { GameState, Action, Card, Rank } from './types';
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

// Check if a player has any unhinted critical cards they might discard
function findCriticalHint(state: GameState, playerIndex: number): Action | null {
  // Prioritize the next player in turn order — they'll act soonest
  for (const targetIdx of getPlayersAfterInTurnOrder(playerIndex)) {
    const target = state.players[targetIdx];

    for (let i = 0; i < target.hand.length; i++) {
      const card = target.hand[i];
      if (!isCritical(card, state)) continue;

      const knowsSuit = target.hintInfo.knownSuits[i] === card.suit;
      const knowsRank = target.hintInfo.knownRanks[i] === card.rank;

      // If they already know both, they hopefully won't discard it
      if (knowsSuit && knowsRank) continue;

      // If they know neither, they might discard it — hint something
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

  // 1. Play a card where we know both suit and rank, and it's the next needed
  for (let i = 0; i < me.hand.length; i++) {
    const knownSuit = me.hintInfo.knownSuits[i];
    const knownRank = me.hintInfo.knownRanks[i];
    if (knownSuit && knownRank) {
      if (state.playArea[knownSuit] === knownRank - 1) {
        return { type: 'PLAY_CARD', playerIndex, cardIndex: i };
      }
    }
  }

  // 2. Play a card where we know the rank and it's playable in every suit that still needs it
  for (let i = 0; i < me.hand.length; i++) {
    const knownRank = me.hintInfo.knownRanks[i];
    const knownSuit = me.hintInfo.knownSuits[i];
    if (knownRank && !knownSuit) {
      const allSuitsReady = SUITS.every(s => state.playArea[s] >= knownRank || state.playArea[s] === knownRank - 1);
      const someNeedIt = SUITS.some(s => state.playArea[s] === knownRank - 1);
      if (allSuitsReady && someNeedIt) {
        return { type: 'PLAY_CARD', playerIndex, cardIndex: i };
      }
    }
  }

  // 3. URGENT: Hint about critical cards another player might discard
  //    This is higher priority than hinting about playable cards
  if (state.infoTokens > 0) {
    const criticalHint = findCriticalHint(state, playerIndex);
    if (criticalHint) return criticalHint;
  }

  // 4. Hint about an immediately playable card in another player's hand
  if (state.infoTokens > 0) {
    const hintAction = findPlayableHint(state, playerIndex);
    if (hintAction) return hintAction;
  }

  // 5. Discard a card we know is useless (already played past its rank in that suit)
  for (let i = 0; i < me.hand.length; i++) {
    const knownSuit = me.hintInfo.knownSuits[i];
    const knownRank = me.hintInfo.knownRanks[i];
    if (knownSuit && knownRank) {
      if (state.playArea[knownSuit] >= knownRank) {
        return { type: 'DISCARD', playerIndex, cardIndex: i };
      }
    }
  }

  // 6. If info tokens are low, give any useful hint instead of discarding
  if (state.infoTokens > 0 && state.infoTokens <= 3) {
    const anyHint = findAnyUsefulHint(state, playerIndex);
    if (anyHint) return anyHint;
  }

  // 7. Discard the oldest card with no hint info (least likely to be important)
  //    But never discard a card we know is critical
  for (let i = 0; i < me.hand.length; i++) {
    const knownSuit = me.hintInfo.knownSuits[i];
    const knownRank = me.hintInfo.knownRanks[i];
    if (!knownSuit && !knownRank) {
      return { type: 'DISCARD', playerIndex, cardIndex: i };
    }
  }

  // 8. Give any valid hint if we have tokens
  if (state.infoTokens > 0) {
    const hintActions = actions.filter(a => a.type === 'GIVE_HINT');
    if (hintActions.length > 0) {
      return hintActions[0];
    }
  }

  // 9. Last resort: discard oldest card
  return { type: 'DISCARD', playerIndex, cardIndex: 0 };
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
  for (const targetIdx of getPlayersAfterInTurnOrder(playerIndex)) {
    const target = state.players[targetIdx];

    // Hint about 5s the target doesn't know about
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
