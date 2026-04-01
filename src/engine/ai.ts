import type { GameState, Action, Rank } from './types';
import { SUITS } from './types';
import { getValidActions } from './game';

export function chooseBotAction(state: GameState, playerIndex: number): Action {
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
  //    (e.g., we know it's a 1 and at least one suit needs a 1 — since we don't know which suit,
  //     only play if ALL suits that could need this rank do need it, i.e., every suit is at rank-1)
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

  // 3. Hint about an immediately playable card in another player's hand
  //    Prefer hinting the next player in turn order so they can act on it soonest
  if (state.infoTokens > 0) {
    const hintAction = findPlayableHint(state, playerIndex);
    if (hintAction) return hintAction;
  }

  // 4. Discard a card we know is useless (already played past its rank in that suit)
  for (let i = 0; i < me.hand.length; i++) {
    const knownSuit = me.hintInfo.knownSuits[i];
    const knownRank = me.hintInfo.knownRanks[i];
    if (knownSuit && knownRank) {
      if (state.playArea[knownSuit] >= knownRank) {
        return { type: 'DISCARD', playerIndex, cardIndex: i };
      }
    }
  }

  // 5. If info tokens are low, give any useful hint instead of discarding
  if (state.infoTokens > 0 && state.infoTokens <= 3) {
    const anyHint = findAnyUsefulHint(state, playerIndex);
    if (anyHint) return anyHint;
  }

  // 6. Discard the oldest card with no hint info (least likely to be important)
  for (let i = 0; i < me.hand.length; i++) {
    const knownSuit = me.hintInfo.knownSuits[i];
    const knownRank = me.hintInfo.knownRanks[i];
    if (!knownSuit && !knownRank) {
      return { type: 'DISCARD', playerIndex, cardIndex: i };
    }
  }

  // 7. Give any valid hint if we have tokens
  if (state.infoTokens > 0) {
    const hintActions = actions.filter(a => a.type === 'GIVE_HINT');
    if (hintActions.length > 0) {
      return hintActions[0];
    }
  }

  // 8. Last resort: discard oldest card
  return { type: 'DISCARD', playerIndex, cardIndex: 0 };
}

function findPlayableHint(state: GameState, playerIndex: number): Action | null {
  const numPlayers = state.players.length;

  // Check players in turn order starting from next player
  for (let offset = 1; offset < numPlayers; offset++) {
    const targetIdx = (playerIndex + offset) % numPlayers;
    const target = state.players[targetIdx];

    for (let i = 0; i < target.hand.length; i++) {
      const card = target.hand[i];
      if (state.playArea[card.suit] !== card.rank - 1) continue;

      // This card is playable — hint about it
      // Prefer rank hint if they don't know the rank, suit hint if they don't know the suit
      // Pick whichever gives more new info, or whichever is more specific (fewer cards matched)
      const knowsSuit = target.hintInfo.knownSuits[i] === card.suit;
      const knowsRank = target.hintInfo.knownRanks[i] === card.rank;

      if (knowsSuit && knowsRank) continue; // They already know everything

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
  // Hint about 5s (they're unique and critical) or about suits with low progress
  const numPlayers = state.players.length;

  for (let offset = 1; offset < numPlayers; offset++) {
    const targetIdx = (playerIndex + offset) % numPlayers;
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
