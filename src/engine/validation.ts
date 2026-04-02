import type { GameState, Action } from './types';

export function isValidAction(state: GameState, action: Action): boolean {
  if (state.status !== 'playing') return false;
  if (action.playerIndex !== state.currentPlayerIndex) return false;

  const player = state.players[action.playerIndex];

  switch (action.type) {
    case 'PLAY_CARD':
      return action.cardIndex >= 0 && action.cardIndex < player.hand.length;

    case 'DISCARD':
      return action.cardIndex >= 0 && action.cardIndex < player.hand.length;

    case 'GIVE_HINT': {
      if (state.infoTokens <= 0) return false;
      if (action.targetPlayerIndex === action.playerIndex) return false;
      const target = state.players[action.targetPlayerIndex];
      if (!target || target.hand.length === 0) return false;
      // Hint must match at least one card
      const matches = target.hand.some(card =>
        action.hint.kind === 'suit'
          ? card.suit === action.hint.suit
          : card.rank === action.hint.rank
      );
      if (!matches) return false;
      // Hint must convey new information to at least one matching card
      const conveysNew = target.hand.some((card, i) => {
        if (action.hint.kind === 'suit') {
          return card.suit === action.hint.suit && target.hintInfo.knownSuits[i] !== action.hint.suit;
        }
        return card.rank === action.hint.rank && target.hintInfo.knownRanks[i] !== action.hint.rank;
      });
      return conveysNew;
    }

    default:
      return false;
  }
}
