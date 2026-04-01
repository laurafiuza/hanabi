import { useReducer, useEffect, useCallback } from 'react';
import type { GameState, Action } from '../engine/types';
import { createGame, applyAction } from '../engine/game';
import { chooseBotAction } from '../engine/ai';

function gameReducer(state: GameState, action: Action | { type: 'NEW_GAME' }): GameState {
  if (action.type === 'NEW_GAME') {
    return createGame();
  }
  return applyAction(state, action as Action);
}

const INITIAL_STATE: GameState = {
  players: [],
  deck: [],
  playArea: { white: 0, yellow: 0, green: 0, blue: 0, red: 0 },
  discardPile: [],
  infoTokens: 8,
  fuseTokens: 1,
  currentPlayerIndex: 0,
  turnsRemaining: null,
  status: 'setup',
  lastActionResult: null,
  history: [],
  turnNumber: 1,
  gameOverReason: null,
};

export function useGame() {
  const [state, dispatch] = useReducer(gameReducer, INITIAL_STATE);

  const startGame = useCallback(() => {
    dispatch({ type: 'NEW_GAME' });
  }, []);

  const doAction = useCallback((action: Action) => {
    dispatch(action);
  }, []);

  // Bot turn scheduling
  useEffect(() => {
    if (state.status !== 'playing') return;
    const currentPlayer = state.players[state.currentPlayerIndex];
    if (!currentPlayer || currentPlayer.isHuman) return;

    const timeout = setTimeout(() => {
      const action = chooseBotAction(state, state.currentPlayerIndex);
      if (action) {
        dispatch(action);
      }
    }, 3000);

    return () => clearTimeout(timeout);
  }, [state.currentPlayerIndex, state.status]);

  return { state, doAction, startGame };
}
