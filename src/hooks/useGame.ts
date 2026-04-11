import { useReducer, useEffect, useCallback } from 'react';
import type { GameState, Action } from '../engine/types';
import { createGame, applyAction } from '../engine/game';
import { chooseBotAction, analyzeHumanAction } from '../engine/ai';

function gameReducer(state: GameState, action: Action | { type: 'NEW_GAME' }): GameState {
  if (action.type === 'NEW_GAME') {
    return createGame();
  }
  const act = action as Action;
  const isHuman = state.players[act.playerIndex]?.isHuman;

  // Analyze the human's action BEFORE applying it (need pre-action state)
  const tip = isHuman ? analyzeHumanAction(state, act) : null;

  const newState = applyAction(state, act);

  if (tip) {
    return {
      ...newState,
      history: [...newState.history, { playerName: 'Tip', text: tip, turn: state.turnNumber }],
    };
  }
  return newState;
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

    let cancelled = false;
    const minDelay = new Promise(resolve => setTimeout(resolve, 2000));

    // Start MCTS immediately — it yields to the event loop internally
    // But always wait at least 2s so the human can follow along
    Promise.all([chooseBotAction(state, state.currentPlayerIndex), minDelay]).then(([action]) => {
      if (!cancelled && action) {
        dispatch(action);
      }
    });

    return () => { cancelled = true; };
  }, [state.currentPlayerIndex, state.status]);

  return { state, doAction, startGame };
}
