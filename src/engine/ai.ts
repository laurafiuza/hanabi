import type { GameState, Action } from './types';
import { getValidActions } from './game';

export function chooseBotAction(state: GameState, playerIndex: number): Action {
  const actions = getValidActions(state, playerIndex);
  // Random valid action for v1
  return actions[Math.floor(Math.random() * actions.length)];
}
