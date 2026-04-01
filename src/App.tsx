import { useEffect } from 'react';
import { useGame } from './hooks/useGame';
import { GameSetup } from './components/GameSetup/GameSetup';
import { GameBoard } from './components/GameBoard/GameBoard';
import { loadWeights } from './engine/rl_model';

function App() {
  const { state, doAction, startGame } = useGame();

  // Load RL model weights on mount (non-blocking, falls back to heuristic)
  useEffect(() => {
    loadWeights().catch(() => {
      console.log('RL model weights not found, using heuristic bot');
    });
  }, []);

  if (state.status === 'setup') {
    return <GameSetup onStart={startGame} />;
  }

  return <GameBoard state={state} doAction={doAction} startGame={startGame} />;
}

export default App;
