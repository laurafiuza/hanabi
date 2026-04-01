import { useGame } from './hooks/useGame';
import { GameSetup } from './components/GameSetup/GameSetup';
import { GameBoard } from './components/GameBoard/GameBoard';

function App() {
  const { state, doAction, startGame } = useGame();

  if (state.status === 'setup') {
    return <GameSetup onStart={startGame} />;
  }

  return <GameBoard state={state} doAction={doAction} startGame={startGame} />;
}

export default App;
