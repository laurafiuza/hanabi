import { useState, useEffect, useRef } from 'react';
import type { GameState, Action, HintValue } from '../../engine/types';
import { getScore } from '../../engine/game';
import { PlayerHand } from '../PlayerHand/PlayerHand';
import { PlayArea } from '../PlayArea/PlayArea';
import { TokenDisplay } from '../TokenDisplay/TokenDisplay';
import { HintSelector } from '../HintSelector/HintSelector';
import { DiscardPile } from '../DiscardPile/DiscardPile';
import { GameOverScreen } from '../GameOverScreen/GameOverScreen';
import { History } from '../History/History';
import { KnowledgeView } from '../KnowledgeView/KnowledgeView';
import styles from './GameBoard.module.css';

interface GameBoardProps {
  state: GameState;
  doAction: (action: Action) => void;
  startGame: () => void;
}

export function GameBoard({ state, doAction, startGame }: GameBoardProps) {
  const [selectedCard, setSelectedCard] = useState<number | null>(null);
  const [showHintSelector, setShowHintSelector] = useState(false);
  const [viewingKnowledgeOf, setViewingKnowledgeOf] = useState<number | null>(null);

  const humanPlayer = state.players[0];
  // Display bots left-to-right: Bottom Left, Top Left, Top Right, Bottom Right
  const botPlayers = [2, 3, 1, 4].map(i => state.players[i]);
  const isHumanTurn = state.currentPlayerIndex === 0 && state.status === 'playing';
  const score = getScore(state.playArea);

  const handleCardClick = (index: number) => {
    if (!isHumanTurn) return;
    setSelectedCard(selectedCard === index ? null : index);
  };

  const handlePlay = () => {
    if (selectedCard === null) return;
    doAction({ type: 'PLAY_CARD', playerIndex: 0, cardIndex: selectedCard });
    setSelectedCard(null);
  };

  const handleDiscard = () => {
    if (selectedCard === null) return;
    doAction({ type: 'DISCARD', playerIndex: 0, cardIndex: selectedCard });
    setSelectedCard(null);
  };

  const handleGiveHint = (targetPlayerIndex: number, hint: HintValue) => {
    doAction({ type: 'GIVE_HINT', playerIndex: 0, targetPlayerIndex, hint });
    setShowHintSelector(false);
  };

  const gameOver = state.status === 'won' || state.status === 'lost' || state.status === 'finished';

  // Format last action description
  const lastActionText = formatLastAction(state);

  return (
    <div className={styles.layout}>
      <div className={styles.board}>
        <TokenDisplay
          infoTokens={state.infoTokens}
          fuseTokens={state.fuseTokens}
          deckSize={state.deck.length}
          score={score}
        />

        {lastActionText && <div className={styles.lastAction}>{lastActionText}</div>}

        <PlayArea playArea={state.playArea} />

        <div className={styles.botHands}>
          {botPlayers.map(player => (
            <PlayerHand
              key={player.id}
              player={player}
              isCurrentPlayer={state.currentPlayerIndex === player.id}
              selectedCardIndex={null}
              onViewKnowledge={() => setViewingKnowledgeOf(player.id)}
            />
          ))}
        </div>

        <div className={styles.humanSection}>
          <PlayerHand
            player={humanPlayer}
            isCurrentPlayer={isHumanTurn}
            selectedCardIndex={selectedCard}
            onCardClick={handleCardClick}
          />

          {isHumanTurn && (
            <div className={styles.actions}>
              <button
                className={styles.actionBtn}
                disabled={selectedCard === null}
                onClick={handlePlay}
              >
                Play
              </button>
              <button
                className={styles.actionBtn}
                disabled={selectedCard === null}
                onClick={handleDiscard}
              >
                Discard
              </button>
              <button
                className={styles.actionBtn}
                disabled={state.infoTokens <= 0}
                onClick={() => setShowHintSelector(true)}
              >
                Hint ({state.infoTokens})
              </button>
            </div>
          )}

          {!isHumanTurn && state.status === 'playing' && (
            <ThinkingIndicator playerName={state.players[state.currentPlayerIndex]?.name} />
          )}
        </div>

        <DiscardPile cards={state.discardPile} />
      </div>

      <div className={styles.sidebar}>
        <History entries={state.history} />
      </div>

      {showHintSelector && (
        <HintSelector
          players={state.players}
          currentPlayerIndex={0}
          onGiveHint={handleGiveHint}
          onCancel={() => setShowHintSelector(false)}
        />
      )}

      {viewingKnowledgeOf !== null && (
        <KnowledgeView
          player={state.players[viewingKnowledgeOf]}
          onClose={() => setViewingKnowledgeOf(null)}
        />
      )}

      {gameOver && (
        <GameOverScreen
          status={state.status}
          score={score}
          reason={state.gameOverReason}
          onNewGame={startGame}
        />
      )}
    </div>
  );
}

function ThinkingIndicator({ playerName }: { playerName: string }) {
  const [elapsed, setElapsed] = useState(0);
  const startRef = useRef(Date.now());

  useEffect(() => {
    startRef.current = Date.now();
    setElapsed(0);
    const interval = setInterval(() => {
      setElapsed(Math.floor((Date.now() - startRef.current) / 1000));
    }, 1000);
    return () => clearInterval(interval);
  }, [playerName]);

  return (
    <div className={styles.waiting}>
      {playerName} thinking{elapsed > 0 ? ` (${elapsed}s)` : ''}...
    </div>
  );
}

function formatLastAction(state: GameState): string | null {
  const result = state.lastActionResult;
  if (!result) return null;

  const playerName = state.players[result.action.playerIndex]?.name ?? 'Unknown';

  switch (result.action.type) {
    case 'PLAY_CARD':
      if (result.success) {
        return `${playerName} played ${result.cardPlayed?.suit} ${result.cardPlayed?.rank} successfully!`;
      }
      return `${playerName} played ${result.cardPlayed?.suit} ${result.cardPlayed?.rank} — BOOM!`;
    case 'DISCARD':
      return `${playerName} discarded ${result.cardPlayed?.suit} ${result.cardPlayed?.rank}`;
    case 'GIVE_HINT': {
      const target = state.players[result.action.targetPlayerIndex]?.name;
      const hint = result.action.hint;
      const desc = hint.kind === 'suit' ? hint.suit : String(hint.rank);
      return `${playerName} told ${target} about their ${desc}s`;
    }
    default:
      return null;
  }
}
