import type { Player } from '../../engine/types';
import { getChopIndex } from '../../engine/ai';
import { Card } from '../Card/Card';
import styles from './PlayerHand.module.css';

interface PlayerHandProps {
  player: Player;
  isCurrentPlayer: boolean;
  selectedCardIndex: number | null;
  onCardClick?: (index: number) => void;
  highlightedCards?: Set<number>;
  onViewKnowledge?: () => void;
}

export function PlayerHand({
  player,
  isCurrentPlayer,
  selectedCardIndex,
  onCardClick,
  highlightedCards,
  onViewKnowledge,
}: PlayerHandProps) {
  const faceUp = !player.isHuman;
  const chopIdx = getChopIndex(player);

  return (
    <div className={`${styles.hand} ${isCurrentPlayer ? styles.active : ''}`}>
      <div className={styles.nameRow}>
        <div className={styles.name}>
          {player.name}
          {isCurrentPlayer && <span className={styles.turnIndicator}> ◄</span>}
        </div>
        {!player.isHuman && onViewKnowledge && (
          <button className={styles.knowledgeBtn} onClick={onViewKnowledge} title="View knowledge">
            🧠
          </button>
        )}
      </div>
      <div className={styles.cards}>
        {player.hand.map((card, i) => (
          <Card
            key={card.id}
            card={card}
            faceUp={faceUp}
            knownSuit={player.hintInfo.knownSuits[i]}
            knownRank={player.hintInfo.knownRanks[i]}
            cardAge={player.hintInfo.cardAges[i]}
            isChop={i === chopIdx}
            selected={selectedCardIndex === i}
            highlighted={highlightedCards?.has(i)}
            onClick={() => onCardClick?.(i)}
          />
        ))}
      </div>
    </div>
  );
}
