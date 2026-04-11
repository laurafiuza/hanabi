import type { Card as CardType, Suit, Rank } from '../../engine/types';
import { SUIT_COLORS } from '../../engine/types';
import styles from './Card.module.css';

interface CardProps {
  card?: CardType;
  faceUp: boolean;
  knownSuit?: Suit | null;
  knownRank?: Rank | null;
  cardAge?: number;
  selected?: boolean;
  highlighted?: boolean;
  onClick?: () => void;
}

export function Card({ card, faceUp, knownSuit, knownRank, cardAge, selected, highlighted, onClick }: CardProps) {
  if (faceUp && card) {
    return (
      <div
        className={`${styles.card} ${styles.faceUp} ${selected ? styles.selected : ''} ${highlighted ? styles.highlighted : ''}`}
        style={{ borderColor: SUIT_COLORS[card.suit], color: SUIT_COLORS[card.suit] }}
        onClick={onClick}
      >
        <div className={styles.rank}>{card.rank}</div>
        <div className={styles.suit}>{suitSymbol(card.suit)}</div>
        {cardAge != null && (
          <div
            className={styles.age}
            title={`Card age: ${cardAge} turn${cardAge === 1 ? '' : 's'} since drawn or last clued. The oldest unclued card is your "chop" — the one you'd discard next.`}
          >
            {cardAge}
          </div>
        )}
      </div>
    );
  }

  return (
    <div
      className={`${styles.card} ${styles.faceDown} ${selected ? styles.selected : ''}`}
      onClick={onClick}
    >
      <div className={styles.hintInfo}>
        {knownSuit && (
          <span className={styles.hintBadge} style={{ color: SUIT_COLORS[knownSuit] }}>
            {suitSymbol(knownSuit)}
          </span>
        )}
        {knownRank && <span className={styles.hintBadge}>{knownRank}</span>}
        {!knownSuit && !knownRank && <span className={styles.unknown}>?</span>}
      </div>
      {cardAge != null && (
          <div
            className={styles.age}
            title={`Card age: ${cardAge} turn${cardAge === 1 ? '' : 's'} since drawn or last clued. The oldest unclued card is your "chop" — the one you'd discard next.`}
          >
            {cardAge}
          </div>
        )}
    </div>
  );
}

function suitSymbol(suit: Suit): string {
  const symbols: Record<Suit, string> = {
    white: '★',
    yellow: '●',
    green: '♦',
    blue: '♠',
    red: '♥',
  };
  return symbols[suit];
}
