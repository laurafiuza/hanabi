import type { Card as CardType, Suit, Rank } from '../../engine/types';
import { SUIT_COLORS } from '../../engine/types';
import styles from './Card.module.css';

interface CardProps {
  card?: CardType;
  faceUp: boolean;
  knownSuit?: Suit | null;
  knownRank?: Rank | null;
  cardAge?: number;
  isChop?: boolean;
  selected?: boolean;
  highlighted?: boolean;
  onClick?: () => void;
}

const CHOP_TOOLTIP = 'The chop is the oldest unclued card — the one that would be discarded next. Protect it if it holds a critical card!';

export function Card({ card, faceUp, knownSuit, knownRank, cardAge, isChop, selected, highlighted, onClick }: CardProps) {
  const ageTooltip = cardAge != null ? `Age: ${cardAge} turn${cardAge === 1 ? '' : 's'} since drawn or last clued.` : '';

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
          <div className={`${styles.age} ${styles.tooltip}`} data-tooltip={ageTooltip}>
            {cardAge}
          </div>
        )}
        {isChop && (
          <div className={`${styles.chopLabel} ${styles.tooltip}`} data-tooltip={CHOP_TOOLTIP}>
            CHOP
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
        <div className={`${styles.age} ${styles.tooltip}`} data-tooltip={ageTooltip}>
          {cardAge}
        </div>
      )}
      {isChop && (
        <div className={`${styles.chopLabel} ${styles.tooltip}`} data-tooltip={CHOP_TOOLTIP}>
          CHOP
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
