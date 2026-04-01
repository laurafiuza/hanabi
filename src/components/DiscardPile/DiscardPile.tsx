import { useState } from 'react';
import type { Card as CardType } from '../../engine/types';
import { SUIT_COLORS } from '../../engine/types';
import styles from './DiscardPile.module.css';

interface DiscardPileProps {
  cards: CardType[];
}

function suitSymbol(suit: string): string {
  const symbols: Record<string, string> = {
    white: '★', yellow: '●', green: '♦', blue: '♠', red: '♥',
  };
  return symbols[suit] || suit;
}

export function DiscardPile({ cards }: DiscardPileProps) {
  const [expanded, setExpanded] = useState(false);

  return (
    <div className={styles.container}>
      <button className={styles.toggle} onClick={() => setExpanded(!expanded)}>
        Discard ({cards.length}) {expanded ? '▼' : '▶'}
      </button>
      {expanded && (
        <div className={styles.list}>
          {cards.length === 0 && <span className={styles.empty}>Empty</span>}
          {cards.map((card, i) => (
            <span
              key={`${card.id}-${i}`}
              className={styles.chip}
              style={{ color: SUIT_COLORS[card.suit] }}
            >
              {suitSymbol(card.suit)}{card.rank}
            </span>
          ))}
        </div>
      )}
    </div>
  );
}
