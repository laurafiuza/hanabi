import type { Suit } from '../../engine/types';
import { SUITS, SUIT_COLORS } from '../../engine/types';
import styles from './PlayArea.module.css';

interface PlayAreaProps {
  playArea: Record<Suit, number>;
}

function suitSymbol(suit: Suit): string {
  const symbols: Record<Suit, string> = {
    white: '★', yellow: '●', green: '♦', blue: '♠', red: '♥',
  };
  return symbols[suit];
}

export function PlayArea({ playArea }: PlayAreaProps) {
  return (
    <div className={styles.area}>
      {SUITS.map(suit => (
        <div
          key={suit}
          className={styles.stack}
          style={{ borderColor: SUIT_COLORS[suit] }}
        >
          <div className={styles.symbol} style={{ color: SUIT_COLORS[suit] }}>
            {suitSymbol(suit)}
          </div>
          <div className={styles.value} style={{ color: SUIT_COLORS[suit] }}>
            {playArea[suit] || '—'}
          </div>
        </div>
      ))}
    </div>
  );
}
