import { useState } from 'react';
import type { Player, Suit, HintValue } from '../../engine/types';
import { SUIT_COLORS } from '../../engine/types';
import styles from './HintSelector.module.css';

interface HintSelectorProps {
  players: Player[];
  currentPlayerIndex: number;
  onGiveHint: (targetPlayerIndex: number, hint: HintValue) => void;
  onCancel: () => void;
}

function suitSymbol(suit: Suit): string {
  const symbols: Record<Suit, string> = {
    white: '★', yellow: '●', green: '♦', blue: '♠', red: '♥',
  };
  return symbols[suit];
}

export function HintSelector({ players, currentPlayerIndex, onGiveHint, onCancel }: HintSelectorProps) {
  const [targetIndex, setTargetIndex] = useState<number | null>(null);

  const otherPlayers = players.filter((_, i) => i !== currentPlayerIndex);

  if (targetIndex === null) {
    return (
      <div className={styles.overlay}>
        <div className={styles.modal}>
          <h3>Give a hint to...</h3>
          <div className={styles.options}>
            {otherPlayers.map(p => (
              <button key={p.id} className={styles.btn} onClick={() => setTargetIndex(p.id)}>
                {p.name}
              </button>
            ))}
          </div>
          <button className={styles.cancelBtn} onClick={onCancel}>Cancel</button>
        </div>
      </div>
    );
  }

  const target = players[targetIndex];
  const suitsInHand = [...new Set(target.hand.map(c => c.suit))];
  const ranksInHand = [...new Set(target.hand.map(c => c.rank))].sort((a, b) => a - b);

  return (
    <div className={styles.overlay}>
      <div className={styles.modal}>
        <h3>Hint for {target.name}</h3>
        <div className={styles.section}>
          <span className={styles.sectionLabel}>Color</span>
          <div className={styles.options}>
            {suitsInHand.map(suit => (
              <button
                key={suit}
                className={styles.btn}
                style={{ color: SUIT_COLORS[suit] }}
                onClick={() => onGiveHint(targetIndex, { kind: 'suit', suit })}
              >
                {suitSymbol(suit)} {suit}
              </button>
            ))}
          </div>
        </div>
        <div className={styles.section}>
          <span className={styles.sectionLabel}>Number</span>
          <div className={styles.options}>
            {ranksInHand.map(rank => (
              <button
                key={rank}
                className={styles.btn}
                onClick={() => onGiveHint(targetIndex, { kind: 'rank', rank })}
              >
                {rank}
              </button>
            ))}
          </div>
        </div>
        <button className={styles.cancelBtn} onClick={() => setTargetIndex(null)}>Back</button>
        <button className={styles.cancelBtn} onClick={onCancel}>Cancel</button>
      </div>
    </div>
  );
}
