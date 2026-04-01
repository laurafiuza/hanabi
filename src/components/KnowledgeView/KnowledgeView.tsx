import type { Player, Suit } from '../../engine/types';
import { SUIT_COLORS } from '../../engine/types';
import styles from './KnowledgeView.module.css';

interface KnowledgeViewProps {
  player: Player;
  onClose: () => void;
}

function suitSymbol(suit: Suit): string {
  const symbols: Record<Suit, string> = {
    white: '★', yellow: '●', green: '♦', blue: '♠', red: '♥',
  };
  return symbols[suit];
}

export function KnowledgeView({ player, onClose }: KnowledgeViewProps) {
  return (
    <div className={styles.overlay} onClick={onClose}>
      <div className={styles.modal} onClick={e => e.stopPropagation()}>
        <h3>{player.name}'s Knowledge</h3>
        <p className={styles.subtitle}>What {player.name} knows about their own hand:</p>
        <div className={styles.cards}>
          {player.hand.map((_, i) => {
            const knownSuit = player.hintInfo.knownSuits[i];
            const knownRank = player.hintInfo.knownRanks[i];
            const hasInfo = knownSuit || knownRank;
            return (
              <div key={i} className={styles.card}>
                <div className={styles.cardLabel}>Card {i + 1}</div>
                {hasInfo ? (
                  <div className={styles.info}>
                    {knownSuit && (
                      <span style={{ color: SUIT_COLORS[knownSuit] }}>
                        {suitSymbol(knownSuit)} {knownSuit}
                      </span>
                    )}
                    {knownRank && <span className={styles.rank}>{knownRank}</span>}
                  </div>
                ) : (
                  <div className={styles.unknown}>No info</div>
                )}
              </div>
            );
          })}
        </div>
        <button className={styles.closeBtn} onClick={onClose}>Close</button>
      </div>
    </div>
  );
}
