import { MAX_INFO_TOKENS, INITIAL_FUSE_TOKENS } from '../../engine/types';
import styles from './TokenDisplay.module.css';

interface TokenDisplayProps {
  infoTokens: number;
  fuseTokens: number;
  deckSize: number;
  score: number;
}

export function TokenDisplay({ infoTokens, fuseTokens, deckSize, score }: TokenDisplayProps) {
  return (
    <div className={styles.bar}>
      <div className={styles.group}>
        <span className={styles.label}>Hints</span>
        <div className={styles.tokens}>
          {Array.from({ length: MAX_INFO_TOKENS }, (_, i) => (
            <span
              key={i}
              className={`${styles.dot} ${i < infoTokens ? styles.active : styles.spent}`}
            />
          ))}
        </div>
      </div>
      <div className={styles.group}>
        <span className={styles.label}>Fuse</span>
        <div className={styles.tokens}>
          {Array.from({ length: INITIAL_FUSE_TOKENS }, (_, i) => (
            <span
              key={i}
              className={`${styles.fuse} ${i < fuseTokens ? styles.fuseActive : styles.fuseSpent}`}
            >
              💣
            </span>
          ))}
        </div>
      </div>
      <div className={styles.group}>
        <span className={styles.label}>Deck</span>
        <span className={styles.number}>{deckSize}</span>
      </div>
      <div className={styles.group}>
        <span className={styles.label}>Score</span>
        <span className={styles.number}>{score}/25</span>
      </div>
    </div>
  );
}
