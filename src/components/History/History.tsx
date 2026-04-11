import { useEffect, useRef } from 'react';
import type { HistoryEntry } from '../../engine/types';
import styles from './History.module.css';

interface HistoryProps {
  entries: HistoryEntry[];
}

export function History({ entries }: HistoryProps) {
  const listRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (listRef.current) {
      listRef.current.scrollTop = listRef.current.scrollHeight;
    }
  }, [entries.length]);

  return (
    <div className={styles.container}>
      <div className={styles.header}>History</div>
      <div className={styles.list} ref={listRef}>
        {entries.length === 0 && <div className={styles.empty}>No actions yet</div>}
        {entries.map((entry, i) => {
          const isTip = entry.playerName === 'Tip';
          return (
            <div key={i} className={`${styles.entry} ${isTip ? styles.tip : ''}`}>
              {isTip ? (
                <span className={styles.tipText}>{entry.text}</span>
              ) : (
                <>
                  <span className={styles.turn}>#{entry.turn}</span>
                  <span className={styles.player}>{entry.playerName}</span>
                  <span className={styles.text}>{entry.text}</span>
                </>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}
