import styles from './GameSetup.module.css';

interface GameSetupProps {
  onStart: () => void;
}

export function GameSetup({ onStart }: GameSetupProps) {
  return (
    <div className={styles.setup}>
      <h1 className={styles.title}>Hanabi</h1>
      <p className={styles.subtitle}>A cooperative card game</p>
      <p className={styles.info}>5 players (you + 4 bots) &bull; 1 fuse token</p>
      <button className={styles.btn} onClick={onStart}>New Game</button>
    </div>
  );
}
