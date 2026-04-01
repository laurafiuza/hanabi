import styles from './GameOverScreen.module.css';

interface GameOverScreenProps {
  status: string;
  score: number;
  reason: string | null;
  onNewGame: () => void;
}

export function GameOverScreen({ status, score, reason, onNewGame }: GameOverScreenProps) {
  let title = 'Game Over';
  let subtitle = '';

  if (status === 'won') {
    title = 'Perfect Score!';
    subtitle = 'All fireworks completed!';
  } else if (status === 'lost') {
    title = 'Boom!';
    subtitle = reason ?? 'The fuse ran out...';
  } else {
    title = 'Game Over';
    subtitle = 'The deck ran out.';
  }

  return (
    <div className={styles.overlay}>
      <div className={styles.modal}>
        <h2 className={styles.title}>{title}</h2>
        <p className={styles.subtitle}>{subtitle}</p>
        <div className={styles.score}>
          <span className={styles.scoreLabel}>Final Score</span>
          <span className={styles.scoreValue}>{score}/25</span>
        </div>
        <button className={styles.btn} onClick={onNewGame}>New Game</button>
      </div>
    </div>
  );
}
