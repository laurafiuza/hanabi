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
    <div className={styles.banner}>
      <div className={styles.bannerContent}>
        <span className={styles.title}>{title}</span>
        <span className={styles.subtitle}>{subtitle}</span>
        <span className={styles.score}>{score}/25</span>
        <button className={styles.btn} onClick={onNewGame}>New Game</button>
      </div>
    </div>
  );
}
