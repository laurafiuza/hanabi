import { useState, useEffect, useRef } from 'react';
import styles from './GameOverScreen.module.css';

interface GameOverScreenProps {
  status: string;
  score: number;
  reason: string | null;
  onNewGame: () => void;
}

const IMPRESSIONS = [
  { min: 0,  max: 5,  text: 'Horrible, booed by the crowd...' },
  { min: 6,  max: 10, text: 'Mediocre, just a hint of scattered applause...' },
  { min: 11, max: 15, text: 'Honorable attempt, but quickly forgotten...' },
  { min: 16, max: 20, text: 'Excellent, crowd pleasing!' },
  { min: 21, max: 24, text: 'Amazing, they will be talking about it for weeks!' },
  { min: 25, max: 25, text: 'Legendary, everyone left speechless, stars in their eyes!' },
];

function getImpression(score: number): string {
  for (const tier of IMPRESSIONS) {
    if (score >= tier.min && score <= tier.max) return tier.text;
  }
  return IMPRESSIONS[0].text;
}

// Firework colors matching the 5 Hanabi suits
const FIREWORK_COLORS = ['#e8e8e8', '#ffd700', '#4caf50', '#42a5f5', '#ef5350'];

interface Particle {
  x: number;
  y: number;
  vx: number;
  vy: number;
  color: string;
  life: number;
  maxLife: number;
  size: number;
}

interface Rocket {
  x: number;
  y: number;
  vy: number;
  targetY: number;
  color: string;
  exploded: boolean;
}

function launchFireworks(canvas: HTMLCanvasElement, intensity: number) {
  const ctx = canvas.getContext('2d')!;
  let particles: Particle[] = [];
  let rockets: Rocket[] = [];
  let animId: number;
  let lastLaunch = 0;

  const launchInterval = Math.max(200, 1200 - intensity * 40);
  const particlesPerBurst = Math.min(80, 20 + intensity * 3);

  function resize() {
    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;
  }
  resize();
  window.addEventListener('resize', resize);

  function spawnRocket() {
    const color = FIREWORK_COLORS[Math.floor(Math.random() * FIREWORK_COLORS.length)];
    rockets.push({
      x: Math.random() * canvas.width * 0.8 + canvas.width * 0.1,
      y: canvas.height,
      vy: -(8 + Math.random() * 4),
      targetY: canvas.height * (0.15 + Math.random() * 0.35),
      color,
      exploded: false,
    });
  }

  function explode(rocket: Rocket) {
    for (let i = 0; i < particlesPerBurst; i++) {
      const angle = (Math.PI * 2 * i) / particlesPerBurst + (Math.random() - 0.5) * 0.3;
      const speed = 2 + Math.random() * 4;
      const life = 40 + Math.random() * 30;
      particles.push({
        x: rocket.x,
        y: rocket.y,
        vx: Math.cos(angle) * speed,
        vy: Math.sin(angle) * speed,
        color: rocket.color,
        life,
        maxLife: life,
        size: 2 + Math.random() * 2,
      });
    }
  }

  function animate(now: number) {
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    if (now - lastLaunch > launchInterval) {
      spawnRocket();
      lastLaunch = now;
    }

    for (const r of rockets) {
      if (r.exploded) continue;
      r.y += r.vy;
      ctx.beginPath();
      ctx.arc(r.x, r.y, 2, 0, Math.PI * 2);
      ctx.fillStyle = r.color;
      ctx.fill();

      if (r.y <= r.targetY) {
        r.exploded = true;
        explode(r);
      }
    }
    rockets = rockets.filter(r => !r.exploded);

    for (const p of particles) {
      p.x += p.vx;
      p.y += p.vy;
      p.vy += 0.06;
      p.vx *= 0.99;
      p.life--;

      const alpha = p.life / p.maxLife;
      ctx.beginPath();
      ctx.arc(p.x, p.y, p.size * alpha, 0, Math.PI * 2);
      ctx.fillStyle = p.color;
      ctx.globalAlpha = alpha;
      ctx.fill();
    }
    ctx.globalAlpha = 1;
    particles = particles.filter(p => p.life > 0);

    animId = requestAnimationFrame(animate);
  }

  animId = requestAnimationFrame(animate);
  spawnRocket();

  return () => {
    cancelAnimationFrame(animId);
    window.removeEventListener('resize', resize);
  };
}

export function GameOverScreen({ status, score, reason, onNewGame }: GameOverScreenProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [dismissed, setDismissed] = useState(false);

  let title = 'Game Over';
  let subtitle = '';

  if (status === 'won') {
    title = 'Legendary!';
    subtitle = 'Everyone left speechless, stars in their eyes!';
  } else if (status === 'lost') {
    title = 'Boom!';
    subtitle = reason ?? 'The fuse ran out...';
  } else {
    title = 'The Show is Over';
    subtitle = getImpression(score);
  }

  const showFireworks = score > 5;

  useEffect(() => {
    if (!showFireworks || !canvasRef.current) return;
    return launchFireworks(canvasRef.current, score);
  }, [showFireworks, score]);

  return (
    <>
      {/* Fireworks canvas is always on top as a pure overlay — doesn't block interaction */}
      {showFireworks && (
        <canvas ref={canvasRef} className={styles.fireworks} />
      )}

      {/* Modal card — can be dismissed to see the game state underneath */}
      {!dismissed && (
        <div className={styles.overlay} onClick={() => setDismissed(true)}>
          <div className={styles.card} onClick={e => e.stopPropagation()}>
            <button className={styles.dismiss} onClick={() => setDismissed(true)} title="Close">
              &times;
            </button>
            <div className={styles.title}>{title}</div>
            <div className={styles.scoreDisplay}>
              <span className={styles.scoreNum}>{score}</span>
              <span className={styles.scoreMax}>/25</span>
            </div>
            <div className={styles.impression}>{subtitle}</div>
            <button className={styles.btn} onClick={onNewGame}>New Game</button>
          </div>
        </div>
      )}

      {/* After dismissing, show a small floating bar so they can still start a new game */}
      {dismissed && (
        <div className={styles.floatingBar}>
          <span className={styles.floatingScore}>{title} — {score}/25</span>
          <button className={styles.btn} onClick={onNewGame}>New Game</button>
        </div>
      )}
    </>
  );
}
