import type { Card, Rank } from './types';
import { SUITS } from './types';

export const RANK_COUNTS: Record<Rank, number> = {
  1: 3,
  2: 2,
  3: 2,
  4: 2,
  5: 1,
};

export function createDeck(): Card[] {
  const cards: Card[] = [];
  for (const suit of SUITS) {
    for (const [rankStr, count] of Object.entries(RANK_COUNTS)) {
      const rank = Number(rankStr) as Rank;
      for (let i = 0; i < count; i++) {
        cards.push({
          id: `${suit}-${rank}-${String.fromCharCode(97 + i)}`,
          suit,
          rank,
        });
      }
    }
  }
  return cards;
}

export function shuffleDeck(deck: Card[]): Card[] {
  const shuffled = [...deck];
  for (let i = shuffled.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [shuffled[i], shuffled[j]] = [shuffled[j], shuffled[i]];
  }
  return shuffled;
}
