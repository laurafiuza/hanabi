export type Suit = 'white' | 'yellow' | 'green' | 'blue' | 'red';
export type Rank = 1 | 2 | 3 | 4 | 5;

export const SUITS: Suit[] = ['white', 'yellow', 'green', 'blue', 'red'];
export const RANKS: Rank[] = [1, 2, 3, 4, 5];

export const SUIT_COLORS: Record<Suit, string> = {
  white: '#e8e8e8',
  yellow: '#ffd700',
  green: '#4caf50',
  blue: '#42a5f5',
  red: '#ef5350',
};

export interface Card {
  id: string;
  suit: Suit;
  rank: Rank;
}

export interface HintInfo {
  knownSuits: (Suit | null)[];
  knownRanks: (Rank | null)[];
}

export interface Player {
  id: number;
  name: string;
  isHuman: boolean;
  hand: Card[];
  hintInfo: HintInfo;
}

export type HintValue =
  | { kind: 'suit'; suit: Suit }
  | { kind: 'rank'; rank: Rank };

export type Action =
  | { type: 'PLAY_CARD'; playerIndex: number; cardIndex: number }
  | { type: 'DISCARD'; playerIndex: number; cardIndex: number }
  | { type: 'GIVE_HINT'; playerIndex: number; targetPlayerIndex: number; hint: HintValue };

export interface LastActionResult {
  action: Action;
  cardPlayed?: Card;
  success?: boolean;
}

export interface HistoryEntry {
  playerName: string;
  text: string;
  turn: number;
}

export interface GameState {
  players: Player[];
  deck: Card[];
  playArea: Record<Suit, number>;
  discardPile: Card[];
  infoTokens: number;
  fuseTokens: number;
  currentPlayerIndex: number;
  turnsRemaining: number | null;
  status: 'setup' | 'playing' | 'won' | 'lost' | 'finished';
  lastActionResult: LastActionResult | null;
  history: HistoryEntry[];
  turnNumber: number;
  gameOverReason: string | null;
}

export const MAX_INFO_TOKENS = 8;
export const INITIAL_FUSE_TOKENS = 1;
export const HAND_SIZE = 4; // 4 cards for 4-5 players
export const NUM_PLAYERS = 5;
