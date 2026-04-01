import type { GameState, Action, Card, Player, Suit, HistoryEntry } from './types';
import { SUITS, HAND_SIZE, NUM_PLAYERS, MAX_INFO_TOKENS, INITIAL_FUSE_TOKENS, nextPlayerInTurnOrder } from './types';
import { createDeck, shuffleDeck } from './deck';
import { createEmptyHintInfo, applyHint, removeCardFromHintInfo, addCardToHintInfo } from './hints';
import { isValidAction } from './validation';

export function createGame(): GameState {
  let deck = shuffleDeck(createDeck());
  const players: Player[] = [];

  const PLAYER_NAMES = ['You', 'Top Right', 'Bottom Left', 'Top Left', 'Bottom Right'];
  for (let i = 0; i < NUM_PLAYERS; i++) {
    const hand = deck.slice(0, HAND_SIZE);
    deck = deck.slice(HAND_SIZE);
    players.push({
      id: i,
      name: PLAYER_NAMES[i],
      isHuman: i === 0,
      hand,
      hintInfo: createEmptyHintInfo(HAND_SIZE),
    });
  }

  const playArea = {} as Record<Suit, number>;
  for (const suit of SUITS) {
    playArea[suit] = 0;
  }

  return {
    players,
    deck,
    playArea,
    discardPile: [],
    infoTokens: MAX_INFO_TOKENS,
    fuseTokens: INITIAL_FUSE_TOKENS,
    currentPlayerIndex: 0,
    turnsRemaining: null,
    status: 'playing',
    lastActionResult: null,
    history: [],
    turnNumber: 1,
    gameOverReason: null,
  };
}

const RANK_COPIES: Record<number, number> = { 1: 3, 2: 2, 3: 2, 4: 2, 5: 1 };

function checkUnwinnable(state: GameState): string | null {
  for (const suit of SUITS) {
    const needed = state.playArea[suit] + 1;
    if (needed > 5) continue; // suit already complete

    // For each rank still needed in this suit, check if all copies are in discard
    for (let rank = needed; rank <= 5; rank++) {
      const totalCopies = RANK_COPIES[rank];
      const discarded = state.discardPile.filter(c => c.suit === suit && c.rank === rank).length;
      if (discarded >= totalCopies) {
        return `All ${suit} ${rank}s have been discarded — ${suit} can never be completed`;
      }
    }
  }
  return null;
}

export function getScore(playArea: Record<Suit, number>): number {
  return SUITS.reduce((sum, suit) => sum + playArea[suit], 0);
}

function drawCard(player: Player, deck: Card[]): { player: Player; deck: Card[] } {
  if (deck.length === 0) return { player, deck };
  const [card, ...rest] = deck;
  return {
    player: {
      ...player,
      hand: [...player.hand, card],
      hintInfo: addCardToHintInfo(player.hintInfo),
    },
    deck: rest,
  };
}

function makeHistoryEntry(state: GameState, playerName: string, text: string): HistoryEntry {
  return { playerName, text, turn: state.turnNumber };
}

export function applyAction(state: GameState, action: Action): GameState {
  if (!isValidAction(state, action)) return state;

  let newState = { ...state, players: state.players.map(p => ({ ...p })), history: [...state.history] };
  let deck = [...state.deck];
  const actingPlayer = state.players[action.playerIndex];

  switch (action.type) {
    case 'PLAY_CARD': {
      const player = { ...newState.players[action.playerIndex] };
      const card = player.hand[action.cardIndex];
      const expectedRank = newState.playArea[card.suit] + 1;
      const success = card.rank === expectedRank;

      // Remove card from hand and update hints
      player.hand = player.hand.filter((_, i) => i !== action.cardIndex);
      player.hintInfo = removeCardFromHintInfo(player.hintInfo, action.cardIndex);

      if (success) {
        newState.playArea = { ...newState.playArea, [card.suit]: card.rank };
        // Playing a 5 gives back an info token
        if (card.rank === 5 && newState.infoTokens < MAX_INFO_TOKENS) {
          newState.infoTokens++;
        }
      } else {
        newState.fuseTokens--;
        newState.discardPile = [...newState.discardPile, card];
      }

      // Draw a new card
      const drawn = drawCard(player, deck);
      player.hand = drawn.player.hand;
      player.hintInfo = drawn.player.hintInfo;
      deck = drawn.deck;

      newState.players[action.playerIndex] = player;
      newState.deck = deck;
      newState.lastActionResult = { action, cardPlayed: card, success };

      if (success) {
        newState.history.push(makeHistoryEntry(state, actingPlayer.name, `played ${card.suit} ${card.rank} ✓`));
      } else {
        newState.history.push(makeHistoryEntry(state, actingPlayer.name, `played ${card.suit} ${card.rank} ✗ BOOM`));
      }

      // Check fuse loss
      if (newState.fuseTokens <= 0) {
        newState.status = 'lost';
        newState.gameOverReason = 'The fuse ran out...';
        newState.turnNumber = state.turnNumber + 1;
        return newState;
      }

      // Check if failed play made game unwinnable
      if (!success) {
        const reason = checkUnwinnable(newState);
        if (reason) {
          newState.status = 'lost';
          newState.gameOverReason = reason;
          newState.turnNumber = state.turnNumber + 1;
          return newState;
        }
      }

      // Check win
      if (getScore(newState.playArea) === 25) {
        newState.status = 'won';
        newState.turnNumber = state.turnNumber + 1;
        return newState;
      }
      break;
    }

    case 'DISCARD': {
      const player = { ...newState.players[action.playerIndex] };
      const card = player.hand[action.cardIndex];

      player.hand = player.hand.filter((_, i) => i !== action.cardIndex);
      player.hintInfo = removeCardFromHintInfo(player.hintInfo, action.cardIndex);
      newState.discardPile = [...newState.discardPile, card];

      if (newState.infoTokens < MAX_INFO_TOKENS) {
        newState.infoTokens++;
      }

      const drawn = drawCard(player, deck);
      player.hand = drawn.player.hand;
      player.hintInfo = drawn.player.hintInfo;
      deck = drawn.deck;

      newState.players[action.playerIndex] = player;
      newState.deck = deck;
      newState.lastActionResult = { action, cardPlayed: card };
      newState.history.push(makeHistoryEntry(state, actingPlayer.name, `discarded ${card.suit} ${card.rank}`));

      // Check if discard made game unwinnable
      const reason = checkUnwinnable(newState);
      if (reason) {
        newState.status = 'lost';
        newState.gameOverReason = reason;
        newState.turnNumber = state.turnNumber + 1;
        return newState;
      }
      break;
    }

    case 'GIVE_HINT': {
      const target = applyHint(newState.players[action.targetPlayerIndex], action.hint);
      newState.players[action.targetPlayerIndex] = target;
      newState.infoTokens--;
      newState.lastActionResult = { action };
      const targetName = state.players[action.targetPlayerIndex].name;
      const desc = action.hint.kind === 'suit' ? action.hint.suit : String(action.hint.rank);
      newState.history.push(makeHistoryEntry(state, actingPlayer.name, `told ${targetName} about ${desc}s`));
      break;
    }
  }

  // Handle deck exhaustion / end-game countdown
  if (deck.length === 0 && newState.turnsRemaining === null) {
    newState.turnsRemaining = NUM_PLAYERS;
  }

  if (newState.turnsRemaining !== null) {
    newState.turnsRemaining--;
    if (newState.turnsRemaining <= 0) {
      newState.status = 'finished';
      newState.turnNumber = state.turnNumber + 1;
      return newState;
    }
  }

  // Advance turn (clockwise)
  newState.currentPlayerIndex = nextPlayerInTurnOrder(state.currentPlayerIndex);
  newState.turnNumber = state.turnNumber + 1;

  return newState;
}

export function getValidActions(state: GameState, playerIndex: number): Action[] {
  if (state.status !== 'playing' || state.currentPlayerIndex !== playerIndex) return [];

  const actions: Action[] = [];
  const player = state.players[playerIndex];

  // Play or discard any card
  for (let i = 0; i < player.hand.length; i++) {
    actions.push({ type: 'PLAY_CARD', playerIndex, cardIndex: i });
    actions.push({ type: 'DISCARD', playerIndex, cardIndex: i });
  }

  // Give hints (if tokens available)
  if (state.infoTokens > 0) {
    for (let t = 0; t < state.players.length; t++) {
      if (t === playerIndex) continue;
      const target = state.players[t];

      // Suit hints
      const suitsInHand = new Set(target.hand.map(c => c.suit));
      for (const suit of suitsInHand) {
        actions.push({
          type: 'GIVE_HINT', playerIndex, targetPlayerIndex: t,
          hint: { kind: 'suit', suit },
        });
      }

      // Rank hints
      const ranksInHand = new Set(target.hand.map(c => c.rank));
      for (const rank of ranksInHand) {
        actions.push({
          type: 'GIVE_HINT', playerIndex, targetPlayerIndex: t,
          hint: { kind: 'rank', rank },
        });
      }
    }
  }

  return actions;
}

