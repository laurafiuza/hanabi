import type { GameState, Action, Card, Player, Rank, Suit, HintValue } from './types';
import { SUITS, RANKS, MAX_INFO_TOKENS } from './types';
import { getValidActions } from './game';
import { chooseMCTSAction } from './mcts';

export const RANK_COPIES: Record<number, number> = { 1: 3, 2: 2, 3: 2, 4: 2, 5: 1 };

// ---------------------------------------------------------------------------
// Action scoring types
// ---------------------------------------------------------------------------

export type ActionTier = 'must' | 'strong' | 'neutral' | 'bad';

export interface ActionScore {
  tier: ActionTier;
  score: number;   // sub-score within tier (higher = better)
  reason: string;  // human-readable, used for tips
}

export const TIER_ORDER: Record<ActionTier, number> = { must: 3, strong: 2, neutral: 1, bad: 0 };

// ---------------------------------------------------------------------------
// Convention helpers
// ---------------------------------------------------------------------------

// The chop is the oldest (leftmost) unclued card — the one you'd discard next.
export function getChopIndex(player: Player): number {
  for (let i = 0; i < player.hand.length; i++) {
    if (player.hintInfo.knownSuits[i] === null && player.hintInfo.knownRanks[i] === null) {
      return i;
    }
  }
  return -1;
}

function isCardTrash(card: Card, state: GameState): boolean {
  if (state.playArea[card.suit] >= card.rank) return true;
  const nextNeeded = state.playArea[card.suit] + 1;
  for (let rank = nextNeeded; rank < card.rank; rank++) {
    const total = RANK_COPIES[rank];
    const discarded = state.discardPile.filter(c => c.suit === card.suit && c.rank === rank).length;
    if (discarded >= total) return true;
  }
  return false;
}

function hintTouchesTrash(target: Player, hint: HintValue, state: GameState): boolean {
  for (let i = 0; i < target.hand.length; i++) {
    const card = target.hand[i];
    const touches = hint.kind === 'suit' ? card.suit === hint.suit : card.rank === hint.rank;
    if (touches && isCardTrash(card, state)) return true;
  }
  return false;
}

function isCritical(card: Card, state: GameState): boolean {
  if (state.playArea[card.suit] >= card.rank) return false;
  const totalCopies = RANK_COPIES[card.rank];
  const discarded = state.discardPile.filter(c => c.suit === card.suit && c.rank === card.rank).length;
  return discarded >= totalCopies - 1;
}

function isRankUseless(knownRank: Rank, state: GameState): boolean {
  return SUITS.every(s => state.playArea[s] >= knownRank);
}

function isCardUseless(suit: Suit, rank: Rank, state: GameState): boolean {
  return state.playArea[suit] >= rank;
}

function isCardPlayable(suit: Suit, rank: Rank, state: GameState): boolean {
  return state.playArea[suit] === rank - 1;
}

function isSuitComplete(suit: Suit, state: GameState): boolean {
  return state.playArea[suit] >= 5;
}

// ---------------------------------------------------------------------------
// Endgame deduction: when deck is empty, deduce cards by elimination
// ---------------------------------------------------------------------------

interface DeducedCard {
  suit: Suit | null;
  rank: Rank | null;
}

// Build the pool of cards not visible anywhere (not in play area, discard, or other hands).
function buildUnknownPool(state: GameState, playerIndex: number): Map<string, number> {
  const counts = new Map<string, number>();
  for (const suit of SUITS) {
    for (const rank of RANKS) {
      counts.set(`${suit}-${rank}`, RANK_COPIES[rank]);
    }
  }
  for (const suit of SUITS) {
    for (let r = 1; r <= state.playArea[suit]; r++) {
      const key = `${suit}-${r}`;
      counts.set(key, counts.get(key)! - 1);
    }
  }
  for (const card of state.discardPile) {
    const key = `${card.suit}-${card.rank}`;
    counts.set(key, counts.get(key)! - 1);
  }
  for (let i = 0; i < state.players.length; i++) {
    if (i === playerIndex) continue;
    for (const card of state.players[i].hand) {
      const key = `${card.suit}-${card.rank}`;
      counts.set(key, counts.get(key)! - 1);
    }
  }
  return counts;
}

// For each card in the player's hand, compute the effective known suit/rank
// by combining hint info with elimination deduction (when deck is empty).
function deduceHand(state: GameState, playerIndex: number): DeducedCard[] {
  const me = state.players[playerIndex];
  const hand = me.hand;
  const result: DeducedCard[] = [];

  // If deck isn't empty, deduction isn't reliable — just use hint info
  if (state.deck.length > 0) {
    for (let i = 0; i < hand.length; i++) {
      result.push({ suit: me.hintInfo.knownSuits[i], rank: me.hintInfo.knownRanks[i] });
    }
    return result;
  }

  // Build pool of unseen cards
  const poolCounts = buildUnknownPool(state, playerIndex);

  // For each hand slot, compute candidates consistent with hints
  type CandidateKey = string; // "suit-rank"
  const slotCandidates: CandidateKey[][] = [];

  for (let i = 0; i < hand.length; i++) {
    const ks = me.hintInfo.knownSuits[i];
    const kr = me.hintInfo.knownRanks[i];
    const candidates: CandidateKey[] = [];
    for (const [key, count] of poolCounts) {
      if (count <= 0) continue;
      const [suit, rankStr] = key.split('-');
      if (ks !== null && suit !== ks) continue;
      if (kr !== null && Number(rankStr) !== kr) continue;
      candidates.push(key);
    }
    slotCandidates.push(candidates);
  }

  // Iterative constraint propagation: if a slot has exactly one candidate,
  // lock it in and remove that card from other slots' candidate lists.
  // Repeat until no more progress.
  const locked: (CandidateKey | null)[] = new Array(hand.length).fill(null);
  const remainingCounts = new Map(poolCounts);

  let changed = true;
  while (changed) {
    changed = false;
    for (let i = 0; i < hand.length; i++) {
      if (locked[i] !== null) continue;

      // Filter candidates by remaining counts
      const viable = slotCandidates[i].filter(key => (remainingCounts.get(key) ?? 0) > 0);
      slotCandidates[i] = viable;

      // Deduplicate: unique card identities
      const uniqueCards = [...new Set(viable)];

      if (uniqueCards.length === 1) {
        // Fully deduced
        locked[i] = uniqueCards[0];
        remainingCounts.set(uniqueCards[0], (remainingCounts.get(uniqueCards[0]) ?? 1) - 1);
        changed = true;
      }
    }
  }

  // Build result
  for (let i = 0; i < hand.length; i++) {
    const ks = me.hintInfo.knownSuits[i];
    const kr = me.hintInfo.knownRanks[i];
    if (locked[i] !== null) {
      const [suit, rankStr] = locked[i]!.split('-');
      result.push({ suit: suit as Suit, rank: Number(rankStr) as Rank });
    } else {
      result.push({ suit: ks, rank: kr });
    }
  }

  return result;
}

function isKnownSafePlay(knownSuit: Suit | null, knownRank: Rank | null, state: GameState): boolean {
  if (knownSuit && knownRank) return isCardPlayable(knownSuit, knownRank, state);
  if (knownRank && !knownSuit) {
    const allReady = SUITS.every(s => state.playArea[s] >= knownRank || state.playArea[s] === knownRank - 1);
    const someNeed = SUITS.some(s => state.playArea[s] === knownRank - 1);
    return allReady && someNeed;
  }
  return false;
}

function hintNewInfoCount(target: Player, hint: HintValue): number {
  let count = 0;
  for (let i = 0; i < target.hand.length; i++) {
    const card = target.hand[i];
    if (hint.kind === 'suit') {
      if (card.suit === hint.suit && target.hintInfo.knownSuits[i] !== hint.suit) count++;
    } else {
      if (card.rank === hint.rank && target.hintInfo.knownRanks[i] !== hint.rank) count++;
    }
  }
  return count;
}

// ---------------------------------------------------------------------------
// Hint classification helpers (classify a specific hint action)
// ---------------------------------------------------------------------------

// Does this hint save a critical card on the target's chop?
function isHintASave(state: GameState, targetIdx: number, hint: HintValue): boolean {
  const target = state.players[targetIdx];
  const chopIdx = getChopIndex(target);
  if (chopIdx < 0) return false;
  const chopCard = target.hand[chopIdx];
  if (!isCritical(chopCard, state)) return false;
  // Does the hint touch the chop card?
  return hint.kind === 'suit' ? chopCard.suit === hint.suit : chopCard.rank === hint.rank;
}

// Does this hint save a needed 2 on the target's chop?
function isHintA2Save(state: GameState, targetIdx: number, hint: HintValue): boolean {
  const target = state.players[targetIdx];
  const chopIdx = getChopIndex(target);
  if (chopIdx < 0) return false;
  const chopCard = target.hand[chopIdx];
  if (chopCard.rank !== 2 || state.playArea[chopCard.suit] >= 2) return false;
  if (isCritical(chopCard, state)) return false; // handled by isHintASave
  // Must be unclued
  if (target.hintInfo.knownSuits[chopIdx] !== null || target.hintInfo.knownRanks[chopIdx] !== null) return false;
  return hint.kind === 'suit' ? chopCard.suit === hint.suit : chopCard.rank === hint.rank;
}

// Does this hint touch at least one currently playable card?
function isHintAPlayClue(state: GameState, targetIdx: number, hint: HintValue): boolean {
  const target = state.players[targetIdx];
  for (let i = 0; i < target.hand.length; i++) {
    const card = target.hand[i];
    const touches = hint.kind === 'suit' ? card.suit === hint.suit : card.rank === hint.rank;
    if (!touches) continue;
    if (state.playArea[card.suit] === card.rank - 1) {
      // Only count if target doesn't already know it's playable
      const knowsSuit = target.hintInfo.knownSuits[i] === card.suit;
      const knowsRank = target.hintInfo.knownRanks[i] === card.rank;
      if (!(knowsSuit && knowsRank)) return true;
    }
  }
  return false;
}

// Does this hint touch a soon-needed card (within 2 ranks of playable)?
function isHintUseful(state: GameState, targetIdx: number, hint: HintValue): boolean {
  const target = state.players[targetIdx];
  for (let i = 0; i < target.hand.length; i++) {
    const card = target.hand[i];
    const touches = hint.kind === 'suit' ? card.suit === hint.suit : card.rank === hint.rank;
    if (!touches) continue;
    const needed = state.playArea[card.suit] + 1;
    if (card.rank >= needed && card.rank <= needed + 1) return true;
  }
  return false;
}

// ---------------------------------------------------------------------------
// Core scoring function
// ---------------------------------------------------------------------------

export function scoreAction(state: GameState, playerIndex: number, action: Action): ActionScore {
  const me = state.players[playerIndex];

  // Use deduction (hint info + elimination when deck is empty) for effective knowledge
  const deduced = deduceHand(state, playerIndex);

  if (action.type === 'PLAY_CARD') {
    const knownSuit = deduced[action.cardIndex].suit;
    const knownRank = deduced[action.cardIndex].rank;
    const isEndgame = state.turnsRemaining !== null;

    if (isKnownSafePlay(knownSuit, knownRank, state)) {
      const rank = knownRank!;
      // Plays outrank saves: playing is free (no token cost) and may
      // make the save unnecessary. Lower rank = higher priority.
      return { tier: 'must', score: 100 - rank, reason: `Play known-safe ${knownSuit ?? ''} ${rank}` };
    }

    if (knownSuit && knownRank) {
      return { tier: 'bad', score: -100, reason: `Playing ${knownSuit} ${knownRank} would bomb` };
    }

    // Partially known (suit or rank) — risky but acceptable in endgame
    if (knownSuit || knownRank) {
      if (isEndgame) {
        return { tier: 'neutral', score: 10, reason: `Endgame play (${knownSuit ?? '?'} ${knownRank ?? '?'})` };
      }
      return { tier: 'bad', score: -30, reason: `Playing a partially known card (${knownSuit ?? '?'} ${knownRank ?? '?'}) is risky` };
    }

    // Truly unknown — acceptable gamble in endgame, too risky otherwise
    if (isEndgame) {
      return { tier: 'neutral', score: 5, reason: 'Endgame desperation play' };
    }
    return { tier: 'bad', score: -200, reason: 'Playing a completely unknown card is too risky' };
  }

  if (action.type === 'DISCARD') {
    const knownSuit = deduced[action.cardIndex].suit;
    const knownRank = deduced[action.cardIndex].rank;

    // Known 5 — never discard
    if (knownRank === 5) {
      return { tier: 'bad', score: -50, reason: 'Never discard a known 5' };
    }

    // Known playable — don't discard
    if (isKnownSafePlay(knownSuit, knownRank, state)) {
      return { tier: 'bad', score: -100, reason: 'Discarding a known-playable card' };
    }

    // Known critical — don't discard
    if (knownSuit && knownRank && isCritical({ suit: knownSuit, rank: knownRank, id: '' }, state)) {
      return { tier: 'bad', score: -80, reason: `Discarding a critical ${knownSuit} ${knownRank}` };
    }

    // Known trash — good discard
    if (knownSuit && knownRank && isCardUseless(knownSuit, knownRank, state)) {
      return { tier: 'strong', score: 50, reason: `Discard known-trash ${knownSuit} ${knownRank}` };
    }
    if (knownRank && isRankUseless(knownRank, state)) {
      return { tier: 'strong', score: 48, reason: `Discard useless rank ${knownRank}` };
    }
    if (knownSuit && !knownRank && isSuitComplete(knownSuit, state)) {
      return { tier: 'strong', score: 46, reason: `Discard from completed suit ${knownSuit}` };
    }

    // Chop discard — acceptable (score higher when tokens are healthy to preserve hint economy)
    const chopIdx = getChopIndex(me);
    if (chopIdx >= 0 && action.cardIndex === chopIdx) {
      const tokenBonus = state.infoTokens >= MAX_INFO_TOKENS ? 40 : state.infoTokens >= 4 ? 10 : 0;
      return { tier: 'neutral', score: 35 + tokenBonus, reason: 'Discard chop' };
    }

    // Non-chop discard — bad
    if (knownSuit === null && knownRank === null) {
      return { tier: 'bad', score: -10, reason: "That wasn't your chop (oldest and leftmost unclued card). Consider discarding your chop instead." };
    }

    // Clued card that isn't known-trash — risky
    return { tier: 'bad', score: -20, reason: 'Discarding a clued card that might be useful' };
  }

  if (action.type === 'GIVE_HINT') {
    const target = state.players[action.targetPlayerIndex];
    const infoCount = hintNewInfoCount(target, action.hint);
    const touchesTrash = hintTouchesTrash(target, action.hint, state);

    // No new info — useless hint
    if (infoCount === 0) {
      return { tier: 'bad', score: 0, reason: 'This hint gives no new information' };
    }

    // Good Touch violation
    if (touchesTrash) {
      // Exception: if it's a critical save, downgrade to strong instead of bad
      if (isHintASave(state, action.targetPlayerIndex, action.hint)) {
        return { tier: 'strong', score: 40 + infoCount, reason: `Save critical on ${target.name}'s chop (touches trash)` };
      }
      return { tier: 'bad', score: 0, reason: "That hint touched a card that's already played or unreachable (Good Touch Principle)." };
    }

    // Critical save — must
    if (isHintASave(state, action.targetPlayerIndex, action.hint)) {
      return { tier: 'must', score: 80 + infoCount, reason: `Save critical card on ${target.name}'s chop` };
    }

    // 2-save — strong
    if (isHintA2Save(state, action.targetPlayerIndex, action.hint)) {
      return { tier: 'strong', score: 60 + infoCount, reason: `Save needed 2 on ${target.name}'s chop` };
    }

    // Play clue — strong
    if (isHintAPlayClue(state, action.targetPlayerIndex, action.hint)) {
      return { tier: 'strong', score: 55 + infoCount, reason: `Play clue for ${target.name}` };
    }

    // Useful hint — neutral
    if (isHintUseful(state, action.targetPlayerIndex, action.hint)) {
      return { tier: 'neutral', score: 30 + infoCount, reason: `Useful hint for ${target.name}` };
    }

    // Other hint with new info — neutral but low value
    return { tier: 'neutral', score: 10 + infoCount, reason: `Hint for ${target.name}` };
  }

  return { tier: 'neutral', score: 0, reason: 'Unknown action' };
}

// ---------------------------------------------------------------------------
// Score all actions and sort by tier + score
// ---------------------------------------------------------------------------

export function scoreAllActions(state: GameState, playerIndex: number): { action: Action; score: ActionScore }[] {
  const actions = getValidActions(state, playerIndex);
  const scored = actions.map(action => ({ action, score: scoreAction(state, playerIndex, action) }));
  scored.sort((a, b) => {
    const tierDiff = TIER_ORDER[b.score.tier] - TIER_ORDER[a.score.tier];
    if (tierDiff !== 0) return tierDiff;
    return b.score.score - a.score.score;
  });
  return scored;
}

// ---------------------------------------------------------------------------
// Heuristic bot (uses scoreAllActions)
// ---------------------------------------------------------------------------

export function heuristicBotAction(state: GameState, playerIndex: number): Action {
  const scored = scoreAllActions(state, playerIndex);
  const nonBad = scored.filter(s => s.score.tier !== 'bad');
  return (nonBad.length > 0 ? nonBad[0] : scored[0]).action;
}

// ---------------------------------------------------------------------------
// Bot entry point (delegates to MCTS with heuristic fallback)
// ---------------------------------------------------------------------------

export async function chooseBotAction(state: GameState, playerIndex: number): Promise<Action> {
  try {
    return await chooseMCTSAction(state, playerIndex);
  } catch {
    return heuristicBotAction(state, playerIndex);
  }
}

// ---------------------------------------------------------------------------
// Convention feedback for the human player
// ---------------------------------------------------------------------------

export function analyzeHumanAction(state: GameState, action: Action): string | null {
  const deduced = deduceHand(state, action.playerIndex);

  // Blunder: playing a card you know is not playable (guaranteed bomb)
  if (action.type === 'PLAY_CARD') {
    const { suit, rank } = deduced[action.cardIndex];
    if (suit && rank && state.playArea[suit] !== rank - 1) {
      return `You played ${suit} ${rank} but that suit needs a ${state.playArea[suit] + 1}.`;
    }
  }

  // Blunder: discarding a known-playable or known-critical card
  if (action.type === 'DISCARD') {
    const { suit, rank } = deduced[action.cardIndex];
    if (suit && rank && isCardPlayable(suit, rank, state)) {
      return `You discarded ${suit} ${rank} but it was playable right now.`;
    }
    if (suit && rank && isCritical({ suit, rank, id: '' }, state)) {
      return `You discarded ${suit} ${rank} — the last copy of a needed card.`;
    }
  }

  return null;
}
