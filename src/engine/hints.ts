import type { HintInfo, HintValue, Player } from './types';

export function createEmptyHintInfo(handSize: number): HintInfo {
  return {
    knownSuits: new Array(handSize).fill(null),
    knownRanks: new Array(handSize).fill(null),
  };
}

export function applyHint(player: Player, hint: HintValue): Player {
  const newHintInfo: HintInfo = {
    knownSuits: [...player.hintInfo.knownSuits],
    knownRanks: [...player.hintInfo.knownRanks],
  };

  for (let i = 0; i < player.hand.length; i++) {
    const card = player.hand[i];
    if (hint.kind === 'suit' && card.suit === hint.suit) {
      newHintInfo.knownSuits[i] = hint.suit;
    } else if (hint.kind === 'rank' && card.rank === hint.rank) {
      newHintInfo.knownRanks[i] = hint.rank;
    }
  }

  return { ...player, hintInfo: newHintInfo };
}

export function removeCardFromHintInfo(hintInfo: HintInfo, cardIndex: number): HintInfo {
  const newSuits = [...hintInfo.knownSuits];
  const newRanks = [...hintInfo.knownRanks];
  newSuits.splice(cardIndex, 1);
  newRanks.splice(cardIndex, 1);
  return { knownSuits: newSuits, knownRanks: newRanks };
}

export function addCardToHintInfo(hintInfo: HintInfo): HintInfo {
  return {
    knownSuits: [...hintInfo.knownSuits, null],
    knownRanks: [...hintInfo.knownRanks, null],
  };
}
