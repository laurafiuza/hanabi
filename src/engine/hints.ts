import type { HintInfo, HintValue, Player } from './types';

export function createEmptyHintInfo(handSize: number): HintInfo {
  return {
    knownSuits: new Array(handSize).fill(null),
    knownRanks: new Array(handSize).fill(null),
    cardAges: new Array(handSize).fill(0),
  };
}

export function applyHint(player: Player, hint: HintValue): Player {
  const newHintInfo: HintInfo = {
    knownSuits: [...player.hintInfo.knownSuits],
    knownRanks: [...player.hintInfo.knownRanks],
    cardAges: [...player.hintInfo.cardAges],
  };

  for (let i = 0; i < player.hand.length; i++) {
    const card = player.hand[i];
    if (hint.kind === 'suit' && card.suit === hint.suit) {
      newHintInfo.knownSuits[i] = hint.suit;
      newHintInfo.cardAges[i] = 0; // reset age on hint
    } else if (hint.kind === 'rank' && card.rank === hint.rank) {
      newHintInfo.knownRanks[i] = hint.rank;
      newHintInfo.cardAges[i] = 0; // reset age on hint
    }
  }

  return { ...player, hintInfo: newHintInfo };
}

export function removeCardFromHintInfo(hintInfo: HintInfo, cardIndex: number): HintInfo {
  const newSuits = [...hintInfo.knownSuits];
  const newRanks = [...hintInfo.knownRanks];
  const newAges = [...hintInfo.cardAges];
  newSuits.splice(cardIndex, 1);
  newRanks.splice(cardIndex, 1);
  newAges.splice(cardIndex, 1);
  return { knownSuits: newSuits, knownRanks: newRanks, cardAges: newAges };
}

export function addCardToHintInfo(hintInfo: HintInfo): HintInfo {
  return {
    knownSuits: [...hintInfo.knownSuits, null],
    knownRanks: [...hintInfo.knownRanks, null],
    cardAges: [...hintInfo.cardAges, 0],
  };
}
