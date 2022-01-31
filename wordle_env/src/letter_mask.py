import numpy as np

from wordle_env.words.valid_answers import valid_answers
from wordle_env.words.valid_words import valid_words


def convert_letter(c: str) -> int:
    return ord(c) - ord("a")


def create_mask():
    all_words = valid_words + valid_answers
    mask = np.zeros((26, len(all_words[0]) - 1, 26))

    for word in all_words:
        for start_ix, c in enumerate(word[:-1]):
            for ix, c2 in enumerate(word[start_ix + 1 :]):
                mask[ord(c) - ord("a"), ix, ord(c2) - ord("a")] = 1
    return mask


class LetterMask:
    def __init__(self):
        self._mask = create_mask()

    def __call__(self, guess: str) -> np.ndarray:
        if len(guess) == 0:
            return np.ones(26)

        mask = np.ones(26)

        for ix, c in enumerate(guess):
            mask *= self._mask[convert_letter(c), len(guess) - ix - 1, :]
        return mask
