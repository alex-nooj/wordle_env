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
        self.guess_set = [w for w in valid_words + valid_answers]

    def __call__(self, guess: str) -> np.ndarray:
        if len(guess) == 5 or len(guess) == 0:
            self.reset()
            return np.ones(26)

        self.guess_set = [
            word
            for word in self.guess_set
            if word[:len(guess)] == guess
        ]

        mask = np.zeros(26)
        for word in self.guess_set:
            mask[convert_letter(word[len(guess)])] = 1
        return mask

    def reset(self):
        self.guess_set = [w for w in valid_words + valid_answers]
