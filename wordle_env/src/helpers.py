import numpy as np


def find_matches(guess: str, answer: str) -> np.ndarray:
    letters = {}
    for c in answer:
        if c in letters:
            letters[c] += 1
        else:
            letters[c] = 1

    matches = np.zeros(len(guess))
    for ix, (c1, c2) in enumerate(zip(guess, answer)):
        if c1 == c2:
            matches[ix] = 1
            letters[c1] -= 1

    for ix, (c1, c2) in enumerate(zip(guess, answer)):
        if c1 != c2 and c1 in answer and letters[c1] > 0:
            matches[ix] = 0.5
            letters[c1] -= 1
    return matches
