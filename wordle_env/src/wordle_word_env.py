from typing import Dict
from typing import Iterable
from typing import Tuple

import warnings
import numpy as np
from gym import spaces

from wordle_env.src.wordle_char_env import OBS_SHAPE
from wordle_env.src.wordle_char_env import WordleCharEnv


class WordleWordEnv(WordleCharEnv):
    """OpenAI Gym environment for wordle, where the action is 5 characters in order for the word.

    Inherits from the WordleCharEnv
    """
    def __init__(self):
        """Constructor for the environment.
        """
        super().__init__()
        self.action_space = spaces.MultiDiscrete([26 for _ in range(OBS_SHAPE[1])])

    def step(self, action: Iterable[int]) -> Tuple[np.ndarray, float, bool, Dict[str, np.ndarray]]:
        """The step function for the environment.

        Parameters
        ----------
        action : Iterable[int]
            Action containing integers in range [0, 26) representing the letters that you'd like to
            guess.

        Raises
        ------
        RuntimeError
            If the action is not the correct length.

        Returns
        -------
        Tuple[np.ndarray, float, bool, Dict[str, np.ndarray]
            The observation, reward, done flag, and a dictionary containing the action mask for the
            current step.
        """
        if len(action) != OBS_SHAPE[1]:
            raise RuntimeError(
                f"Action must be of length {OBS_SHAPE[1]}, recieved length {len(action)}"
            )

        guess = ""
        for c in action:
            guess += chr(ord("a") + c)

        if guess not in self._masker.guess_set:
            warnings.warn("Warning: Word is not in vocab.")
            return self._prev_step

        obs, reward, done, info = None, None, None, None
        for c in action:
            obs, reward, done, info = super().step(c, check=False)

        return obs, reward, done, info
