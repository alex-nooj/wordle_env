from typing import Optional

from colorama import Fore, Style
from colorama import Back

import numpy as np
import gym
from gym import spaces

from wordle_env.src.helpers import find_matches
from wordle_env.src.letter_mask import LetterMask
from wordle_env.words.valid_answers import valid_answers

OBS_SHAPE = (6, 5, 2)
OBS_DTYPE = np.float


class WordleCharEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.action_space = spaces.Discrete(26)
        self.observation_space = spaces.Box(
            low=0, high=27, shape=OBS_SHAPE, dtype=OBS_DTYPE
        )
        self.words = valid_answers
        self._answer = None
        self.state = None
        self.step_num = 0
        self._curr_guess = ""
        self._masker = LetterMask()

    def step(self, action: int):
        # Insert the new letter into the right space to get the observation
        row = self.step_num // OBS_SHAPE[1]
        col = self.step_num % OBS_SHAPE[1]
        self.state[row, col, 0] = action + 1

        self._curr_guess += chr(ord("a") + action)
        self.step_num += 1

        # Determine done condition
        done = self.step_num == OBS_SHAPE[0] * OBS_SHAPE[1]

        # Calculate reward
        if len(self._curr_guess) == 5:
            answer_check = find_matches(self._curr_guess, self._answer)
            self.state[row, :, 1] = answer_check
            reward = np.sum(answer_check)
            if reward == 5:
                reward = 10
                done = True
            self._curr_guess = ""
        else:
            reward = 0

        return (
            self.state,
            reward,
            done,
            {"mask": self._masker(self._curr_guess)},
        )

    def reset(self):
        self._answer = np.random.choice(self.words)
        self.state = np.zeros((6, 5, 2), dtype=OBS_DTYPE)
        self.step_num = 0

        return self.state

    def render(self, mode="human", close=False):
        if self.step_num % 5 == 0:
            for row in self.state:
                for col in row:
                    if col[0] == 0:
                        print(Fore.WHITE + Back.BLACK + " ", end="")
                    else:
                        if col[1] == 0:
                            print(
                                Fore.WHITE
                                + Back.BLACK
                                + chr(int(col[0]) + ord("a") - 1),
                                end="",
                            )
                        elif col[1] == 0.5:
                            print(
                                Fore.BLACK
                                + Back.YELLOW
                                + chr(int(col[0]) + ord("a") - 1),
                                end="",
                            )
                        else:
                            print(
                                Fore.BLACK
                                + Back.GREEN
                                + chr(int(col[0]) + ord("a") - 1),
                                end="",
                            )
                    print(Style.RESET_ALL, end="")
                print()
            print(f"Guess {self.step_num // 5}")

    def seed(self, seed: Optional[int] = 0):
        np.random.seed(seed)
