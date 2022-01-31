# Wordle Env: A Daily Word Environment for Reinforcement Learning
## Setup
Steps:
1. `git pull git@github.com:alex-nooj/wordle_env.git`
2. From the `wordle_env` directory: 
    * `pip install -e .`
    * `pip install -r requirements.txt`
    
## What's Included:
### `WordleCharEnv`
A Wordle Gym Environment that takes actions one character at a time.

The environment will provide a mask that lets the player/agent know which letters it can 
guess next. For example, if the agent guesses "q", then the mask will return that the 
only valid letters to guess next will be specific vowels, such as "a" and "u". Once the 
agent guesses "u", then the mask will include all the letters that can form a word that
starts with "qu", and so on.

The environment will also make sure that a letter that leads to an invalid letter cannot
be entered. In the case that this does occur, the environment will remain on the current 
state.

### `WordleWordEnv`
A Wordle Gym Environment that takes actions a whole word at a time.

Similar to the `WordleCharEnv`, if an invalid word is entered into the `WordleWordEnv`, 
the environment will produce a warning and remain on the current state.