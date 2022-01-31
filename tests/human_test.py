from wordle_env.src.wordle_char_env import WordleCharEnv


def run():
    env = WordleCharEnv()

    env.reset()
    done = False

    print(env._answer)

    while not done:
        guess = ""
        print("Guess: ")
        while len(guess) < 5:
            guess += input()
        if len(guess) > 5:
            guess = guess[:5]

        for c in guess:
            obs, reward, done, mask = env.step(ord(c) - ord("a"))
        env.render()


if __name__ == "__main__":
    run()
