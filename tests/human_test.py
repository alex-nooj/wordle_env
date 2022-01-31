from wordle_env.src.wordle_char_env import WordleCharEnv


def run():
    env = WordleCharEnv()

    env.reset()
    done = False

    print(env._answer)

    while not done:
        c = input()
        if len(c) > 1:
            c = c[0]
        obs, reward, done, mask = env.step(ord(c) - ord("a"))
        env.render()


if __name__ == "__main__":
    run()
