import time

from wordle_env.src.wordle_word_env import WordleWordEnv


def speed_test(env) -> int:
    env.reset()
    done = False
    guess = [ord(c) - ord("a") for c in "bhais"]
    steps = 0
    while not done:
        steps += 5
        obs, reward, done, mask = env.step(guess)
    return steps


if __name__ == "__main__":
    steps = 0
    start_time = time.time()
    for _ in range(100):
        steps += speed_test(WordleWordEnv())
    end_time = time.time()
    time_diff = end_time - start_time
    print(f"{steps} steps in {time_diff} seconds ({steps/time_diff} steps/sec)")
