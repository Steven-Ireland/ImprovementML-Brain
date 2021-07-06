from gym import spaces
from stable_baselines import PPO2
import numpy as np
import time

import game_env

if __name__ == "__main__":
    env = game_env.make_env(0)()

    model = PPO2.load("trained_agent_40")

    obs = env.reset()

    for i in range(10_000):
        action, _state = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        time.sleep(0.5 / 60.0)

        if done:
            obs = env.reset()