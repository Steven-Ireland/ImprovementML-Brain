from gym import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv #, VecNormalize
import numpy as np
import sys

import game_env

BUF_SIZE = 8192

if __name__ == "__main__":
    env = SubprocVecEnv([game_env.make_env(i) for i in range(int(sys.argv[1]))]) #, norm_obs=True, norm_reward=True)

    model = PPO("MlpPolicy", env, verbose=1)

    for i in range(10):
        print(f"-----EPOCH {i}-----", flush=True)
        model.learn(total_timesteps=1000000)
        model.save(f"isitworking{i}")

    # obs = env.reset()
    # for i in range(1000):
    #     action, _states = model.predict(obs, deterministic=True)
    #     obs, reward, done, info = env.step(action)
    #     if done:
    #         obs = env.reset()