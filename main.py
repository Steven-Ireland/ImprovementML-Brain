from stable_baselines import PPO2
from stable_baselines.common.vec_env import SubprocVecEnv, VecNormalize
import numpy as np
import sys

import game_env

BUF_SIZE = 8192

policy_kwargs = dict(net_arch=[256, 128, 128])

if __name__ == "__main__":
    env = VecNormalize(SubprocVecEnv([game_env.make_env(i) for i in range(int(sys.argv[1]))]) , norm_obs=True)

    model = PPO2("MlpPolicy", env, n_steps=1024, verbose=True, policy_kwargs=policy_kwargs, tensorboard_log="./logs/")

    for i in range(101):
        print(f"-----EPOCH {i}-----", flush=True)
        model.learn(total_timesteps=1000000)
        if (i%5 == 0):
            model.save(f"trained_agent_{i}")
            env.save(f"environment_{i}")

    # obs = env.reset()
    # for i in range(1000):
    #     action, _states = model.predict(obs, deterministic=True)
    #     obs, reward, done, info = env.step(action)
    #     if done:
    #         obs = env.reset()