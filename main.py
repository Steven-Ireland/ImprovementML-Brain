from operator import itemgetter
import json
import numpy as np
import time

import game_env
import custom_vec_env

BUF_SIZE = 8192

POLICY_KWARGS = dict(net_arch=[300, 200, 100]) # Taken from deepmind

MODEL_KWARGS = dict(learning_rate=0.0005, noptepochs=10, nminibatches=8, n_steps=1024, gamma=0.995)

if __name__ == "__main__":
    from stable_baselines.ppo2 import PPO2
    from stable_baselines.common.policies import MlpPolicy
    from stable_baselines.common.vec_env import SubprocVecEnv, VecNormalize

    import tensorflow as tf
    tf.test.is_gpu_available(cuda_only=False, min_cuda_compute_capability=None)

    env = VecNormalize(custom_vec_env.BetterVecEnv([game_env.make_env(i) for i in range(0,32)], wait_duration=0.15, start_method="spawn")) #, norm_obs=True, norm_reward=True)

    model = PPO2(MlpPolicy, env, verbose=1, policy_kwargs=POLICY_KWARGS, **MODEL_KWARGS)

    for i in range(20):
        print("STARTING ROUND " + str(i))
        model.learn(total_timesteps=2000000)
        model.save("isitworking")
        env.save("isitworking_normalization") 
        print("FINISHED ROUND")

    print("FINISHED TRAINING!")