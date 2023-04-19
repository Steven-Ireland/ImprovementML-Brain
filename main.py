import numpy as np
import sys
import game_env

from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.env.vector_env import VectorEnv
from ray.tune.registry import register_env

num_game_instances = int(sys.argv[1])
num_internal_envs = int(sys.argv[2])

register_env("game_env", lambda config: game_env.GameEnv(num_game_instances, config))

algo = (
    PPOConfig()
    .environment(env="game_env")
    .rollouts(num_envs_per_worker=num_internal_envs, num_rollout_workers=num_game_instances, enable_connectors=False)
    .resources(num_gpus=1)
    .training(train_batch_size=5000, model={"fcnet_hiddens": [512, 512]})
    .framework("torch")
    .checkpointing(export_native_model_files=True)
    .build(use_copy=False)
)

# algo = Algorithm.from_checkpoint("/home/scien/ray_results/PPO_game_env_2023-04-16_14-05-435vw2cbfc/checkpoint_000101")

if __name__ == "__main__":
    for i in range(5000):
        print(f"-----EPOCH {i}-----", flush=True)
        algo.train()
        if (i%500 == 0):
            # checkpoint_dir = algo.save()
            policy = algo.get_policy()
            policy.export_model(f"./trained/model_{i}_result")
            # print(f"Checkpoint saved in directory {checkpoint_dir}")

    # obs = env.reset()
    # for i in range(1000):
    #     action, _states = model.predict(obs, deterministic=True)
    #     obs, reward, done, info = env.step(action)
    #     if done:
    #         obs = env.reset()