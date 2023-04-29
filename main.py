import numpy as np
import sys
from gymnasium import spaces

from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.env.vector_env import VectorEnv
from ray.tune.registry import register_env
from ray.rllib.policy.policy import PolicySpec
from ray.rllib.env.policy_server_input import PolicyServerInput
from ray.tune.logger.tensorboardx import TBXLoggerCallback
from environment.unity_env import UnityEnv

SERVER_BASE_PORT = 8776

# TODO: Make this a cli arg
action_size = 36
observation_size = 103

action_space = spaces.Box(np.array([-1] * action_size), np.array([1] * action_size), dtype='float32')
observation_space= spaces.Box(np.array([-np.inf] * observation_size), np.array([np.Inf] * observation_size), dtype='float32')

register_env("unity_env", lambda config: UnityEnv(7776))

# algo = Algorithm.from_checkpoint("/home/scien/ray_results/PPO_game_env_2023-04-16_14-05-435vw2cbfc/checkpoint_000101")

if __name__ == "__main__":
    config = (
        PPOConfig()
        .environment(
            env='unity_env', 
        )
        # Use the existing algorithm process to run the server.
        .rollouts(num_rollout_workers=1, rollout_fragment_length=64)
        .resources(num_gpus=1)
        # 60hz * 5 seconds * 10 instances
        .training(train_batch_size=256, model={"fcnet_hiddens": [512, 512]})
        .multi_agent(policies={
            "main": PolicySpec(
                observation_space= observation_space, #TODO
                action_space= action_space, #TODO
            )}, 
            policy_mapping_fn=lambda agent_id, episode, worker, **kwargs: "main",
            count_steps_by='env_steps',
            policy_states_are_swappable=True,
            policies_to_train=["main"]
        )
        .framework("torch")
        .checkpointing(export_native_model_files=True)
    )

    algo = config.build(use_copy=False)

    for i in range(5000):
        results = algo.train()

        print(f"-----EPOCH {i} ------", flush=True)
        print(results["episode_reward_mean"])
        
        if (i%500 == 0):
            checkpoint_dir = algo.save('checkpoints/')
            policy = algo.get_policy('main')
            policy.export_model(f"./trained/model_{i}_result")
            print(f"Checkpoint saved in directory {checkpoint_dir}")

    # obs = env.reset()
    # for i in range(1000):
    #     action, _states = model.predict(obs, deterministic=True)
    #     obs, reward, done, info = env.step(action)
    #     if done:
    #         obs = env.reset()