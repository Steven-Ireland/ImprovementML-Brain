import numpy as np
import sys
import game_env
from gymnasium import spaces

from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.env.vector_env import VectorEnv
from ray.tune.registry import register_env
from ray.rllib.policy.policy import PolicySpec
from ray.rllib.env.policy_server_input import PolicyServerInput

num_policy_workers = int(sys.argv[1])

SERVER_ADDRESS = "localhost"
SERVER_BASE_PORT = 8776

# TODO: Make this a cli arg
action_size = 36
observation_size = 103

action_space = spaces.Box(np.array([-1] * action_size), np.array([1] * action_size), dtype='float32')
observation_space= spaces.Box(np.array([-np.inf] * observation_size), np.array([np.Inf] * observation_size), dtype='float32')

config = (
    PPOConfig()
    .environment(
        env=None, 
        observation_space=observation_space, #TODO
        action_space=action_space, #TODO
    )
    # Use the existing algorithm process to run the server.
    .rollouts(num_rollout_workers=num_policy_workers, enable_connectors=False)
    .resources(num_gpus=1)
    .training(train_batch_size=5000, model={"fcnet_hiddens": [512, 512]})
    .multi_agent(policies={"main": PolicySpec(
        observation_space= observation_space, #TODO
        action_space= action_space, #TODO
    )}, policy_mapping_fn=lambda agent_id, episode, worker, **kwargs: "main")
    .framework("torch")
    .checkpointing(export_native_model_files=True)
)

# algo = Algorithm.from_checkpoint("/home/scien/ray_results/PPO_game_env_2023-04-16_14-05-435vw2cbfc/checkpoint_000101")

if __name__ == "__main__":
    # `InputReader` generator (returns None if no input reader is needed on
    # the respective worker).
    def _input(ioctx):
        # We are remote worker or we are local worker with num_workers=0:
        # Create a PolicyServerInput.
        if ioctx.worker_index > 0 or ioctx.worker.num_workers == 0:
            return PolicyServerInput(
                ioctx,
                SERVER_ADDRESS,
                SERVER_BASE_PORT + ioctx.worker_index - (1 if ioctx.worker_index > 0 else 0),
            )
        # No InputReader (PolicyServerInput) needed.
        else:
            return None

    algo = config.offline_data(input_=_input).build(use_copy=False)

    for i in range(5000):
        results = algo.train()
        reward_mean = results["episode_reward_mean"]

        print(f"-----EPOCH {i} | REWARD: {reward_mean} -----", flush=True)
        
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