import socket
from gym import spaces
from operator import itemgetter
import json
from stable_baselines import PPO2
from stable_baselines.common.policies import MlpLstmPolicy
from stable_baselines.common.vec_env import SubprocVecEnv, VecNormalize
import numpy as np

import tensorflow as tf

import game_env

BUF_SIZE = 8192

def make_env(rank):
    def thunk():
        # create an INET, STREAMing socket
        serversocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # bind the socket to a public host, and a well-known port
        serversocket.bind(('', 7776 + rank))
        # become a server socket
        serversocket.listen(1)

        print("Listening on " + str(7776 + rank))
        (clientsocket, address) = serversocket.accept()

        ## TODO: Pull the space inits into the env passed through this?
        greeting = clientsocket.recv(BUF_SIZE).decode()
        observation_size, action_size = itemgetter('observationSize', 'actionSize')(json.loads(greeting))

        print("Socket " + str(7776 + rank) + " connected")

        # See observation.jsonc for the spec        
        ## TODO: Make the space size dynamically driven from the unity scene
        high = np.array([np.inf] * observation_size)
        observation_space = spaces.Box(-high, high, dtype='float32')
        action_space = spaces.Box(np.array([-1] * action_size), np.array([1] * action_size), dtype='float32')
        
        env = game_env.GameEnv(clientsocket, observation_space, action_space)
        obs = env.reset()

        return env

    return thunk

if __name__ == "__main__":
    tf.test.is_gpu_available(cuda_only=False, min_cuda_compute_capability=None)

    env = VecNormalize(SubprocVecEnv([make_env(i) for i in range(0,8)])) #, norm_obs=True, norm_reward=True)

    model = PPO2(MlpLstmPolicy, env, verbose=1, 
        n_steps=2048,
        ent_coef=0.001)

    for i in range(20):
        print("STARTING ROUND " + str(i))
        model.learn(total_timesteps=1000000)
        model.save("isitworking")
        print("FINISHED ROUND")

    print("FINISHED TRAINING!")