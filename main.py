import socket
from gym import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv #, VecNormalize
import numpy as np

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

        (clientsocket, address) = serversocket.accept()

        ## TODO: Pull the space inits into the env passed through this?
        greeting = clientsocket.recv(BUF_SIZE).decode()
        print(f"> {greeting}")

        # See observation.jsonc for the spec        
        ## TODO: Make the space size dynamically driven from the unity scene
        high = np.array([np.inf] * 15)
        observation_space = spaces.Box(-high, high, dtype='float32')
        action_space = spaces.Box(np.array([-1] * 2), np.array([1] * 2), dtype='float32')
        
        env = game_env.GameEnv(clientsocket, observation_space, action_space)
        obs = env.reset()

        return env

    return thunk

if __name__ == "__main__":
    env = SubprocVecEnv([make_env(i) for i in range(0,8)]) #, norm_obs=True, norm_reward=True)

    model = PPO("MlpPolicy", env, verbose=1)

    for i in range(20):
        model.learn(total_timesteps=100000)
        model.save("isitworking")

    obs = env.reset()
    for i in range(1000):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        if done:
            obs = env.reset()