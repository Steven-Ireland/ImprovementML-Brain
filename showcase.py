import socket
from gym import spaces
from operator import itemgetter
import json
from stable_baselines.ppo2 import PPO2
import numpy as np

import game_env

BUF_SIZE = 8192


# create an INET, STREAMing socket
serversocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# bind the socket to a public host, and a well-known port
serversocket.bind(('', 7776))
# become a server socket
serversocket.listen(1)

(clientsocket, address) = serversocket.accept()

## TODO: Pull the space inits into the env passed through this?
greeting = clientsocket.recv(BUF_SIZE).decode()
observation_size, action_size = itemgetter('observationSize', 'actionSize')(json.loads(greeting))

# See observation.jsonc for the spec        
## TODO: Make the space size dynamically driven from the unity scene
high = np.array([np.inf] * observation_size)
observation_space = spaces.Box(-high, high, dtype='float32')
action_space = spaces.Box(np.array([-1] * action_size), np.array([1] * action_size), dtype='float32')

env = game_env.GameEnv(clientsocket, observation_space, action_space)
obs = env.reset()

model = PPO2.load("isitworking")

obs = env.reset()
for i in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    if done:
        obs = env.reset()