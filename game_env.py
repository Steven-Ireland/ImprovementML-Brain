import json
from operator import itemgetter
from gym import Env
import numpy as np

BUF_SIZE = 8192

# Initializes env based on websocket input - 
class GameEnv(Env):
    seed = 1

    def __init__(self, socket, observation_space, action_space):
        self.socket = socket
        
        self.observation_space = observation_space
        self.action_space = action_space

    def step(self, actions):
        #print(f"Sending actions {actions}")
        self.socket.send(json.dumps({'actions': actions.tolist()}).encode()) # Send actions to game
        #print(f"Sent actions, waiting for response")
        resp = self.socket.recv(BUF_SIZE).decode() # Get observation, reward back
        #print(f"Got response {resp}")

        reward, observations, done = itemgetter('reward', 'observations', 'done')(json.loads(resp))

        return np.array(observations), reward, done, {}
    
    def reset(self):
        #print("Sending reset")
        self.socket.send("RESET".encode())
        #print("Sent Reset")

        resp = self.socket.recv(BUF_SIZE).decode()
        observations = itemgetter('observations')(json.loads(resp))

        #print(f"Got response {resp}")
        return np.array(observations)
    
    def render(self):
        return
    
    def close(self):
        return
    
    def seed(self, seed):
        self.seed = seed
