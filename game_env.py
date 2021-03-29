import json
from operator import itemgetter
from gym import Env
import numpy as np

# Initializes env based on websocket input - 
class GameEnv(Env):
    seed = 1

    def __init__(self, websocket, observation_space, action_space):
        self.websocket = websocket
        
        self.observation_space = observation_space
        self.action_space = action_space

    def step(self, actions):
        #print(f"Sending actions {actions}")
        self.websocket.send(json.dumps({'actions': actions.tolist()})) # Send actions to game
        #print(f"Sent actions, waiting for response")
        resp = self.websocket.recv() # Get observation, reward back
        #print(f"Got response {resp}")

        reward, observations, done = itemgetter('reward', 'observations', 'done')(json.loads(resp))

        return np.array(observations), reward, done, {}
    
    def reset(self):
        #print("Sending reset")
        self.websocket.send("RESET")
        #print("Sent Reset")

        resp = self.websocket.recv()
        observations = itemgetter('observations')(json.loads(resp))

        #print(f"Got response {resp}")
        return np.array(observations)
    
    def render(self):
        return
    
    def close(self):
        return
    
    def seed(self, seed):
        self.seed = seed
