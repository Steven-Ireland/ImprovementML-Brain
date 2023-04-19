import json
from operator import itemgetter
from gymnasium import Env, spaces
import numpy as np
import socket
from ray.rllib.env.external_env import ExternalEnv

BUF_SIZE = 8192

# Initializes env based on websocket input - 
class GameEnv(ExternalEnv):
    seed = 1

    def __init__(self, max_workers, env_config):
        print(env_config.worker_index)
        rank = (env_config.worker_index-1)

        # create an INET, STREAMing socket
        serversocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # bind the socket to a public host, and a well-known port
        serversocket.bind(('', 7776 + rank))
        # become a server socket
        serversocket.listen(1)

        print(f"waiting for connection on port {7776 + rank}", flush=True)

        (clientsocket, address) = serversocket.accept()

        greeting = clientsocket.recv(BUF_SIZE).decode()
        observation_size, action_size = itemgetter('observationSize', 'actionSize')(json.loads(greeting))

        high = np.array([np.inf] * observation_size)
        observation_space = spaces.Box(-high, high, dtype='float32')
        action_space = spaces.Box(np.array([-1] * action_size), np.array([1] * action_size), dtype='float32')

        print(action_size)
        print(action_space)

        print(f"Game Env {rank} set up", flush=True)
        
        self.socket = clientsocket
        self.observation_space = observation_space
        self.action_space = action_space
        ExternalEnv.__init__(self, self.action_space, self.observation_space)

    def run(self):
        eid = self.start_episode()
        obs = self.reset()
        while True:
            action = self.get_action(eid, obs)
            obs, reward, done, _, info = self.step(action)
            self.log_returns(eid, reward, info=info)
            if done:
                self.end_episode(eid, obs)
                obs = self.reset()
                eid = self.start_episode()
        

    def step(self, actions):
        # print(f"Sending actions {actions}")
        self.socket.send(json.dumps({'actions': actions.tolist()}).encode()) # Send actions to game
        #print(f"Sent actions, waiting for response")
        resp = self.socket.recv(BUF_SIZE).decode() # Get observation, reward back
        #print(f"Got response {resp}", flush=True)

        reward, observations, done = itemgetter('reward', 'observations', 'done')(json.loads(resp))

        return np.array(observations, dtype='float32'), reward, done, False, {}
    
    def reset(self, *, seed=None, options=None):
        #print("Sending reset")
        self.socket.send("RESET".encode())
        #print("Sent Reset")

        resp = self.socket.recv(BUF_SIZE).decode()
        observations = itemgetter('observations')(json.loads(resp))

        return np.array(observations, dtype='float32')
    
    def render(self):
        return
    
    def close(self):
        return
    
    def seed(self, seed):
        self.seed = seed