import json
import socket
from operator import itemgetter
from gym import Env, spaces
import numpy as np

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

        print("Socket " + str(7776 + rank) + " connected - obs_size=" + str(observation_size) + ", action_size=" + str(action_size)) 

        # See observation.jsonc for the spec        
        ## TODO: Make the space size dynamically driven from the unity scene
        high = np.array([np.inf] * observation_size)
        observation_space = spaces.Box(-high, high, dtype='float32')
        action_space = spaces.Box(np.array([-1] * action_size), np.array([1] * action_size), dtype='float32')
        
        env = GameEnv(clientsocket, observation_space, action_space)
        obs = env.reset()

        return env

    return thunk

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
