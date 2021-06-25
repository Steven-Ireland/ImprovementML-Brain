import gym
import socket
from operator import itemgetter
import numpy as np
import multiprocessing as mp
import json
import sys

HOST = "127.0.0.1"
NUM_CLIENTS = 8
BASE_PORT = 7776
BUF_SIZE = 8192

def start_client(rank, render=False):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client:
        env = gym.make('BipedalWalker-v3')
        client.connect((HOST, BASE_PORT + rank))

        action_size = len(env.action_space.sample().flatten())
        observation_size = len(env.observation_space.sample().flatten())
        print(env.observation_space)

        print("Starting space with action size", action_size, "and obs size", observation_size)
        client.send(json.dumps({
            'action_size': action_size,
            'observation_size': observation_size
        }).encode())

        env.reset()
        
        while True:
            next_task = client.recv(BUF_SIZE).decode()
            if (next_task == "RESET"):
                observations = env.reset()
                client.send(json.dumps({
                    'observations': observations.tolist()
                }).encode())
                continue

            actions = itemgetter('actions')(json.loads(next_task))

            observations, reward, done, info = env.step(np.asarray(actions))

            client.send(json.dumps({
                'observations': observations.tolist(),
                'reward': reward,
                'done': done
            }).encode())

            if (render):
                env.render()


if __name__ == "__main__":
    procs = []
    num_clients = int(sys.argv[1])
    render = len(sys.argv) > 2
    for i in range(num_clients):
        client = mp.Process(target=start_client, args=(i,render,))
        client.start()
        procs.append(client)
    
    for proc in procs:
        proc.join()
