import multiprocessing as mp
import subprocess
import os
import sys
import time
from ray.rllib.env.policy_client import PolicyClient
from environment.unity_env import UnityEnv
import datetime

BASE_ENV_PORT = 7776
BASE_POLICY_PORT = 8776

GAME_PATH = "/mnt/c/Users/scien/ImprovementML/bin/ImprovementML.exe"

def create_policy_client(policy_id):
    policy_port = BASE_POLICY_PORT + policy_id
    game_port = BASE_ENV_PORT + policy_id

    client = PolicyClient(
        f"http://localhost:{policy_port}", inference_mode="local"
    )

    env = UnityEnv(game_port)
    episode_id = client.start_episode()

    obs, _ = env.reset()

    while True:
        # print("Getting action", flush=True)
        action = client.get_action(episode_id, obs)
        # print("Got action", flush=True)

        obs, reward, terminated, truncated, info = env.step(action)

        client.log_returns(episode_id, reward)

        if terminated["__all__"] or truncated["__all__"]:
            client.end_episode(episode_id, obs)
            obs, _ = env.reset()
            episode_id = client.start_episode()

if __name__ == "__main__":
    procs = []
    games = []
    num_policy_workers = int(sys.argv[1])
    num_internal = int(sys.argv[2])

    for i in range(num_policy_workers):
        print(f"Starting client {i}", flush=True)

        """, """
        client = mp.Process(target=create_policy_client, args=([i]))
        client.start()
        procs.append(client)
        time.sleep(5)
        print(7776 + i)
        #games.append(subprocess.Popen([GAME_PATH, "-p", str(7776 + i), "-n", str(num_internal)]))
    
    try:
        for proc in procs:
            proc.join()
        for game in games:
            game.wait()
    except KeyboardInterrupt:
        print("Exiting")
        
    for proc in procs:
        proc.kill()
    for game in games:
        game.kill()

