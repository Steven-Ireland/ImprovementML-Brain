import multiprocessing as mp
import subprocess
import os
import sys
import time
from ray.tune.logger import pretty_print
from ray.rllib.env.policy_client import PolicyClient
import datetime

BASE_ENV_PORT = 7776
BASE_POLICY_PORT = 8776

GAME_PATH = "C:\\Users\\scien\\ImprovementML\\bin\\ImprovementML.exe"
                

if __name__ == "__main__":
    games = []
    num_internal = int(sys.argv[1])

    games.append(subprocess.Popen([GAME_PATH, "-p", str(7776), "-n", str(num_internal), "-batchmode", "-nographics"]))
    
    try:
        for game in games:
            game.wait()
    except KeyboardInterrupt:
        print("Exiting")
        
    for game in games:
        game.kill()

