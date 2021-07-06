import multiprocessing as mp
import os
import sys
import time

HOST = "127.0.0.1"
NUM_CLIENTS = 8
BASE_PORT = 7776
BUF_SIZE = 8192

GAME_PATH = "C:\\Users\\scien\\ImprovementML\\bin\\ImprovementML.exe"

def start_client(rank, render=False):
    opts = ""
    if (not render):
        opts += f" -batchmode -nographics "
    
    os.system(GAME_PATH + " -p " + str(rank + 7776) + opts)

if __name__ == "__main__":
    procs = []
    num_clients = int(sys.argv[1])
    render = len(sys.argv) > 2
    for i in range(num_clients):
        print(f"Starting client {i}", flush=True)
        client = mp.Process(target=start_client, args=(i,render,))
        client.start()
        procs.append(client)
        time.sleep(0.5)
    
    try:
        for proc in procs:
            proc.join()
    except KeyboardInterrupt:
        for proc in procs:
            proc.terminate()

