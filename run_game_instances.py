import multiprocessing as mp
import os
import sys
import time

HOST = "127.0.0.1"
NUM_CLIENTS = 8
BASE_PORT = 7776
BUF_SIZE = 8192

GAME_PATH = "/mnt/c/Users/scien/ImprovementML/bin/ImprovementML.exe"

def start_client(rank, num_internal, render=False):
    opts = ""
    if (not render):
        opts += f" -batchmode -nographics "

    offset = num_internal * rank
    
    os.system(GAME_PATH + " -p " + str(offset + 7776) + " -n " + str(num_internal) + opts)

if __name__ == "__main__":
    procs = []
    num_clients = int(sys.argv[1])
    num_internal = int(sys.argv[2])
    render = len(sys.argv) > 3
    for i in range(num_clients):
        print(f"Starting client {i}", flush=True)
        client = mp.Process(target=start_client, args=(i,num_internal, render,))
        client.start()
        procs.append(client)
        time.sleep(0.5)
    
    try:
        for proc in procs:
            proc.join()
    except KeyboardInterrupt:
        for proc in procs:
            proc.terminate()

