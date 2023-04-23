import json
from typing import Tuple
import numpy as np
import random
import time
import socket
import datetime

from gymnasium import spaces

from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.utils.annotations import PublicAPI
from ray.rllib.utils.typing import MultiAgentDict

from environment.network_utils import socket_read, socket_write

def map_object(value_mapper, some_dict):
  return {key: value_mapper(value) for key, value in some_dict.items()}

def is_done(some_dict):
    return all([v['done'] for _, v in some_dict.items()])

def process_step_result(step_result):
    return (
      # Observations
      map_object(lambda v: np.array(v['observations'], dtype='float32'), step_result),
      # Rewards
      map_object(lambda v: v['reward'], step_result),
      # Done
      {"__all__": is_done(step_result)},
      # Terminate
      {"__all__": False},
      # Info
      map_object(lambda v: {}, step_result)
    ) 

# TODO: Make this a cli arg
action_size = 36
observation_size = 103

action_space = spaces.Box(np.array([-1] * action_size), np.array([1] * action_size), dtype='float32')
observation_space= spaces.Box(np.array([-np.inf] * observation_size), np.array([np.Inf] * observation_size), dtype='float32')


@PublicAPI
class UnityEnv(MultiAgentEnv):
    """A MultiAgentEnv representing a single Unity3D game instance."""

    def __init__(
        self,
        port: int = None,
    ):
        super().__init__()

        # Try connecting to the Unity3D game instance. If a port is blocked
        clientsocket = None
        while True:
            try:
              # create an INET, STREAMing socket
              serversocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
              # bind the socket to a public host, and a well-known port
              serversocket.bind(('', port))
              # become a server socket
              serversocket.listen(1)
            except:
              time.sleep(random.randint(1, 10))
              continue

            print(f"waiting for connection on port {port}", flush=True)

            (clientsocket, _) = serversocket.accept()
            self.socket = clientsocket

            # Ignore this for now
            # greeting = clientsocket.recv(BUF_SIZE).decode()
            # observation_size, action_size = itemgetter('observationSize', 'actionSize')(json.loads(greeting))

            break
        
        # TODO: Initialize environment based on skeleton file
        self.action_space = action_space
        self.observation_space = observation_space

    def step(
        self, action_dict: MultiAgentDict
    ) -> Tuple[
        MultiAgentDict, MultiAgentDict, MultiAgentDict, MultiAgentDict, MultiAgentDict
    ]:
        """Performs one multi-agent step through the game.

        Args:
            action_dict: Multi-agent action dict with:
                keys=agent identifier consisting of
                [MLagents behavior name, e.g. "Goalie?team=1"] + "_" +
                [Agent index, a unique MLAgent-assigned index per single agent]

        Returns:
            tuple:
                - obs: Multi-agent observation dict.
                    Only those observations for which to get new actions are
                    returned.
                - rewards: Rewards dict matching `obs`.
                - dones: Done dict with only an __all__ multi-agent entry in
                    it. __all__=True, if episode is done for all agents.
                - infos: An (empty) info dict.
        """

        #print(f"Sending actions {action_dict}")

        
        socket_write(self.socket, json.dumps(map_object(lambda ac: {"actions": ac.tolist()}, action_dict))) # Send actions to game
        
        #print(f"Sent actions, waiting for response")

        before_step=datetime.datetime.now()

        resp = socket_read(self.socket) # Get observation, reward back
        after_step=datetime.datetime.now()
        delta = after_step - before_step

        print(f"Action retrieval took {delta.seconds}s {delta.microseconds / 1000.00}ms")
        #print(f"Got response {resp}", flush=True)

        agent_result_dict = json.loads(resp)
        return process_step_result(agent_result_dict)

    def reset(
        self, *, seed=None, options=None
    ) -> Tuple[MultiAgentDict, MultiAgentDict]:
        """
        Resets the entire Unity3D scene (a single multi-agent episode).
        
        Returns: 
          tuple:
            - Rewards
            - Infos
        """
        #print("Sending reset")
        socket_write(self.socket, "RESET")
        #print("Sent Reset")

        resp = socket_read(self.socket)
        agent_result_dict = json.loads(resp)

        #print(f"Got reset {agent_result_dict}")

        observations, _, _, _, infos = process_step_result(agent_result_dict)

        return (observations, infos)