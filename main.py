import websocket
from gym import spaces
from stable_baselines3 import PPO
import numpy as np

import game_env

def start(websocket):
    greeting = websocket.recv()
    print(f"> {greeting}")

    # See observation.jsonc for the spec
    high = np.array([np.inf] * 15)
    observation_space = spaces.Box(-high, high, dtype='float32')
    action_space = spaces.Box(np.array([-1] * 2), np.array([1] * 2), dtype='float32')

    env = game_env.GameEnv(websocket, observation_space, action_space)
    obs = env.reset()

    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=200000)
    model.save("isitworking")

    obs = env.reset()
    for i in range(1000):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done = env.step(action)
        if done:
            obs = env.reset()

ws = websocket.WebSocket()
ws.connect("ws://localhost:7777")

start(ws)