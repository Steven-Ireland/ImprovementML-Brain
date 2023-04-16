from stable_baselines import PPO2
from operator import itemgetter
import numpy as np
from urllib.parse import unquote
import time 
import cherrypy
import pickle

cherrypy.config.update({'server.socket_port': 7775})

model = PPO2.load("trained_agent_5")
with open('environment_5', 'rb') as f:
    vec_normalize = pickle.load(f)


class Root(object):
    @cherrypy.expose
    @cherrypy.tools.json_in()
    @cherrypy.tools.json_out()
    def predict(self):
        observations = itemgetter('observations')(cherrypy.request.json)

        print(time.perf_counter())
        print(observations)
        actions, _state =  model.predict(vec_normalize.normalize_obs(np.array(observations)), deterministic=True)
        print(actions)
        print(time.perf_counter(), flush=True)
        
        return({
            "actions": actions.tolist()
        })

if __name__ == "__main__":
    print("|||||||||||||||||")
    print(vec_normalize.obs_rms)
    print("|||||||||||||||||")

    try:
        cherrypy.quickstart(Root(), '/')
    except KeyboardInterrupt:
        pass

    print("Server stopped.")