from stable_baselines import PPO2
from operator import itemgetter
import numpy as np
from urllib.parse import unquote
import time 
import cherrypy

cherrypy.config.update({'server.socket_port': 7775})

model = PPO2.load("trained_agent_40")

class Root(object):
    @cherrypy.expose
    @cherrypy.tools.json_in()
    @cherrypy.tools.json_out()
    def predict(self):
        observations = itemgetter('observations')(cherrypy.request.json)

        print(time.perf_counter())
        actions, _state =  model.predict(np.array(observations), deterministic=True)
        print(time.perf_counter(), flush=True)
        
        return({
            "actions": actions.tolist()
        })

if __name__ == "__main__":
    try:
        cherrypy.quickstart(Root(), '/')
    except KeyboardInterrupt:
        pass

    print("Server stopped.")