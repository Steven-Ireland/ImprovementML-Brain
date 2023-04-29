from operator import itemgetter
import random
import numpy as np
from urllib.parse import unquote
import time 
import cherrypy
import torch

cherrypy.config.update({'server.socket_port': 7775})

pytorch_model = torch.load("./trained/model_2000_result/model.pt")

class Root(object):
    @cherrypy.expose
    @cherrypy.tools.json_in()
    @cherrypy.tools.json_out()
    def predict(self):
        observations = itemgetter('observations')(cherrypy.request.json)


        print(time.perf_counter())
        print(observations)
        print(len(observations))

        actions = pytorch_model(
            input_dict={
                "obs": torch.from_numpy(np.array([observations], dtype=np.float32)).to('cuda'),
            },
            state=[torch.tensor(0)],  # dummy value
            seq_lens=torch.tensor(0),  # dummy value
        )

        print("detached\n", flush=True)

        mean, log_std = torch.chunk(actions[0], 2, dim=1)
        parsed_mean = torch.flatten(mean).tolist()
        parsed_sigma = torch.flatten(log_std).tolist()

        parsed_actions = [0] * len(parsed_mean) 
        for x in range(len(parsed_mean)):
            parsed_actions[x] = random.normalvariate(parsed_mean[x], parsed_sigma[x])
        
        print(f"Action length: {parsed_actions}")
        print(time.perf_counter(), flush=True)
        
        return({
            "actions": parsed_actions
        })

if __name__ == "__main__":
    try:
        cherrypy.quickstart(Root(), '/')
    except KeyboardInterrupt:
        pass

    print("Server stopped.")