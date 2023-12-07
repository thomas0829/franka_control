import base64
import requests
import time

import numpy as np
from helpers import inverse_discretize

ip_address = 'https://contributions-provides-bound-spanking.trycloudflare.com'
# IP address always changes every time the server is rebooted. Please check the text-generation-webui cli interface
msg = 'Pick up the cloth'
start = time.time()
with open('obs.png', 'rb') as f:
    img_str = base64.b64encode(f.read()).decode('utf-8')
    prompt = f'### human: \nWhat action should the robot take to `{msg}`\n<img src="data:image/jpeg;base64,{img_str}">### gpt: '
    response = requests.post(f'{ip_address}/v1/completions', json={'prompt': prompt, 'max_tokens': 256, 'stopping_strings': ['\n###']}).json()
    reponse_txt = response['choices'][0]['text']
    minmaxlst = [[-1,1],[-1,1],[-1,1],[-1,1],[-1,1],[-1,1]]

    action = inverse_discretize(reponse_txt, minmaxlst)[:3]
    action = np.concatenate((action, np.ones(1)))

end = time.time()
print(f'Took { end - start } seconds')