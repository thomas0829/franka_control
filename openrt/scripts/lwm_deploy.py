"""
deploy.py

Provide a lightweight server/client implementation for deploying OpenVLA models (through the HF AutoClass API) over a
REST API. This script implements *just* the server, with specific dependencies and instructions below.

Note that for the *client*, usage just requires numpy/json-numpy, and requests; example usage below!

Dependencies:
    => Server (runs OpenVLA model on GPU): `pip install uvicorn fastapi json-numpy`
    => Client: `pip install requests json-numpy`

Client (Standalone) Usage (assuming a server running on 0.0.0.0:8000):

```
import requests
import json_numpy
json_numpy.patch()
import numpy as np

action = requests.post(
    "http://0.0.0.0:8000/act",
    json={"image": np.zeros((256, 256, 3), dtype=np.uint8), "instruction": "do something"}
).json()

Note that if your server is not accessible on the open web, you can use ngrok, or forward ports to your client via ssh:
    => `ssh -L 8000:localhost:8000 ssh USER@<SERVER_IP>`
"""

import os.path

# ruff: noqa: E402
import json_numpy

json_numpy.patch()
import json
import logging
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Union

import draccus
import numpy as np
# import torch
import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor

import sys
sys.path.append('/home/joel/seonghyeon/World-Model')
from lwm.delta_action_sampler_override_bridge import DeltaActionSampler
from lwm.action_sampler_bridge import ActionSampler
from tux import JaxDistributedConfig
from lwm.delta_llama import VideoLLaMAConfig
from tux import define_flags_with_default, JaxDistributedConfig, set_random_seed
import csv
import random

class FLAGSClass:
    def __init__(self, flag_dict):
        for key, value in flag_dict.items():
            setattr(self, key, value)

# def set_random_seed(seed):
#     random.seed(seed)
#     np.random.seed(seed)

# === Server Interface ===
class LWMServer:
    def __init__(
            self, 
            load_checkpoint: Union[str, Path], 
            vqgan_checkpoint: Union[str, Path], 
            seed: int,
            mesh_dim: str, 
            dtype: str, 
            load_llama_config: str, 
            updata_llama_config: str, 
            tokens_per_delta: int, 
            tokens_per_action: int, 
            vocab_file: str, 
            multi_image: int, 
            jax_distributed: dict,
            action_scale_file: str
        ) -> Path:
        """
        A simple server for OpenVLA models; exposes `/act` to predict an action for a given image + instruction.
            => Takes in {"image": np.ndarray, "instruction": str, "unnorm_key": Optional[str]}
            => Returns  {"action": np.ndarray}
        """
        
        # JaxDistributedConfig.initialize(args.jax_distributed)
        set_random_seed(seed)
        tokenizer = VideoLLaMAConfig.get_tokenizer_config()
        llama = VideoLLaMAConfig.get_default_config()
        tokenizer.vocab_file = vocab_file
        kwargs = {
            "vqgan_checkpoint": vqgan_checkpoint,
            "seed": seed,
            "mesh_dim": mesh_dim,
            "dtype": dtype,
            "load_llama_config": load_llama_config,
            "update_llama_config": updata_llama_config,
            "tokens_per_delta": tokens_per_delta,
            "tokens_per_action": tokens_per_action,
            "vocab_file": vocab_file,
            "multi_image": multi_image,
            "jax_distributed": jax_distributed,
            "action_scale_file": action_scale_file,
            "tokenizer": tokenizer,
            "llama": llama,
            "load_checkpoint": load_checkpoint,
            "image_aug": True,
        }
        self.tokens_per_delta = tokens_per_delta
        self.cnt = 0
        flags = FLAGSClass(kwargs)

        if kwargs['tokens_per_delta'] > 0:
            self.model = DeltaActionSampler(flags)
        else: 
            self.model = ActionSampler(flags)
        self.load_checkpoint= load_checkpoint
        self.action_scale_list = []
        with open(action_scale_file, 'r') as file:
            reader = csv.reader(file)
            next(reader) 
            for row in reader:
                # Convert the string values to float and add them to the csv_data list
                self.action_scale_list.append([float(value) for value in row if value.strip()])

        # self.device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

        # [Hacky] Load Dataset Statistics from Disk (if passing a path to a fine-tuned model)
        # if os.path.isdir(self.load_checkpoint):
        #     with open(Path(self.load_checkpoint) / "dataset_statistics.json", "r") as f:
        #         self.vla.norm_stats = json.load(f)

    def predict_action(self, payload: Dict[str, Any]) -> str:
        self.cnt +=1
        try:
            if double_encode := "encoded" in payload:
                # Support cases where `json_numpy` is hard to install, and numpy arrays are "double-encoded" as strings
                assert len(payload.keys() == 1), "Only uses encoded payload!"
                payload = json.loads(payload["encoded"])

            # Parse payload components
            image, instruction = payload["image"], payload["instruction"]
            # instruction = payload["instruction"]

            print("instruction", instruction)
            # convert list to numpy array
            
            # unnorm_key = payload.get("unnorm_key", None)

            prompts = [{'image': [image], 'question':instruction}]

            # save image to disk
            image2 = np.array(image).astype(np.uint8)
            image2 = Image.fromarray(image2)
            # image2.save(f"data/real_eval_milk/image_{self.cnt}.jpg")

            outputs, vision_output, text_output = self.model(prompts)
            norm_raw_actions = text_output[0]
            print("norm raw actions", norm_raw_actions)
            action = norm_raw_actions

            action = self.get_averaged_values(action)

            if double_encode:
                return JSONResponse(json_numpy.dumps(action))
            else:
                return JSONResponse(action)
        except:  # noqa: E722
            logging.error(traceback.format_exc())
            logging.warning(
                "Your request threw an error; make sure your request complies with the expected format:\n"
                "{'image': np.ndarray, 'instruction': str}\n"
                "You can optionally an `unnorm_key: str` to specific the dataset statistics you want to use for "
                "de-normalizing the output actions."
            )
            return "error"
    def get_averaged_values(self, indices):
        averaged_values = []
        for row_idx, idx in enumerate(indices):
            try:
                value1 = self.action_scale_list[row_idx][idx]
                value2 = self.action_scale_list[row_idx][idx + 1]
                average = (value1 + value2) / 2
            except: 
                print("index out of range")
                average = 1
            averaged_values.append(average)
        return averaged_values

    def run(self, host: str = "0.0.0.0", port: int = 8000) -> None:
        self.app = FastAPI()
        self.app.post("/act")(self.predict_action)
        uvicorn.run(self.app, host=host, port=port)


@dataclass
class DeployConfig:
    # fmt: off
    load_checkpoint: Union[str, Path] = ""               # HF Hub Path (or path to local run directory)

    # Server Configuration
    host: str = "0.0.0.0"                                               # Host IP Address
    port: int = 32820                                                    # Host Port
    vqgan_checkpoint: str = "/home/joel/seonghyeon/World-Model/checkpoints/lwm_checkpoints/vqgan"
    seed: int = 1234
    mesh_dim: str = "1,-1,1,1"
    dtype: str = "bf16"
    load_llama_config: str = "7b"
    update_llama_config: str = ""
    tokens_per_delta: int = 1
    tokens_per_action: int = 7
    vocab_file: str = "/home/joel/seonghyeon/World-Model/checkpoints/lwm_checkpoints/tokenizer.model"
    multi_image: int = 1
    jax_distributed: dict = JaxDistributedConfig.get_default_config()
    action_scale_file: str = ""


@draccus.wrap()
def deploy(cfg: DeployConfig) -> None:
    server = LWMServer(cfg.load_checkpoint, cfg.vqgan_checkpoint, cfg.seed, cfg.mesh_dim, cfg.dtype, cfg.load_llama_config, cfg.update_llama_config, cfg.tokens_per_delta, cfg.tokens_per_action, cfg.vocab_file, cfg.multi_image, cfg.jax_distributed, cfg.action_scale_file)
    server.run(cfg.host, port=cfg.port)


if __name__ == "__main__":
    deploy()

# python deploy.py --load_checkpoint checkpoints/latent_finetune_25traj_single_carrot2plate/streaming_params_200 --action_scale_file data/0809_carrot_to_plate_llava_256.csv --update_llama_config ""dict(action_vocab_size=256,delta_vocab_size=8,theta=50000000,max_sequence_length=2048,use_flash_attention=True,scan_attention=True,scan_query_chunk_size=512,scan_key_chunk_size=1024,remat_attention='nothing_saveable',scan_mlp=True,scan_mlp_chunk_size=8192,remat_mlp='nothing_saveable',remat_block='nothing_saveable',scan_layers=True)" --tokens_per_delta 4
# python lwm_deploy.py --load_checkpoint "checkpoints/latent_finetune_25traj_single_carrot2plate/streaming_params_200" --action_scale_file data/0809_carrot_to_plate_llava_256.csv --update_llama_config "dict(action_vocab_size=256,delta_vocab_size=8,theta=50000000,max_sequence_length=2048,use_flash_attention=True,scan_attention=True,scan_query_chunk_size=512,scan_key_chunk_size=1024,remat_attention='nothing_saveable',scan_mlp=True,scan_mlp_chunk_size=8192,remat_mlp='nothing_saveable',remat_block='nothing_saveable',scan_layers=True)" --tokens_per_delta 4



# CUDA_VISIBLE_DEVICES=0 python lwm_deploy.py --load_checkpoint "params::/home/joel/seonghyeon/World-Model/checkpoints/latent_finetune_25traj_single_carrot2plate/streaming_params_200" --action_scale_file /home/joel/seonghyeon/World-Model/data/0809_carrot_to_plate_llava_256.csv --update_llama_config "dict(action_vocab_size=256,delta_vocab_size=8,theta=50000000,max_sequence_length=2048,use_flash_attention=True,scan_attention=True,scan_query_chunk_size=512,scan_key_chunk_size=1024,remat_attention='nothing_saveable',scan_mlp=True,scan_mlp_chunk_size=8192,remat_mlp='nothing_saveable',remat_block='nothing_saveable',scan_layers=True)" --tokens_per_delta 4

# CUDA_VISIBLE_DEVICES=0 python lwm_deploy.py --load_checkpoint "params::/home/joel/seonghyeon/World-Model/checkpoints/latent_finetune_25traj_single_carrot2plate_5hz/streaming_params_200" --action_scale_file /home/joel/seonghyeon/World-Model/data/0809_carrot_to_plate_llava_5hz_256.csv --update_llama_config "dict(action_vocab_size=256,delta_vocab_size=8,theta=50000000,max_sequence_length=2048,use_flash_attention=True,scan_attention=True,scan_query_chunk_size=512,scan_key_chunk_size=1024,remat_attention='nothing_saveable',scan_mlp=True,scan_mlp_chunk_size=8192,remat_mlp='nothing_saveable',remat_block='nothing_saveable',scan_layers=True)" --tokens_per_delta 4