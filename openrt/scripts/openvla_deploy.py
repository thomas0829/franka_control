import io
import os
import pickle
import time

import hydra
import imageio
import numpy as np
import requests
from PIL import Image

import json_numpy

json_numpy.patch()
import json
import logging
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Union
import sys

import draccus
import torch
import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor


#from utils.experiment import hydra_to_dict, set_random_seed, setup_wandb
import wandb
import random
import numpy as np
import omegaconf
import cv2

def hydra_to_dict(cfg):
    cfg_dict = omegaconf.OmegaConf.to_container(
        cfg, resolve=True, throw_on_missing=True
    )
    return cfg_dict

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    #torch.manual_seed(seed)

def setup_wandb(cfg, name, entity, project):

    run = wandb.init(
        name=name,
        entity=entity,
        project=project,
        config=omegaconf.OmegaConf.to_container(
            cfg, resolve=True, throw_on_missing=True
        ),
        settings=wandb.Settings(start_method="thread"),
    )

    return run

#from utils.logger import configure_logger
#from utils.system import get_device, set_gpu_mode

# === Utilities ===
SYSTEM_PROMPT = (
    "A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions."
)


def get_openvla_prompt(instruction: str, openvla_path: Union[str, Path]) -> str:
    if "v01" in openvla_path:
        return f"{SYSTEM_PROMPT} USER: What action should the robot take to {instruction.lower()}? ASSISTANT:"
    else:
        return f"In: What action should the robot take to {instruction.lower()}?\nOut:"
    

class OpenVLAServer:
    def __init__(self, openvla_path: Union[str, Path], attn_implementation: Optional[str] = "flash_attention_2") -> Path:
        """
        A simple server for OpenVLA models; exposes `/act` to predict an action for a given image + instruction.
            => Takes in {"image": np.ndarray, "instruction": str, "unnorm_key": Optional[str]}
            => Returns  {"action": np.ndarray}
        """
        self.openvla_path, self.attn_implementation = openvla_path, attn_implementation
        self.device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

        # Load VLA Model using HF AutoClasses
        self.processor = AutoProcessor.from_pretrained(self.openvla_path, trust_remote_code=True)
        self.vla = AutoModelForVision2Seq.from_pretrained(
            self.openvla_path,
            attn_implementation=attn_implementation,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        ).to(self.device)

        # [Hacky] Load Dataset Statistics from Disk (if passing a path to a fine-tuned model)
        if os.path.isdir(self.openvla_path):
            with open(Path(self.openvla_path) / "dataset_statistics.json", "r") as f:
                self.vla.norm_stats = json.load(f)

    def predict_action(self, payload: Dict[str, Any]) -> str:
        try:
            if double_encode := "encoded" in payload:
                # Support cases where `json_numpy` is hard to install, and numpy arrays are "double-encoded" as strings
                assert len(payload.keys() == 1), "Only uses encoded payload!"
                payload = json.loads(payload["encoded"])

            # Parse payload components
            image, instruction = payload["image"], payload["instruction"]
            unnorm_key = payload.get("unnorm_key", None)

            # Run VLA Inference
            prompt = get_openvla_prompt(instruction, self.openvla_path)
            image = np.array(image)
            print(image.shape)
            print(image)
            #print(image.shape, image)
            inputs = self.processor(prompt, Image.fromarray(image).convert("RGB")).to(self.device, dtype=torch.bfloat16)
            action = self.vla.predict_action(**inputs, unnorm_key=unnorm_key, do_sample=False)

            if double_encode:
                return json_numpy.dumps(action)
            else:
                return action
        except:  # noqa: E722
            logging.error(traceback.format_exc())
            logging.warning(
                "Your request threw an error; make sure your request complies with the expected format:\n"
                "{'image': np.ndarray, 'instruction': str}\n"
                "You can optionally an `unnorm_key: str` to specific the dataset statistics you want to use for "
                "de-normalizing the output actions."
            )
            return "error"
    def run(self, host: str = "0.0.0.0", port: int = 8000) -> None:
        self.app = FastAPI()
        self.app.post("/act")(self.predict_action)
        uvicorn.run(self.app, host=host, port=port)
        


@dataclass
class DeployConfig:
    # fmt: off
    openvla_path: Union[str, Path] = "openvla/openvla-7b"               # HF Hub Path (or path to local run directory)

    # Server Configuration
    host: str = "0.0.0.0"                                               # Host IP Address
    port: int = 8000                                                    # Host Port

    # fmt: on


@draccus.wrap()
def deploy(cfg: DeployConfig) -> None:
    server = OpenVLAServer(cfg.openvla_path)
    server.run(cfg.host, port=cfg.port)


if __name__ == "__main__":
    deploy()