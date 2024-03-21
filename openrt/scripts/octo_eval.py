import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

from datetime import datetime
from functools import partial
import os
import time
from dataclasses import dataclass
from typing import Optional

from absl import app, flags, logging
import click
import cv2
import hydra
import imageio
import jax
import jax.numpy as jnp
import numpy as np

from asid.wrapper.asid_vec import make_env
from utils.experiment import (
    setup_wandb,
    hydra_to_dict,
    set_random_seed,
)
from octo.model.octo_model import OctoModel
from octo.utils.gym_wrappers import (
    HistoryWrapper,
    TemporalEnsembleWrapper,
    UnnormalizeActionProprio,
)


os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"


@hydra.main(config_path="../configs/", config_name="octo_eval_sim", version_base="1.1")
def run_experiment(cfg):

    env = make_env(
        robot_cfg_dict=hydra_to_dict(cfg.robot),
        env_cfg_dict=hydra_to_dict(cfg.env),
        seed=cfg.seed,
        device_id=0,
        verbose=False,
    )

    from robot.gymnasium_wrapper import OctoPreprocessingWrapper

    env = OctoPreprocessingWrapper(
        env,
        img_keys=["left_rgb" if env.unwrapped.sim else "215122255213_rgb"],
        proprio_keys=["lowdim_ee", "lowdim_qpos"],
    )

    # load models
    # model = OctoModel.load_pretrained("hf://rail-berkeley/octo-small")
    model = OctoModel.load_pretrained("logdir/finetune_pick_red_cube_synthetic_sl/octo_finetune/experiment_20240320_152140")

    # import ipdb; ipdb.set_trace()
    # wrap the robot environment
    env = UnnormalizeActionProprio(
        # env, model.dataset_statistics["bridge_dataset"], normalization_type="normal"
        env,
        model.dataset_statistics,
        normalization_type="normal",
    )
    env = HistoryWrapper(env, horizon=cfg.inference.horizon)
    env = TemporalEnsembleWrapper(env, pred_horizon=cfg.inference.pred_horizon)
    # switch TemporalEnsembleWrapper with RHCWrapper for receding horizon control
    # env = RHCWrapper(env, FLAGS.exec_horizon)

    # create policy function
    @jax.jit
    def sample_actions(
        pretrained_model: OctoModel,
        observations,
        tasks,
        rng,
    ):
        # add batch dim to observations
        observations = jax.tree_map(lambda x: x[None], observations)
        actions = pretrained_model.sample_actions(
            observations,
            tasks,
            rng=rng,
        )
        # remove batch dim
        return actions[0]

    def supply_rng(f, rng=jax.random.PRNGKey(0)):
        def wrapped(*args, **kwargs):
            nonlocal rng
            rng, key = jax.random.split(rng)
            return f(*args, rng=key, **kwargs)

        return wrapped

    policy_fn = supply_rng(
        partial(
            sample_actions,
            model,
        )
    )

    goal_image = jnp.zeros((256, 256, 3), dtype=np.uint8)
    goal_instruction = ""

    # goal sampling loop
    while True:
        modality = click.prompt(
            "Language or goal image?", type=click.Choice(["l", "g"])
        )

        if modality == "g":
            pass
            if click.confirm("Take a new goal?", default=True):
                goal_eep = cfg.inference.goal_eep
                env.go_to_eep(eep=goal_eep, gripper_pos=0.0)

                obs = env.get_observation(extended=True)
                eps = 1e-2
                while np.linalg.norm(obs["robot_state"]["joint_velocities"]) > eps:
                    pass

                input("Press [Enter] when ready for taking the goal image. ")

                goal = env.get_observation()
                goal["proprio"] = np.array(goal["proprio"])
                cv2.imwrite(f"prev_goal_img.jpeg", goal["image_primary"])
                goal["proprio"] = np.expand_dims(goal["proprio"], axis=0)
                goal["image_primary"] = np.expand_dims(goal["image_primary"], axis=0)
                # obs = wait_for_obs(widowx_client)
                # obs = convert_obs(obs, FLAGS.im_size)
                # goal = jax.tree_map(lambda x: x[None], obs)

            # Format task for the model
            task = model.create_tasks(goals={"image_primary": goal["image_primary"]})
            goal_image = goal["image_primary"][0]
            # For logging purposes
            goal_instruction = ""

        elif modality == "l":
            print("Current instruction: ", goal_instruction)
            if click.confirm("Take a new instruction?", default=True):
                text = input("Instruction?")
            # Format task for the model
            task = model.create_tasks(texts=[text])
            # For logging purposes
            goal_instruction = text
            goal_image = jnp.zeros_like(goal_image)
        else:
            raise NotImplementedError()

        input("Press [Enter] to start.")

        # reset env
        obs, _ = env.reset()
        time.sleep(2.0)

        # do rollout
        last_tstep = time.time()
        images = []
        goals = []
        t = 0
        while t < cfg.inference.num_timesteps:
            if time.time() > last_tstep + cfg.inference.step_duration:
                last_tstep = time.time()

                # save images
                images.append(obs["image_primary"][-1])
                goals.append(goal_image)

                if cfg.inference.show_image:
                    bgr_img = cv2.cvtColor(obs["image_primary"][-1], cv2.COLOR_RGB2BGR)
                    cv2.imshow("img_view", bgr_img)
                    cv2.waitKey(20)

                # get action
                forward_pass_time = time.time()
                # import ipdb; ipdb.set_trace()
                if cfg.inference.horizon > 1:
                    obs["pad_mask"] *= 0
                obs["pad_mask"] = obs["pad_mask"].astype(bool)
                obs["proprio"] = obs["proprio"].astype(np.float32)

                action = np.array(policy_fn(obs, task), dtype=np.float64)
                print("forward pass time: ", time.time() - forward_pass_time)

                # perform environment step
                start_time = time.time()
                # action = np.c_[action, np.zeros(action.shape[0])]
                # assert action.shape[1] == 8
                obs, _, _, truncated, _ = env.step(action)
                print("step time: ", time.time() - start_time)

                t += 1

                if truncated:
                    break

        # save video
        if cfg.inference.video_save_path is not None:
            os.makedirs(cfg.inference.video_save_path, exist_ok=True)
            curr_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            save_path = os.path.join(
                cfg.inference.video_save_path,
                f"{curr_time}.mp4",
            )
            video = np.concatenate([np.stack(goals), np.stack(images)], axis=1)
            imageio.mimsave(save_path, video, fps=1.0 / cfg.inference.step_duration * 3)


if __name__ == "__main__":
    run_experiment()
