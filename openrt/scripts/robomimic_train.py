"""
The main entry point for training policies.

Args:
    config (str): path to a config json that will be used to override the default settings.
        If omitted, default settings are used. This is the preferred way to run experiments.

    algo (str): name of the algorithm to run. Only needs to be provided if @config is not
        provided.

    name (str): if provided, override the experiment name defined in the config

    dataset (str): if provided, override the dataset path defined in the config

    debug (bool): set this flag to run a quick training run for debugging purposes    
"""

import argparse
import json
import numpy as np
import time
import os
import shutil
import psutil
import sys
import signal
import socket
import traceback
import shlex
import subprocess
import tempfile
from copy import deepcopy

from collections import OrderedDict
from datetime import datetime

import robomimic

import torch
from torch.utils.data import DataLoader

import robomimic
import robomimic.utils.train_utils as TrainUtils
import robomimic.utils.torch_utils as TorchUtils
import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.lang_utils as LangUtils
import robomimic.macros as Macros
from robomimic.config import config_factory
from robomimic.algo import algo_factory, RolloutPolicy, HTAMPRolloutPolicy, MotionPlannerRolloutPolicy
from robomimic.utils.log_utils import PrintLogger, DataLogger, flush_warnings, log_warning


def run_rollouts_in_subprocess(
    model,
    config,
    log_dir,
    env_meta,
    shape_meta,
    obs_normalization_stats,
    action_normalization_stats,
    mp_action_normalization_stats,
    task_spec,
    cached_motion_planner_targets,
    epoch,
    video_dir=None,
):
    """
    Policy evaluations for OV need to run in a subprocess because if we keep the environment running too
    long, the process can run out of memory or segfault due to rendering issues.
    """
    ensure_tamp_success = (config.experiment.rollout.htamp_ensure_tamp_success if config.experiment.rollout.mode == "tamp_gated" else False)
    assert not ensure_tamp_success, "TODO: support this"

    # first move model off of GPU and free unused memory
    model.nets = model.nets.to("cpu")
    torch.cuda.empty_cache()

    rollout_log_dir = os.path.join(log_dir, "rollout_logs")
    intermediate_video_log_dir = os.path.join(log_dir, "..", "intermediate_rollout_videos")
    os.makedirs(rollout_log_dir, exist_ok=True)
    os.makedirs(intermediate_video_log_dir, exist_ok=True)

    # collect complete set of environments we need to do rollout evals on, along with any specific horizons and kwargs
    all_env_names = [env_meta["env_name"]]
    all_env_horizons = [config.experiment.rollout.horizon]
    all_env_kwargs = [None]
    all_env_langs = [env_meta["env_lang"]]
    if config.experiment.additional_envs is not None:
        for name in config.experiment.additional_envs:
            all_env_names.append(name)

        if config.experiment.additional_horizons is not None:
            assert len(config.experiment.additional_horizons) == len(config.experiment.additional_envs), "mismatch in additional_horizons"
            all_env_horizons += list(config.experiment.additional_horizons)
        else:
            all_env_horizons += [config.experiment.rollout.horizon] * len(config.experiment.additional_envs)
    
        if config.experiment.additional_envs_kwargs is not None:
            assert len(config.experiment.additional_envs_kwargs) == len(config.experiment.additional_envs)
            all_env_kwargs += list(config.experiment.additional_envs_kwargs)
        else:
            all_env_kwargs += [None] * len(config.experiment.additional_envs)

        if config.experiment.additional_envs_langs is not None:
            assert len(config.experiment.additional_envs_langs) == len(config.experiment.additional_envs)
            all_env_langs += list(config.experiment.additional_envs_langs)
        else:
            all_env_langs += [None] * len(config.experiment.additional_envs)
    assert len(all_env_names) == len(all_env_kwargs)
    assert len(all_env_names) == len(all_env_langs)

    # final results will be collected into the structs below
    all_rollout_logs = dict()
    video_paths = dict()

    with tempfile.TemporaryDirectory(dir=log_dir) as td:
        # save latest model in temporary file
        tmp_model_path = os.path.join(td, "tmp.pth")
        tmp_json_path = os.path.join(td, "tmp.json")
        tmp_error_path = os.path.join(td, "error.txt")
        TrainUtils.save_model(
            model=model,
            config=config,
            env_meta=env_meta,
            shape_meta=shape_meta,
            ckpt_path=tmp_model_path,
            obs_normalization_stats=obs_normalization_stats,
            action_normalization_stats=action_normalization_stats,
            mp_action_normalization_stats=mp_action_normalization_stats,
            task_spec=task_spec,
            cached_motion_planner_targets=cached_motion_planner_targets,
        )
        shutil.copyfile(tmp_model_path, "/tmp/tmp.pth")

        for env_id in range(len(all_env_names)):
            cur_env_name = all_env_names[env_id]
            cur_env_horizon = all_env_horizons[env_id]
            cur_env_kwargs = all_env_kwargs[env_id]
            cur_env_lang = all_env_langs[env_id]

            # NOTE: we will keep trying to execute the rollouts, since isaac sim might segfault or timeout
            max_retries = 10 # this many retries allowed without completing any successful rollouts
            num_retries = 0
            num_attempts_total = 0
            num_rollouts_target = config.experiment.rollout.n
            num_rollouts_completed = 0
            all_rollout_stats = []
            while True:

                # some paths
                rollout_log_path = os.path.join(rollout_log_dir, "rollout_{}_epoch_{}_attempt_{}.txt".format(cur_env_name, epoch, num_attempts_total))
                rollout_json_path = os.path.join(rollout_log_dir, "rollout_{}_epoch_{}_attempt_{}.json".format(cur_env_name, epoch, num_attempts_total))
                rollout_error_path = os.path.join(rollout_log_dir, "rollout_{}_error_epoch_{}_attempt_{}.txt".format(cur_env_name, epoch, num_attempts_total))

                # setup run_trained_agent.py in subprocess to evaluate the policy
                path_to_eval_script = os.path.join(robomimic.__path__[0], "scripts/run_trained_agent.py")
                # ep_timeout = 15 # max 15 minutes per rollout
                num_rollouts_for_command = num_rollouts_target - num_rollouts_completed
                path_to_python = "/isaac-sim/python.sh" if os.path.exists("/isaac-sim/python.sh") else "python"
                cmd = "{} {} --agent {} --n_rollouts {} --json_path {} --error_path {}".format(
                    path_to_python, path_to_eval_script, tmp_model_path, num_rollouts_for_command, rollout_json_path, rollout_error_path,
                )
                # env-specific args
                cmd += " --env {} --horizon {}".format(cur_env_name, cur_env_horizon)
                if cur_env_kwargs is not None:
                    # need string quotes for parser to take dictionary as single string input
                    cmd += " --env_kwargs \"{}\"".format(cur_env_kwargs)
                if LangUtils.LANG_COND_ENABLED:
                    assert cur_env_lang is not None
                    cmd += " --env_lang \"{}\"".format(cur_env_lang)
                if config.experiment.render_video:
                    assert video_dir is not None
                    video_str = "_epoch_{}.mp4".format(epoch)
                    video_path = os.path.join(video_dir, "{}{}".format(cur_env_name, video_str))
                    cmd += " --video_path {} --video_skip {}".format(video_path, config.experiment.get("video_skip", 5))

                # add suffix to redirect log into a file
                cmd += " > {} 2>&1".format(rollout_log_path)
                
                # execute
                print("")
                print("{} rollout attempt {}".format(cur_env_name, num_attempts_total))
                print("{} rollout retry attempt {}".format(cur_env_name, num_retries))
                print("launching subprocess for policy evaluation with command:")
                print(cmd)
                print("")

                # (see https://stackoverflow.com/questions/4789837/how-to-terminate-a-python-subprocess-launched-with-shell-true/4791612#4791612)
                proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, preexec_fn=os.setsid)
                # proc.wait()

                # monitor the subprocess here
                check_interval = 60 # check every N seconds for process termination
                heartbeat_interval = 300 # time (in seconds) between print statements in main loop to ensure it is still alive
                ep_timeout = 15 # max 15 minutes per rollout - we will monitor the rollout_json_path file since it is updated after each rollout
                heartbeat_timestamp = time.time()
                last_ep_time = time.time()
                last_file_timestamp = None
                process_timed_out = False
                while True:

                    if proc.poll() is not None:
                        # process terminated
                        break

                    # check if we finished a new eval
                    if os.path.exists(rollout_json_path):
                        file_timestamp = os.path.getmtime(rollout_json_path)
                        if (last_file_timestamp is None) or (file_timestamp != last_file_timestamp):
                            # finished new episode - update times
                            last_ep_time = time.time()
                            last_file_timestamp = file_timestamp

                    # check for timeout (an eval took too long)
                    if time.time() - last_ep_time > (60. * ep_timeout):
                        # process timed out
                        process_timed_out = True
                        # proc.kill()
                        try:
                            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)  # Send the signal to all the process groups
                        except ProcessLookupError:
                            pass
                        break

                    # maybe print a heartbeat message
                    if time.time() - heartbeat_timestamp > heartbeat_interval:
                        current_datetime = datetime.now()
                        formatted_datetime = current_datetime.strftime("%B %d, %Y %H:%M:%S")
                        print("run_rollouts_in_subprocess heartbeat\n  {}\n  {}\n  Num Rollouts Current {} out of Target {} with Num Attempts {} and Num Retry Attempts {}\n".format(formatted_datetime, cur_env_name, num_rollouts_completed, num_rollouts_target, num_attempts_total, num_retries))
                        heartbeat_timestamp = time.time()

                    # sleep until next check interval
                    time.sleep(check_interval)

                # figure out number of rollouts done by subproc
                num_rollouts_completed_by_proc = 0
                if os.path.exists(rollout_json_path):
                    with open(rollout_json_path) as f_json:
                        rollout_stats = json.load(f_json)["stats"]
                    all_rollout_stats += rollout_stats
                    num_rollouts_completed_by_proc = len(rollout_stats)

                # handle error here (did not reach target number of rollouts)
                if num_rollouts_completed_by_proc < num_rollouts_for_command:
                    fail_reason = "timeout" if process_timed_out else "crash"
                    print("Got failed rollout attempt on env name {}! Reason: {}. Num rollouts completed {} out of {} desired.".format(cur_env_name, fail_reason, num_rollouts_completed_by_proc, num_rollouts_for_command))
                    # notify_on_slack(config=config, res_str="Subproc Error (Reason: {}) on Env Name {}, Rollout Epoch {} and Attempt {}, Retry Attempt {}".format(fail_reason, cur_env_name, epoch, num_attempts_total, num_retries), important_stats=None)

                # maybe copy video over to another directory
                if config.experiment.render_video and os.path.exists(video_path):
                    video_name = os.path.basename(video_path)[:-4] + "_attempt_{}.mp4".format(num_attempts_total)
                    shutil.copyfile(
                        video_path,
                        os.path.join(intermediate_video_log_dir, video_name),
                    )

                # update rollout count and break if necessary
                num_rollouts_completed += num_rollouts_completed_by_proc
                num_attempts_total += 1
                if num_rollouts_completed == 0:
                    num_retries += 1
                if num_retries >= max_retries:
                    raise Exception("Rollout evaluation for epoch {} and env {} failed after reaching max number of retries without a single rollout completion ({})".format(epoch, cur_env_name, max_retries))
                if num_rollouts_completed >= num_rollouts_target:
                    break

            # parse results from rollout logs and save to variables we will return
            from robomimic.scripts.run_trained_agent import average_rollout_logs
            avg_rollout_stats = average_rollout_logs(all_rollout_stats)
            all_rollout_logs[cur_env_name] = avg_rollout_stats
            video_paths[cur_env_name] = video_path

    # move model back to GPU
    model.nets = model.nets.to(model.device)

    return all_rollout_logs, video_paths


def notify_on_slack(config, res_str, important_stats=None):
    """maybe give slack notification"""
    if Macros.SLACK_TOKEN is not None:
        from robomimic.scripts.give_slack_notification import give_slack_notif
        msg = "Completed the following training run!\nHostname: {}\nExperiment Name: {}\n".format(socket.gethostname(), config.experiment.name)
        msg += "```{}```".format(res_str)
        if important_stats is not None:
            msg += "\nRollout Success Rate Stats"
            msg += "\n```{}```".format(important_stats)
        give_slack_notif(msg)


def train(config, device, auto_remove_exp=False, resume=False):
    """
    Train a model using the algorithm.
    """

    # time this run
    start_time = time.time()
    time_elapsed = 0.

    # first set seeds
    np.random.seed(config.train.seed)
    torch.manual_seed(config.train.seed)

    # set num workers
    torch.set_num_threads(1)

    print("\n============= New Training Run with Config =============")
    print(config)
    print("")
    log_dir, ckpt_dir, video_dir, time_dir = TrainUtils.get_exp_dir(config, auto_remove_exp_dir=auto_remove_exp, resume=resume)

    # path for latest model and backup (to support @resume functionality)
    latest_model_path = os.path.join(time_dir, "last.pth")
    latest_model_backup_path = os.path.join(time_dir, "last_bak.pth")

    if config.experiment.logging.terminal_output_to_txt:
        # log stdout and stderr to a text file
        logger = PrintLogger(os.path.join(log_dir, 'log.txt'))
        sys.stdout = logger
        sys.stderr = logger

    # hack for language conditioning - add low-dim modality for language embedding
    if config.observation.language_conditioned:
        # assert LangUtils.LANG_OBS_KEY not in config.observation.modalities.obs.low_dim
        # assert LangUtils.LANG_OBS_KEY not in config.observation.modalities.obs.lang
        LangUtils.LANG_COND_ENABLED = True
        if LangUtils.LANG_OBS_KEY not in config.observation.modalities.obs.lang:
            with config.values_unlocked():
                # config.observation.modalities.obs.low_dim.append(LangUtils.LANG_OBS_KEY)
                config.observation.modalities.obs.lang.append(LangUtils.LANG_OBS_KEY)

    # read config to set up metadata for observation modalities (e.g. detecting rgb observations)
    ObsUtils.initialize_obs_utils_with_config(config)

    # make sure the dataset exists

    dataset_paths_to_check = [config.train.data]
    if isinstance(config.train.data, str):
        dataset_paths_to_check = [os.path.expandvars(os.path.expanduser(config.train.data))]
    else:
        dataset_paths_to_check = [os.path.expandvars(os.path.expanduser(ds_cfg["path"])) for ds_cfg in config.train.data]
    for dataset_path in dataset_paths_to_check:
        if not os.path.exists(dataset_path):
            raise Exception("Dataset at provided path {} not found!".format(dataset_path))

    # NOTE: for now, env_meta, action_keys, and shape_meta are all inferred from the first dataset if multiple datasets are used for training.
    if len(dataset_paths_to_check) > 1:
        log_warning("Env meta and shape meta will be inferred from first dataset at path {}".format(dataset_paths_to_check[0]))

    # assert len(dataset_paths_to_check) == 1
    dataset_path = dataset_paths_to_check[0]

    # load basic metadata from training file
    print("\n============= Loaded Environment Metadata =============")
    if config.experiment.rollout.enabled and (config.experiment.rollout.mode == "motion_planner"):
        assert isinstance(config.train.data, str)
        # motion planner rollouts need MG version of environment
        import mimicgen.utils.file_utils as MG_FileUtils
        env_meta = MG_FileUtils.get_env_metadata_from_dataset(dataset_path=dataset_path)
    else:
        env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path=dataset_path)

    # update env meta if applicable
    TensorUtils.deep_update(env_meta, config.experiment.env_meta_update_dict)

    action_keys, _ = FileUtils.get_action_info_from_config(config)
    shape_meta = FileUtils.get_shape_metadata_from_dataset(
        dataset_path=dataset_path,
        action_keys=action_keys,
        all_obs_keys=config.all_obs_keys,
        language_conditioned=config.observation.language_conditioned,
        verbose=True,
    )

    if config.experiment.env is not None:
        env_meta["env_name"] = config.experiment.env
        print("=" * 30 + "\n" + "Replacing Env to {}\n".format(env_meta["env_name"]) + "=" * 30)
    env_meta["env_lang"] = config.experiment.env_lang

    if config.experiment.rollout.mode == "motion_planner":
        # read information from hdf5 needed for motion planner rollouts
        mg_task_spec, mg_cached_motion_planner_targets = FileUtils.get_motion_planner_metadata_from_dataset(dataset_path=dataset_path)

    # create environments
    envs = OrderedDict()
    env_rollout_horizons = OrderedDict()
    is_simpler_ov_env = EnvUtils.is_simpler_ov_env(env_meta=env_meta)
    should_run_rollouts_in_subprocess = config.experiment.rollout.get("use_subproc_eval", False) or is_simpler_ov_env
    if config.experiment.rollout.enabled and (not should_run_rollouts_in_subprocess):
        # create environments for validation runs
        env_names = [env_meta["env_name"]]
        env_horizons = [config.experiment.rollout.horizon]
        env_additional_kwargs = [None]
        env_langs = [env_meta["env_lang"]]

        if config.experiment.additional_envs is not None:
            for name in config.experiment.additional_envs:
                env_names.append(name)

            if config.experiment.additional_horizons is not None:
                assert len(config.experiment.additional_horizons) == len(config.experiment.additional_envs), "mismatch in additional_horizons"
                env_horizons += list(config.experiment.additional_horizons)
            else:
                env_horizons += [config.experiment.rollout.horizon] * len(config.experiment.additional_envs)

            if config.experiment.additional_envs_kwargs is not None:
                assert len(config.experiment.additional_envs_kwargs) == len(config.experiment.additional_envs)
                env_additional_kwargs += list(config.experiment.additional_envs_kwargs)
            else:
                env_additional_kwargs += [None] * len(config.experiment.additional_envs)

            if config.experiment.additional_envs_langs is not None:
                assert len(config.experiment.additional_envs_langs) == len(config.experiment.additional_envs)
                env_langs += list(config.experiment.additional_envs_langs)
            else:
                env_langs += [None] * len(config.experiment.additional_envs)
        assert len(env_names) == len(env_additional_kwargs)
        assert len(env_names) == len(env_langs)

        for env_id in range(len(env_names)):

            # maybe incorporate any additional kwargs for this specific env
            env_meta_for_this_env = deepcopy(env_meta)
            if env_additional_kwargs[env_id] is not None:
                TensorUtils.deep_update(env_meta_for_this_env["env_kwargs"], env_additional_kwargs[env_id])

            # language conditioning
            if LangUtils.LANG_COND_ENABLED:
                assert env_langs[env_id] is not None
            if env_langs[env_id] is not None:
                env_meta_for_this_env["env_lang"] = env_langs[env_id]

            env_name_for_this_env = env_names[env_id]
            env_horizon_for_this_env = env_horizons[env_id]
            env = EnvUtils.create_env_from_metadata(
                env_meta=env_meta_for_this_env,
                env_name=env_name_for_this_env, 
                render=config.experiment.render, 
                render_offscreen=config.experiment.render_video,
                use_image_obs=shape_meta["use_images"], 
                use_depth_obs=shape_meta["use_depths"], 
            )
            env = EnvUtils.wrap_env_from_config(env, config=config) # apply environment wrapper, if applicable
            assert env.name not in envs, "Got duplicate env with name {}".format(env.name)
            envs[env.name] = env
            env_rollout_horizons[env.name] = env_horizon_for_this_env
            print(envs[env.name])

        # maybe wrap environment(s) with object-centric wrapper here
        if config.experiment.rollout.htamp_object_centric.enabled:
            assert config.experiment.rollout.mode in ["tamp_gated", "motion_planner"]

            import mimicgen
            from mimicgen.envs.robosuite.obj_centric import FrameCentricWrapper, get_padding_sizes_for_env

            for env_name in envs:
                init_kwargs = config.experiment.rollout.htamp_object_centric.init_kwargs
                init_kwargs = dict(init_kwargs) if init_kwargs is not None else dict()
                assert "frame_centric_observable_names" not in init_kwargs
                assert "padding_sizes" not in init_kwargs

                frame_centric_observable_names = config.experiment.rollout.htamp_object_centric.observable_names
                variants = config.experiment.rollout.htamp_object_centric.obs_variant
                include_grasps = (config.experiment.rollout.mode == "motion_planner") or config.experiment.rollout.htamp_grasp_conditions

                if variants is not None:
                    assert len(frame_centric_observable_names) == len(variants)
                else:
                    assert (frame_centric_observable_names is None) or (len(frame_centric_observable_names) == 1 and frame_centric_observable_names[0] == "frame_centric_observable")

                # determine if padding is required (due to different sizes for the frame-centric observables)
                padding_sizes = get_padding_sizes_for_env(env=env.base_env, env_name=env_name, variants=variants, include_grasps=include_grasps)

                envs[env_name].env = FrameCentricWrapper(
                    envs[env_name].env,
                    frame_centric_observable_names=frame_centric_observable_names,
                    padding_sizes=padding_sizes,
                    **init_kwargs,
                )

        if config.experiment.rollout.mode == "tamp_gated":

            # HACK: set some temp dir paths based on PID to avoid race conditions
            from pybullet_planning.pybullet_tools import utils as utils1  # isort:skip
            from pddlstream.examples.pybullet.utils.pybullet_tools import utils as utils2  # isort:skip
            from pddlstream.pddlstream.algorithms import downward  # isort:skip

            utils1.TEMP_DIR = "/tmp/temp-{}/".format(os.getpid())
            utils2.TEMP_DIR = "/tmp/temp-{}/".format(os.getpid())
            downward.TEMP_DIR = "/tmp/temp-{}/".format(os.getpid())
            utils1.VHACD_DIR = "/tmp/vhacd-{}/".format(os.getpid())

            # make htamp objects for tamp planning
            htamp_policies = OrderedDict()
            joint_controller_configs = OrderedDict()
            osc_controller_configs = OrderedDict()
            for env_name in envs:
                htamp_env = envs[env_name]

                joint_controller_configs[env_name] = None
                osc_controller_configs[env_name] = None
                if config.experiment.rollout.htamp_use_joint_actions:
                    # NOTE: we need to swap to joint position controller before constructing hitl-tamp object
                    from robosuite.controllers import load_controller_config
                    joint_controller_configs[env_name] = load_controller_config(default_controller="JOINT_POSITION")
                    # # use absolute joint actions with higher kp for better tracking performance
                    # joint_controller_configs[env_name]["kp"] = 300.
                    # joint_controller_configs[env_name]["control_delta"] = False
                    osc_controller_configs[env_name] = htamp_env.base_env.switch_controllers(joint_controller_configs[env_name])

                from htamp.hitl_tamp import HitlTAMP
                htamp_policies[env_name] = HitlTAMP(
                    wrapper=htamp_env.unwrapped if hasattr(htamp_env, "unwrapped") else htamp_env,
                    tasks=None,
                    osc=(not config.experiment.rollout.htamp_use_joint_actions),
                    # backoff=(not config.experiment.rollout.htamp_use_joint_actions),
                    grasp_conditions=config.experiment.rollout.htamp_grasp_conditions,
                    show_planner_gui=False,
                    dataset_path=os.path.expandvars(os.path.expanduser(config.experiment.rollout.htamp_constraints_path)),
                    noise_level=config.experiment.rollout.htamp_noise_level,
                )
                htamp_policies[env_name].setup()

        elif config.experiment.rollout.mode == "motion_planner":

            # HACK: set some temp dir paths based on PID to avoid race conditions
            from pybullet_planning.pybullet_tools import utils as utils1  # isort:skip
            from pddlstream.examples.pybullet.utils.pybullet_tools import utils as utils2  # isort:skip
            from pddlstream.pddlstream.algorithms import downward  # isort:skip

            import htamp
            from htamp.scripts.test_motion_planner import MotionPlanner

            # create motion planner objects for rollouts with motion planner
            motion_planners = OrderedDict()
            joint_controller_configs = OrderedDict()
            osc_controller_configs = OrderedDict()
            for env_name in envs:
                this_env = envs[env_name]

                joint_controller_configs[env_name] = None
                osc_controller_configs[env_name] = None
                from robosuite.controllers import load_controller_config
                joint_controller_configs[env_name] = load_controller_config(default_controller="JOINT_POSITION")
                # # use absolute joint actions with higher kp for better tracking performance
                # joint_controller_configs[env_name]["kp"] = 300.
                # joint_controller_configs[env_name]["control_delta"] = False
                osc_controller_configs[env_name] = this_env.base_env.switch_controllers(joint_controller_configs[env_name])
                this_env.base_env.switch_controllers(osc_controller_configs[env_name])

                motion_planners[env_name] = MotionPlanner(
                    env=this_env.unwrapped if hasattr(this_env, "unwrapped") else this_env,
                    use_gui=config.experiment.render,
                    collision_free=False,
                    use_osc=False,
                    plan_to_contact=config.experiment.rollout.mp.use_mp_contact,
                )

    print("")

    # setup for a new training run
    data_logger = DataLogger(
        log_dir,
        config,
        log_tb=config.experiment.logging.log_tb,
        log_wandb=config.experiment.logging.log_wandb,
    )
    model = algo_factory(
        algo_name=config.algo_name,
        config=config,
        obs_key_shapes=shape_meta["all_shapes"],
        ac_dim=shape_meta["ac_dim"],
        device=device,
    )

    if resume:
        # load ckpt dict
        print("*" * 50)
        print("resuming from ckpt at {}".format(latest_model_path))
        try:
            ckpt_dict = FileUtils.load_dict_from_checkpoint(ckpt_path=latest_model_path)
        except Exception as e:
            print("got error: {} when loading from {}".format(e, latest_model_path))
            print("trying backup path {}".format(latest_model_backup_path))
            ckpt_dict = FileUtils.load_dict_from_checkpoint(ckpt_path=latest_model_backup_path)

        # load model weights and optimizer state
        model.deserialize(ckpt_dict["model"])
        print("*" * 50)
    
    # save the config as a json file
    with open(os.path.join(log_dir, '..', 'config.json'), 'w') as outfile:
        json.dump(config, outfile, indent=4)

    print("\n============= Model Summary =============")
    print(model)  # print model summary
    print("")

    # load training data
    trainset, validset = TrainUtils.load_data_for_training(
        config, obs_keys=shape_meta["all_obs_keys"])
    train_sampler = { k: trainset[k].get_dataset_sampler() for k in trainset }
    print("\n============= Training Datasets =============")
    for k in trainset:
        print("Dataset Key: {}".format(k))
        print(trainset[k])
    print("")
    if validset is not None:
        print("\n============= Validation Dataset =============")
        print(validset)
        print("")

    # maybe retreve statistics for normalization
    obs_normalization_stats = None
    if config.train.hdf5_normalize_obs:
        obs_normalization_stats = trainset["data"].get_obs_normalization_stats()
    action_normalization_stats = trainset["data"].get_action_normalization_stats()
    mp_action_normalization_stats = None
    if "classifier" in trainset:
        # TODO: remove this hack and clean up
        mp_action_normalization_stats = trainset["classifier"].get_action_normalization_stats()

    # initialize data loaders
    train_loader = {
        k: DataLoader(
            dataset=trainset[k],
            sampler=train_sampler[k],
            batch_size=config.train.batch_size,
            shuffle=(train_sampler[k] is None),
            num_workers=config.train.num_data_workers,
            drop_last=True
        ) for k in trainset
    }

    if config.experiment.validate:
        # cap num workers for validation dataset at 1
        num_workers = min(config.train.num_data_workers, 1)
        valid_sampler = {k: validset[k].get_dataset_sampler() for k in validset}
        valid_loader = {
            k: DataLoader(
                dataset=validset[k],
                sampler=valid_sampler[k],
                batch_size=config.train.batch_size,
                shuffle=(valid_sampler[k] is None),
                num_workers=num_workers,
                drop_last=True
            ) for k in validset
        }
    else:
        valid_loader = None

    # print all warnings before training begins
    print("*" * 50)
    print("Warnings generated by robomimic have been duplicated here (from above) for convenience. Please check them carefully.")
    flush_warnings()
    print("*" * 50)
    print("")

    # main training loop
    best_valid_loss = None
    best_return = None
    best_success_rate = None
    best_success_rate_epoch = None
    best_avg_task_success_rate = None
    best_avg_task_success_rate_epoch = None
    best_tamp_failure_info = None
    all_tamp_failure_info = None
    if config.experiment.rollout.enabled:
        best_return = dict()
        best_success_rate = dict()
        best_success_rate_epoch = dict()
        best_tamp_failure_info = dict()
        all_tamp_failure_info = dict()
    last_saved_epoch_with_rollouts = None
    last_ckpt_time = time.time()

    start_epoch = 1 # epoch numbers start at 1
    if resume:
        # load variable state needed for train loop
        variable_state = ckpt_dict["variable_state"]
        start_epoch = variable_state["epoch"] + 1 # start at next epoch, since this recorded the last epoch of training completed
        best_valid_loss = variable_state["best_valid_loss"]
        best_return = variable_state["best_return"]
        best_success_rate = variable_state["best_success_rate"]
        best_success_rate_epoch = variable_state["best_success_rate_epoch"]
        best_avg_task_success_rate = variable_state["best_avg_task_success_rate"]
        best_avg_task_success_rate_epoch = variable_state["best_avg_task_success_rate_epoch"]
        best_tamp_failure_info = variable_state["best_tamp_failure_info"]
        all_tamp_failure_info = variable_state["all_tamp_failure_info"]
        last_saved_epoch_with_rollouts = variable_state["last_saved_epoch_with_rollouts"]
        time_elapsed = variable_state["time_elapsed"]
        print("*" * 50)
        print("resuming training from epoch {}".format(start_epoch))
        print("*" * 50)

    need_sync_results = (Macros.RESULTS_SYNC_PATH_ABS is not None)
    if need_sync_results:
        # these paths will be updated after each evaluation
        best_ckpt_paths_synced = None
        best_video_paths_synced = None
        last_ckpt_paths_synced = None
        last_video_paths_synced = None
        log_dir_path_synced = os.path.join(Macros.RESULTS_SYNC_PATH_ABS, "logs")
        if resume:
            # load variable state
            best_ckpt_paths_synced = variable_state["best_ckpt_paths_synced"]
            best_video_paths_synced = variable_state["best_video_paths_synced"]
            last_ckpt_paths_synced = variable_state["last_ckpt_paths_synced"]
            last_video_paths_synced = variable_state["last_video_paths_synced"]

    # number of learning steps per epoch (defaults to a full dataset pass)
    train_num_steps = config.experiment.epoch_every_n_steps
    valid_num_steps = config.experiment.validation_epoch_every_n_steps

    for epoch in range(start_epoch, config.train.num_epochs + 1):
        step_log = TrainUtils.run_epoch(
            model=model,
            data_loader=train_loader,
            epoch=epoch,
            num_steps=train_num_steps,
            obs_normalization_stats=obs_normalization_stats,
        )
        model.on_epoch_end(epoch)

        # setup checkpoint path
        epoch_ckpt_name = "model_epoch_{}".format(epoch)

        # check for recurring checkpoint saving conditions
        should_save_ckpt = False
        if config.experiment.save.enabled:
            time_check = (config.experiment.save.every_n_seconds is not None) and \
                (time.time() - last_ckpt_time > config.experiment.save.every_n_seconds)
            epoch_check = (config.experiment.save.every_n_epochs is not None) and \
                (epoch > 0) and (epoch % config.experiment.save.every_n_epochs == 0)
            epoch_list_check = (epoch in config.experiment.save.epochs)
            should_save_ckpt = (time_check or epoch_check or epoch_list_check)
        ckpt_reason = None
        if should_save_ckpt:
            last_ckpt_time = time.time()
            ckpt_reason = "time"

        print("Train Epoch {}".format(epoch))
        print(json.dumps(step_log, sort_keys=True, indent=4))
        for k, v in step_log.items():
            if k.startswith("Time_"):
                data_logger.record("Timing_Stats/Train_{}".format(k[5:]), v, epoch)
            else:
                data_logger.record("Train/{}".format(k), v, epoch)

        # Evaluate the model on validation set
        if config.experiment.validate:
            with torch.no_grad():
                step_log = TrainUtils.run_epoch(
                    model=model,
                    data_loader=valid_loader,
                    epoch=epoch,
                    validate=True,
                    num_steps=valid_num_steps,
                    obs_normalization_stats=obs_normalization_stats,
                )
            for k, v in step_log.items():
                if k.startswith("Time_"):
                    data_logger.record("Timing_Stats/Valid_{}".format(k[5:]), v, epoch)
                else:
                    data_logger.record("Valid/{}".format(k), v, epoch)

            print("Validation Epoch {}".format(epoch))
            print(json.dumps(step_log, sort_keys=True, indent=4))

            # save checkpoint if achieve new best validation loss
            valid_check = "Loss" in step_log
            if valid_check and (best_valid_loss is None or (step_log["Loss"] <= best_valid_loss)):
                best_valid_loss = step_log["Loss"]
                if config.experiment.save.enabled and config.experiment.save.on_best_validation:
                    epoch_ckpt_name += "_best_validation_{}".format(best_valid_loss)
                    should_save_ckpt = True
                    ckpt_reason = "valid" if ckpt_reason is None else ckpt_reason

        # Evaluate the model by by running rollouts

        # do rollouts at fixed rate or if it's time to save a new ckpt
        video_paths = None
        rollout_check = (epoch % config.experiment.rollout.rate == 0) or (should_save_ckpt and ckpt_reason == "time")
        did_rollouts = False
        if config.experiment.rollout.enabled and (epoch > config.experiment.rollout.warmstart) and rollout_check:

            if should_run_rollouts_in_subprocess:
                # run rollouts in subprocess

                all_rollout_logs, video_paths = run_rollouts_in_subprocess(
                    model=model,
                    config=config,
                    log_dir=log_dir,
                    env_meta=env_meta,
                    shape_meta=shape_meta,
                    obs_normalization_stats=obs_normalization_stats,
                    action_normalization_stats=action_normalization_stats,
                    mp_action_normalization_stats=mp_action_normalization_stats,
                    task_spec=(mg_task_spec if config.experiment.rollout.mode == "motion_planner" else None),
                    cached_motion_planner_targets=(mg_cached_motion_planner_targets if config.experiment.rollout.mode == "motion_planner" else None),
                    epoch=epoch,
                    video_dir=video_dir if config.experiment.render_video else None,
                )
            else:
                # run rollouts in current process

                # wrap model as a RolloutPolicy to prepare for rollouts
                if config.experiment.rollout.mode == "tamp_gated":
                    rollout_model = {
                        env_name : HTAMPRolloutPolicy(
                            model,
                            htamp_policy=htamp_policies[env_name],
                            env=envs[env_name],
                            htamp_use_joint_actions=config.experiment.rollout.htamp_use_joint_actions,
                            joint_controller_config=joint_controller_configs[env_name],
                            osc_controller_config=osc_controller_configs[env_name],
                            obs_normalization_stats=obs_normalization_stats,
                            action_normalization_stats=action_normalization_stats,
                            use_object_centric=config.experiment.rollout.htamp_object_centric.enabled,
                            obj_centric_ref_kwargs=config.experiment.rollout.htamp_object_centric.set_ref_kwargs,
                            obj_centric_observable_names=config.experiment.rollout.htamp_object_centric.observable_names,
                            obj_centric_obs_variants=config.experiment.rollout.htamp_object_centric.obs_variant,
                            obj_centric_noise=config.experiment.rollout.htamp_object_centric.noise,
                            use_policy_termination_classifier=config.experiment.rollout.htamp_use_policy_termination_classifier,
                            policy_termination_classifier_hold_mode=config.experiment.rollout.mp.policy_termination_classifier_hold_mode,
                            policy_termination_classifier_hold_count=config.experiment.rollout.mp.policy_termination_classifier_hold_count,
                        )
                        for env_name in envs
                    }
                elif config.experiment.rollout.mode == "motion_planner":
                    rollout_model = {
                        env_name : MotionPlannerRolloutPolicy(
                            model,
                            motion_planner=motion_planners[env_name],
                            env=envs[env_name],
                            task_spec=mg_task_spec,
                            cached_motion_planner_targets=mg_cached_motion_planner_targets,
                            joint_controller_config=joint_controller_configs[env_name],
                            osc_controller_config=osc_controller_configs[env_name],
                            obs_normalization_stats=obs_normalization_stats,
                            action_normalization_stats=action_normalization_stats,
                            use_mp_target_classifier=config.experiment.rollout.mp.use_mp_target_classifier,
                            mp_target_classifier_replan=config.experiment.rollout.mp.mp_target_classifier_replan,
                            use_mp_target_predictor=config.experiment.rollout.mp.use_mp_target_predictor,
                            use_policy_termination_classifier=config.experiment.rollout.mp.use_policy_termination_classifier,
                            policy_termination_classifier_hold_mode=config.experiment.rollout.mp.policy_termination_classifier_hold_mode,
                            policy_termination_classifier_hold_count=config.experiment.rollout.mp.policy_termination_classifier_hold_count,
                            policy_termination_classifier_tamp_ablation=config.experiment.rollout.mp.policy_termination_classifier_tamp_ablation,
                            mp_action_normalization_stats=mp_action_normalization_stats,
                            use_object_centric=config.experiment.rollout.htamp_object_centric.enabled,
                            obj_centric_ref_kwargs=config.experiment.rollout.htamp_object_centric.set_ref_kwargs,
                            obj_centric_observable_names=config.experiment.rollout.htamp_object_centric.observable_names,
                            obj_centric_obs_variants=config.experiment.rollout.htamp_object_centric.obs_variant,
                            obj_centric_noise=config.experiment.rollout.htamp_object_centric.noise,
                        )
                        for env_name in envs
                    }
                else:
                    rollout_model = RolloutPolicy(
                        model,
                        obs_normalization_stats=obs_normalization_stats,
                        action_normalization_stats=action_normalization_stats,
                    )

                num_episodes = config.experiment.rollout.n
                ensure_tamp_success = (config.experiment.rollout.htamp_ensure_tamp_success if config.experiment.rollout.mode == "tamp_gated" else False)
                all_rollout_logs, video_paths = TrainUtils.rollout_with_stats(
                    policy=rollout_model,
                    envs=envs,
                    horizon=env_rollout_horizons,
                    use_goals=config.use_goals,
                    num_episodes=num_episodes,
                    render=config.experiment.render,
                    video_dir=video_dir if config.experiment.render_video else None,
                    epoch=epoch,
                    video_skip=config.experiment.get("video_skip", 5),
                    terminate_on_success=config.experiment.rollout.terminate_on_success,
                    use_env_policy_step_count=config.experiment.rollout.use_env_policy_step_count,
                    ensure_tamp_success=ensure_tamp_success,
                )
                # if config.experiment.rollout.mode == "motion_planner":
                #     # rename some keys
                #     all_rollout_logs["MP_Success_Rate"] = all_rollout_logs.pop("TAMP_Success_Rate")
                #     all_rollout_logs["Success_Rate_no_MP_fail"] = all_rollout_logs.pop("Success_Rate_no_TAMP_fail")

            # summarize results from rollouts to tensorboard and terminal
            for env_name in all_rollout_logs:
                rollout_logs = all_rollout_logs[env_name]
                for k, v in rollout_logs.items():
                    if k == "TAMP_Failure_Info":
                        continue
                    if k.startswith("Time_"):
                        data_logger.record("Timing_Stats/Rollout_{}_{}".format(env_name, k[5:]), v, epoch)
                    else:
                        data_logger.record("Rollout/{}/{}".format(k, env_name), v, epoch, log_stats=True)

                # update global tamp failure counts
                for tamp_k in rollout_logs["TAMP_Failure_Info"]:
                    if env_name not in all_tamp_failure_info:
                        all_tamp_failure_info[env_name] = dict()
                    if tamp_k not in all_tamp_failure_info[env_name]:
                        all_tamp_failure_info[env_name][tamp_k] = 0
                    all_tamp_failure_info[env_name][tamp_k] += rollout_logs["TAMP_Failure_Info"][tamp_k]

                print("\nEpoch {} Rollouts took {}s (avg) with results:".format(epoch, rollout_logs["time"]))
                print('Env: {}'.format(env_name))
                print(json.dumps(rollout_logs, sort_keys=True, indent=4))

            # checkpoint and video saving logic
            updated_stats = TrainUtils.should_save_from_rollout_logs(
                epoch=epoch,
                all_rollout_logs=all_rollout_logs,
                best_return=best_return,
                best_success_rate=best_success_rate,
                best_success_rate_epoch=best_success_rate_epoch,
                best_avg_task_success_rate=best_avg_task_success_rate,
                best_avg_task_success_rate_epoch=best_avg_task_success_rate_epoch,
                best_tamp_failure_info=best_tamp_failure_info,
                epoch_ckpt_name=epoch_ckpt_name,
                save_on_best_rollout_return=config.experiment.save.on_best_rollout_return,
                save_on_best_rollout_success_rate=config.experiment.save.on_best_rollout_success_rate,
            )
            best_return = updated_stats["best_return"]
            best_success_rate = updated_stats["best_success_rate"]
            best_success_rate_epoch = updated_stats["best_success_rate_epoch"]
            best_avg_task_success_rate = updated_stats["best_avg_task_success_rate"]
            best_avg_task_success_rate_epoch = updated_stats["best_avg_task_success_rate_epoch"]
            best_tamp_failure_info = updated_stats["best_tamp_failure_info"]
            epoch_ckpt_name = updated_stats["epoch_ckpt_name"]
            should_save_ckpt = (config.experiment.save.enabled and updated_stats["should_save_ckpt"]) or should_save_ckpt
            if updated_stats["ckpt_reason"] is not None:
                ckpt_reason = updated_stats["ckpt_reason"]

            did_rollouts = True

        # Only keep saved videos if the ckpt should be saved (but not because of validation score)
        should_save_video = (should_save_ckpt and (ckpt_reason != "valid")) or config.experiment.keep_all_videos
        if video_paths is not None and not should_save_video:
            for env_name in video_paths:
                os.remove(video_paths[env_name])

        # # maybe upload rollout videos to wandb
        # if Macros.USE_WANDB and (video_paths is not None):
        #     for k in video_paths:
        #         if os.path.exists(video_paths[k]):
        #             wandb.log({"video": wandb.Video(video_paths[k], format="mp4")})

        # Save model checkpoints based on conditions (success rate, validation loss, etc)
        if should_save_ckpt and did_rollouts:
            last_saved_epoch_with_rollouts = epoch

        # get variable state for saving model
        variable_state = dict(
            epoch=epoch,
            best_valid_loss=best_valid_loss,
            best_return=best_return,
            best_success_rate=best_success_rate,
            best_success_rate_epoch=best_success_rate_epoch,
            best_avg_task_success_rate=best_avg_task_success_rate,
            best_avg_task_success_rate_epoch=best_avg_task_success_rate_epoch,
            best_tamp_failure_info=best_tamp_failure_info,
            all_tamp_failure_info=all_tamp_failure_info,
            last_saved_epoch_with_rollouts=last_saved_epoch_with_rollouts,
            time_elapsed=(time.time() - start_time + time_elapsed), # keep track of total time elapsed, including previous runs
        )
        if need_sync_results:
            variable_state.update(dict(
                best_ckpt_paths_synced=best_ckpt_paths_synced,
                best_video_paths_synced=best_video_paths_synced,
                last_ckpt_paths_synced=last_ckpt_paths_synced,
                last_video_paths_synced=last_video_paths_synced,
            ))

        if should_save_ckpt:
            TrainUtils.save_model(
                model=model,
                config=config,
                env_meta=env_meta,
                shape_meta=shape_meta,
                variable_state=variable_state,
                ckpt_path=os.path.join(ckpt_dir, epoch_ckpt_name + ".pth"),
                obs_normalization_stats=obs_normalization_stats,
                action_normalization_stats=action_normalization_stats,
                mp_action_normalization_stats=mp_action_normalization_stats,
                task_spec=(mg_task_spec if config.experiment.rollout.mode == "motion_planner" else None),
                cached_motion_planner_targets=(mg_cached_motion_planner_targets if config.experiment.rollout.mode == "motion_planner" else None),
            )

        # always save latest model for resume functionality
        print("\nsaving latest model at {}...\n".format(latest_model_path))
        TrainUtils.save_model(
            model=model,
            config=config,
            env_meta=env_meta,
            shape_meta=shape_meta,
            variable_state=variable_state,
            ckpt_path=latest_model_path,
            obs_normalization_stats=obs_normalization_stats,
            action_normalization_stats=action_normalization_stats,
            mp_action_normalization_stats=mp_action_normalization_stats,
            task_spec=(mg_task_spec if config.experiment.rollout.mode == "motion_planner" else None),
            cached_motion_planner_targets=(mg_cached_motion_planner_targets if config.experiment.rollout.mode == "motion_planner" else None),
        )

        # keep a backup model in case last.pth is malformed (e.g. job died last time during saving)
        shutil.copyfile(latest_model_path, latest_model_backup_path)
        print("\nsaved backup of latest model at {}\n".format(latest_model_backup_path))

        # maybe sync some results back to scratch space (only if rollouts happened)
        if did_rollouts and need_sync_results:
            print("Sync results back to sync path: {}".format(Macros.RESULTS_SYNC_PATH_ABS))

            # get best model checkpoint for each environment and corresponding video for each environment
            best_ckpt_paths_to_sync = dict()
            best_video_paths_to_sync = dict()
            for env_name in best_success_rate_epoch:
                cur_best_ckpt_path_to_sync, cur_best_video_paths_to_sync, cur_best_epoch_to_sync = TrainUtils.get_model_from_output_folder(
                    models_path=ckpt_dir,
                    videos_path=video_dir if config.experiment.render_video else None,
                    epoch=best_success_rate_epoch[env_name],
                    # env_name=env_name,
                )
                best_ckpt_paths_to_sync[env_name] = cur_best_ckpt_path_to_sync
                if config.experiment.render_video:
                    # assert len(cur_best_video_path_to_sync) == 1, "got these videos {} for env_name {}".format(cur_best_video_path_to_sync, env_name)  # should only be one video at most
                    best_video_paths_to_sync[env_name] = cur_best_video_paths_to_sync

            if last_saved_epoch_with_rollouts is not None:
                # get last model checkpoint and videos for each environment
                last_ckpt_path_to_sync, last_video_paths_to_sync, last_epoch_to_sync = TrainUtils.get_model_from_output_folder(
                    models_path=ckpt_dir,
                    videos_path=video_dir if config.experiment.render_video else None,
                    epoch=last_saved_epoch_with_rollouts,
                )
                assert last_epoch_to_sync == last_saved_epoch_with_rollouts

            # get best model checkpoint and corresponding videos for best average task success rate, if evaluating on more than one env
            best_avg_task_ckpt_path_to_sync = None
            best_avg_task_video_paths_to_sync = []
            if len(best_success_rate_epoch) > 1:
                best_avg_task_ckpt_path_to_sync, best_avg_task_video_paths_to_sync, _ = TrainUtils.get_model_from_output_folder(
                    models_path=ckpt_dir,
                    videos_path=video_dir if config.experiment.render_video else None,
                    epoch=best_avg_task_success_rate_epoch,
                )

            # deprecated - now we ask for explicit epochs
            # best_ckpt_path_to_sync, best_video_path_to_sync, best_epoch_to_sync = TrainUtils.get_model_from_output_folder(
            #     models_path=ckpt_dir,
            #     videos_path=video_dir if config.experiment.render_video else None,
            #     best=True,
            # )
            # last_ckpt_path_to_sync, last_video_path_to_sync, last_epoch_to_sync = TrainUtils.get_model_from_output_folder(
            #     models_path=ckpt_dir,
            #     videos_path=video_dir if config.experiment.render_video else None,
            #     last=True,
            # )

            # clear last files that we synced over
            if best_ckpt_paths_synced is not None:
                for path in best_ckpt_paths_synced:
                    os.remove(path)
            if last_ckpt_paths_synced is not None:
                for path in last_ckpt_paths_synced:
                    os.remove(path)
            if best_video_paths_synced is not None:
                for path in best_video_paths_synced:
                    os.remove(path)
            if last_video_paths_synced is not None:
                for path in last_video_paths_synced:
                    os.remove(path)
            if os.path.exists(log_dir_path_synced):
                shutil.rmtree(log_dir_path_synced)

            # sync best ckpts per env and remember sync paths
            best_ckpt_paths_synced = []
            for env_name in best_ckpt_paths_to_sync:
                to_sync = os.path.join(
                    Macros.RESULTS_SYNC_PATH_ABS,
                    os.path.basename(best_ckpt_paths_to_sync[env_name])[:-4] + "_best_{}.pth".format(env_name),
                )
                shutil.copyfile(best_ckpt_paths_to_sync[env_name], to_sync)
                best_ckpt_paths_synced.append(to_sync)
            if best_avg_task_ckpt_path_to_sync is not None:
                to_sync = os.path.join(
                    Macros.RESULTS_SYNC_PATH_ABS,
                    os.path.basename(best_avg_task_ckpt_path_to_sync)[:-4] + "_AVG_best.pth",
                )
                shutil.copyfile(best_avg_task_ckpt_path_to_sync, to_sync)
                best_ckpt_paths_synced.append(to_sync)

            # sync last ckpt and remember sync path
            to_sync = os.path.join(
                Macros.RESULTS_SYNC_PATH_ABS,
                os.path.basename(last_ckpt_path_to_sync)[:-4] + "_last.pth",
            )
            shutil.copyfile(last_ckpt_path_to_sync, to_sync)
            last_ckpt_paths_synced = [to_sync]

            if config.experiment.render_video:

                # sync best videos per env and remember sync paths
                best_video_paths_synced = []
                for env_name in best_video_paths_to_sync:
                    for best_vid_path in best_video_paths_to_sync[env_name]:
                        to_sync = os.path.join(
                            Macros.RESULTS_SYNC_PATH_ABS,
                            os.path.basename(best_vid_path)[:-4] + "_best_{}_{}.mp4".format(env_name, best_success_rate[env_name]),
                        )
                        shutil.copyfile(best_vid_path, to_sync)
                        best_video_paths_synced.append(to_sync)
                for best_vid_path in best_avg_task_video_paths_to_sync:
                    to_sync = os.path.join(
                        Macros.RESULTS_SYNC_PATH_ABS,
                        os.path.basename(best_vid_path)[:-4] + "_AVG_best_{}.mp4".format(best_avg_task_success_rate),
                    )
                    shutil.copyfile(best_vid_path, to_sync)
                    best_video_paths_synced.append(to_sync)

                # sync last videos per env and remember sync paths
                last_video_paths_synced = []
                for last_vid_path in last_video_paths_to_sync:
                    to_sync = os.path.join(
                        Macros.RESULTS_SYNC_PATH_ABS,
                        os.path.basename(last_vid_path)[:-4] + "_last_{}.mp4".format(env_name),
                    )
                    shutil.copyfile(last_vid_path, to_sync)
                    last_video_paths_synced.append(to_sync)

            # sync logs dir
            shutil.copytree(log_dir, log_dir_path_synced)
            # sync config json
            shutil.copyfile(
                os.path.join(log_dir, '..', 'config.json'),
                os.path.join(Macros.RESULTS_SYNC_PATH_ABS, 'config.json')
            )

        # Finally, log memory usage in MB
        process = psutil.Process(os.getpid())
        mem_usage = int(process.memory_info().rss / 1000000)
        data_logger.record("System/RAM Usage (MB)", mem_usage, epoch)
        print("\nEpoch {} Memory Usage: {} MB\n".format(epoch, mem_usage))

    # terminate logging
    data_logger.close()

    # sync logs after closing data logger to make sure everything was transferred
    if need_sync_results:
        print("Sync results back to sync path: {}".format(Macros.RESULTS_SYNC_PATH_ABS))
        # sync logs dir
        if os.path.exists(log_dir_path_synced):
            shutil.rmtree(log_dir_path_synced)
        shutil.copytree(log_dir, log_dir_path_synced)

    # collect important statistics
    important_stats = dict()
    prefix = "Rollout/Success_Rate/"
    exception_prefix = "Rollout/Exception_Rate/"
    exception_retry_prefix = "Rollout/Exception_Rate_Retry/"

    success_rates_by_env = dict()
    all_success_rates_by_epoch = dict() # all success rates by epoch (one per env)
    for k in data_logger._data:
        if k.startswith(prefix):
            env_name = k[len(prefix):]
            success_rates_by_env[env_name] = deepcopy(data_logger._data[k]) # dict mapping epoch to success rate
            for sr_epoch in success_rates_by_env[env_name]:
                if sr_epoch not in all_success_rates_by_epoch:
                    all_success_rates_by_epoch[sr_epoch] = []
                all_success_rates_by_epoch[sr_epoch].append(success_rates_by_env[env_name][sr_epoch])
            stats = data_logger.get_stats(k)
            important_stats["{}-max".format(env_name)] = stats["max"]
            important_stats["{}-mean".format(env_name)] = stats["mean"]
        elif k.startswith(exception_prefix):
            env_name = k[len(exception_prefix):]
            stats = data_logger.get_stats(k)
            important_stats["{}-exception-rate-max".format(env_name)] = stats["max"]
            important_stats["{}-exception-rate-mean".format(env_name)] = stats["mean"]
        elif k.startswith(exception_retry_prefix):
            env_name = k[len(exception_retry_prefix):]
            stats = data_logger.get_stats(k)
            important_stats["{}-exception-rate-retry-max".format(env_name)] = stats["max"]
            important_stats["{}-exception-rate-retry-mean".format(env_name)] = stats["mean"]

    if config.experiment.rollout.enabled:
        # get best average task success rate across all tasks and epochs
        avg_task_success_rates_by_epoch = dict() # average task success rate across all envs
        for sr_epoch in all_success_rates_by_epoch:
            if len(all_success_rates_by_epoch[sr_epoch]) == len(success_rates_by_env):
                avg_task_success_rates_by_epoch[sr_epoch] = np.mean(all_success_rates_by_epoch[sr_epoch])
        best_avg_task_epoch = max(avg_task_success_rates_by_epoch, key=lambda k: avg_task_success_rates_by_epoch[k])
        last_avg_task_epoch = max(list(avg_task_success_rates_by_epoch.keys()))
        important_stats["avg-task-max"] = avg_task_success_rates_by_epoch[best_avg_task_epoch]
        important_stats["avg-task-last"] = avg_task_success_rates_by_epoch[last_avg_task_epoch]
        important_stats["avg-task-max-at-epoch"] = best_avg_task_epoch
        important_stats["avg-task-max-all-envs"] = {
            env_name : success_rates_by_env[env_name][best_avg_task_epoch]
            for env_name in success_rates_by_env
        }

        # get per-env best SR and last SR as well
        all_env_names = list(success_rates_by_env.keys())
        max_epoch = max(list(success_rates_by_env[all_env_names[0]].keys()))
        best_epoch_by_env = dict()
        for env_name in all_env_names:
            best_epoch = max(success_rates_by_env[env_name], key=lambda k: success_rates_by_env[env_name][k])
            best_epoch_by_env[env_name] = best_epoch

            # record best epoch
            important_stats["{}-max-at-epoch".format(env_name)] = best_epoch

            # record success rate at last epoch
            important_stats["{}-last".format(env_name)] = success_rates_by_env[env_name][max_epoch]

            # record success rates for all other envs at this env's best epoch
            for env_name2 in all_env_names:
                if env_name == env_name2:
                    continue
                important_stats["{}-at-{}-max".format(env_name2, env_name)] = success_rates_by_env[env_name2][best_epoch]

    # add in tamp failure info
    if best_tamp_failure_info is not None:
        for env_name in best_tamp_failure_info:
            important_stats["best_tamp_failure_info_{}".format(env_name)] = best_tamp_failure_info[env_name]
            if env_name in all_tamp_failure_info:
                important_stats["all_tamp_failure_info_{}".format(env_name)] = all_tamp_failure_info[env_name]

    # add in time taken
    important_stats["time spent (hrs)"] = "{:.2f}".format((time.time() - start_time + time_elapsed) / 3600.)

    # write stats to disk
    json_file_path = os.path.join(log_dir, "important_stats.json")
    with open(json_file_path, 'w') as f:
        # preserve original key ordering
        json.dump(important_stats, f, sort_keys=False, indent=4)

    return important_stats


def main(args):

    if args.config is not None:
        ext_cfg = json.load(open(args.config, 'r'))
        config = config_factory(ext_cfg["algo_name"])
        # update config with external json - this will throw errors if
        # the external config has keys not present in the base algo config
        with config.values_unlocked():
            config.update(ext_cfg)
    else:
        config = config_factory(args.algo)

    if args.dataset is not None:
        config.train.data = args.dataset

    if args.name is not None:
        config.experiment.name = args.name

    if args.output is not None:
        config.train.output_dir = args.output

    # get torch device
    device = TorchUtils.get_torch_device(try_to_use_cuda=config.train.cuda)

    # # wandb project name
    # wandb_project_name = args.wandb_project_name

    # maybe modify config for debugging purposes
    if args.debug:
        Macros.DEBUG = True

        # shrink length of training to test whether this run is likely to crash
        config.unlock()
        config.lock_keys()

        # train and validate (if enabled) for 3 gradient steps, for 2 epochs
        config.experiment.epoch_every_n_steps = 3
        config.experiment.validation_epoch_every_n_steps = 3
        config.train.num_epochs = 2

        # if rollouts are enabled, try 2 rollouts at end of each epoch, with 10 environment steps
        config.experiment.rollout.rate = 1
        config.experiment.rollout.n = 2
        config.experiment.rollout.horizon = 10
        config.experiment.rollout.use_env_policy_step_count = True # ensure policy is used for a few steps

        if config.experiment.additional_horizons is not None:
            config.experiment.additional_horizons = [10] * len(config.experiment.additional_horizons)

        # send output to a temporary directory
        config.train.output_dir = "/tmp/tmp_trained_models"

        # # set wandb project name to "test" since it isn't an official run
        # wandb_project_name = "test"

    # lock config to prevent further modifications and ensure missing keys raise errors
    config.lock()

    # # maybe setup wandb
    # if Macros.USE_WANDB:
    #     # set api key as env variable
    #     os.environ["WANDB_API_KEY"] = Macros.WANDB_API_KEY

    #     wandb.init(
    #         project=wandb_project_name,
    #         entity=Macros.WANDB_ENTITY,
    #         sync_tensorboard=True,
    #         name=config.experiment.name,
    #         config=config,
    #     )

    # catch error during training and print it
    res_str = "finished run successfully!"
    important_stats = None
    important_stats_str = None
    try:
        important_stats = train(config, device=device, auto_remove_exp=args.auto_remove_exp, resume=args.resume)
    except Exception as e:
        res_str = "run failed with error:\n{}\n\n{}".format(e, traceback.format_exc())
    print(res_str)
    if important_stats is not None:
        important_stats_str = json.dumps(important_stats, indent=4)
        print("\nRollout Success Rate Stats")
        print(important_stats_str)

        # maybe sync important stats back
        if Macros.RESULTS_SYNC_PATH_ABS is not None:
            json_file_path = os.path.join(Macros.RESULTS_SYNC_PATH_ABS, "important_stats.json")
            with open(json_file_path, 'w') as f:
                # preserve original key ordering
                json.dump(important_stats, f, sort_keys=False, indent=4)

    # # maybe cleanup wandb
    # if Macros.USE_WANDB:
    #     wandb.finish()

    # maybe give slack notification
    notify_on_slack(config=config, res_str=res_str, important_stats=important_stats_str)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # External config file that overwrites default config
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="(optional) path to a config json that will be used to override the default settings. \
            If omitted, default settings are used. This is the preferred way to run experiments.",
    )

    # Algorithm Name
    parser.add_argument(
        "--algo",
        type=str,
        help="(optional) name of algorithm to run. Only needs to be provided if --config is not provided",
    )

    # Experiment Name (for tensorboard, saving models, etc.)
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="(optional) if provided, override the experiment name defined in the config",
    )

    # Dataset path, to override the one in the config
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="(optional) if provided, override the dataset path defined in the config",
    )

    # Output path, to override the one in the config
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="(optional) if provided, override the output folder path defined in the config",
    )

    # force delete the experiment folder if it exists
    parser.add_argument(
        "--auto-remove-exp",
        action='store_true',
        help="force delete the experiment folder if it exists",
    )

    # debug mode
    parser.add_argument(
        "--debug",
        action='store_true',
        help="set this flag to run a quick training run for debugging purposes",
    )

    # resume training from latest checkpoint
    parser.add_argument(
        "--resume",
        action='store_true',
        help="set this flag to resume training from latest checkpoint",
    )

    # # wandb project name
    # parser.add_argument(
    #     "--wandb_project_name",
    #     type=str,
    #     default="test",
    # )

    args = parser.parse_args()
    main(args)

