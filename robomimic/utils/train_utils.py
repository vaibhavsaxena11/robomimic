"""
This file contains several utility functions used to define the main training loop. It 
mainly consists of functions to assist with logging, rollouts, and the @run_epoch function,
which is the core training logic for models in this repository.
"""
import os
import time
import datetime
import shutil
import json
import h5py
import imageio
import numpy as np
import numbers
import pathlib
from copy import deepcopy
from collections import OrderedDict

import torch

import robomimic
import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.log_utils as LogUtils

from robomimic.utils.dataset import SequenceDataset
from robomimic.envs.env_base import EnvBase
from robomimic.algo import RolloutPolicy


def get_exp_dir(config, auto_remove_exp_dir=False):
    """
    Create experiment directory from config. If an identical experiment directory
    exists and @auto_remove_exp_dir is False (default), the function will prompt 
    the user on whether to remove and replace it, or keep the existing one and
    add a new subdirectory with the new timestamp for the current run.

    Args:
        auto_remove_exp_dir (bool): if True, automatically remove the existing experiment
            folder if it exists at the same path.
    
    Returns:
        log_dir (str): path to created log directory (sub-folder in experiment directory)
        output_dir (str): path to created models directory (sub-folder in experiment directory)
            to store model checkpoints
        video_dir (str): path to video directory (sub-folder in experiment directory)
            to store rollout videos
    """
    # timestamp for directory names
    t_now = time.time()
    time_str = datetime.datetime.fromtimestamp(t_now).strftime('%Y%m%d%H%M%S')

    # create directory for where to dump model parameters, tensorboard logs, and videos
    base_output_dir = os.path.expanduser(config.train.output_dir)
    if not os.path.isabs(base_output_dir):
        # relative paths are specified relative to robomimic module location
        base_output_dir = os.path.join(robomimic.__path__[0], base_output_dir)
    base_output_dir = os.path.join(base_output_dir, config.experiment.name)
    if os.path.exists(base_output_dir):
        if not auto_remove_exp_dir:
            ans = input("WARNING: model directory ({}) already exists! \noverwrite? (y/n)\n".format(base_output_dir))
        else:
            ans = "y"
        if ans == "y":
            print("REMOVING")
            shutil.rmtree(base_output_dir)

    # only make model directory if model saving is enabled
    output_dir = None
    if config.experiment.save.enabled:
        output_dir = os.path.join(base_output_dir, time_str, "models")
        os.makedirs(output_dir)

    # tensorboard directory
    log_dir = os.path.join(base_output_dir, time_str, "logs")
    os.makedirs(log_dir)

    # video directory
    video_dir = os.path.join(base_output_dir, time_str, "videos")
    os.makedirs(video_dir)
    return log_dir, output_dir, video_dir


def load_data_for_training(config, obs_keys):
    """
    Data loading at the start of an algorithm.

    Args:
        config (BaseConfig instance): config object
        obs_keys (list): list of observation modalities that are required for
            training (this will inform the dataloader on what modalities to load)

    Returns:
        train_dataset (SequenceDataset instance): train dataset object
        valid_dataset (SequenceDataset instance): valid dataset object (only if using validation)
    """

    # config can contain an attribute to filter on
    filter_by_attribute = config.train.hdf5_filter_key

    # load the dataset into memory
    if config.experiment.validate:
        assert not config.train.hdf5_normalize_obs, "no support for observation normalization with validation data yet"
        train_filter_by_attribute = "train"
        valid_filter_by_attribute = "valid"
        if filter_by_attribute is not None:
            train_filter_by_attribute = "{}_{}".format(filter_by_attribute, train_filter_by_attribute)
            valid_filter_by_attribute = "{}_{}".format(filter_by_attribute, valid_filter_by_attribute)
        train_dataset = dataset_factory(config, obs_keys, filter_by_attribute=train_filter_by_attribute)
        valid_dataset = dataset_factory(config, obs_keys, filter_by_attribute=valid_filter_by_attribute)
    else:
        train_dataset = dataset_factory(config, obs_keys, filter_by_attribute=filter_by_attribute)
        valid_dataset = None

    return train_dataset, valid_dataset


def dataset_factory(config, obs_keys, filter_by_attribute=None, dataset_path=None):
    """
    Create a SequenceDataset instance to pass to a torch DataLoader.

    Args:
        config (BaseConfig instance): config object

        obs_keys (list): list of observation modalities that are required for
            training (this will inform the dataloader on what modalities to load)

        filter_by_attribute (str): if provided, use the provided filter key
            to select a subset of demonstration trajectories to load

        dataset_path (str): if provided, the SequenceDataset instance should load
            data from this dataset path. Defaults to config.train.data.

    Returns:
        dataset (SequenceDataset instance): dataset object
    """
    if dataset_path is None:
        dataset_path = config.train.data

    ds_kwargs = dict(
        hdf5_path=dataset_path,
        obs_keys=obs_keys,
        dataset_keys=config.train.dataset_keys,
        load_next_obs=True, # make sure dataset returns s'
        frame_stack=1, # no frame stacking
        seq_length=config.train.seq_length,
        pad_frame_stack=True,
        pad_seq_length=True, # pad last obs per trajectory to ensure all sequences are sampled
        get_pad_mask=False,
        goal_mode=config.train.goal_mode,
        hdf5_cache_mode=config.train.hdf5_cache_mode,
        hdf5_use_swmr=config.train.hdf5_use_swmr,
        hdf5_normalize_obs=config.train.hdf5_normalize_obs,
        filter_by_attribute=filter_by_attribute
    )
    dataset = SequenceDataset(**ds_kwargs)

    return dataset


def run_rollout(
        policy, 
        env, 
        horizon,
        use_goals=False,
        render=False,
        video_writer=None,
        video_skip=5,
        terminate_on_success=False,
    ):
    """
    Runs a rollout in an environment with the current network parameters.

    Args:
        policy (RolloutPolicy instance): policy to use for rollouts.

        env (EnvBase instance): environment to use for rollouts.

        horizon (int): maximum number of steps to roll the agent out for

        use_goals (bool): if True, agent is goal-conditioned, so provide goal observations from env

        render (bool): if True, render the rollout to the screen

        video_writer (imageio Writer instance): if not None, use video writer object to append frames at 
            rate given by @video_skip

        video_skip (int): how often to write video frame

        terminate_on_success (bool): if True, terminate episode early as soon as a success is encountered

    Returns:
        results (dict): dictionary containing return, success rate, etc.
    """
    assert isinstance(policy, RolloutPolicy)
    assert isinstance(env, EnvBase)

    policy.start_episode()

    ob_dict = env.reset()
    goal_dict = None
    if use_goals:
        # retrieve goal from the environment
        goal_dict = env.get_goal()

    results = {}
    video_count = 0  # video frame counter

    total_reward = 0.
    success = { k: False for k in env.is_success() } # success metrics

    ac_strings = []
    if env.name == "Ravens":
        pick_error = 0.
        place_error = 0.
        n_horizon = 0
    try:
        for step_i in range(horizon):

            # get action from policy
            ac = policy(ob=ob_dict, goal=goal_dict)
            ac_strings.append(f"t={step_i} {str(ac)}")

            if env.name == "Ravens":
                ac_oracle = env.oracle.act(env.env._get_obs(), None)
                pick_error += np.linalg.norm(ac_oracle["pose0"][0] - ac[:3])
                place_error += np.linalg.norm(ac_oracle["pose1"][0] - ac[7:10])
                n_horizon += 1

            # play action
            ob_dict, r, done, _ = env.step(ac)

            # render to screen
            if render:
                env.render(mode="human")

            # compute reward
            total_reward += r

            cur_success_metrics = env.is_success()
            for k in success:
                success[k] = success[k] or cur_success_metrics[k]

            # visualization
            if video_writer is not None:
                if video_count % video_skip == 0:
                    video_img = env.render(mode="rgb_array", height=512, width=512)
                    video_writer.append_data(video_img)

                video_count += 1

            # break if done
            if done or (terminate_on_success and success["task"]):
                break

    except env.rollout_exceptions as e:
        print("WARNING: got rollout exception {}".format(e))

    results["Return"] = total_reward
    results["Horizon"] = step_i + 1
    results["Success_Rate"] = float(success["task"])
    results["Actions"] = ac_strings
    if env.name == "Ravens":
        results["Avg_Pick_Error"] = pick_error/n_horizon
        results["Avg_Place_Error"] = place_error/n_horizon

    # log additional success metrics
    for k in success:
        if k != "task":
            results["{}_Success_Rate".format(k)] = float(success[k])

    return results


def rollout_with_stats(
        policy,
        envs,
        horizon,
        use_goals=False,
        num_episodes=None,
        render=False,
        video_dir=None,
        video_path=None,
        epoch=None,
        video_skip=5,
        terminate_on_success=False,
        verbose=False,
    ):
    """
    A helper function used in the train loop to conduct evaluation rollouts per environment
    and summarize the results.

    Can specify @video_dir (to dump a video per environment) or @video_path (to dump a single video
    for all environments).

    Args:
        policy (RolloutPolicy instance): policy to use for rollouts.

        envs (dict): dictionary that maps env_name (str) to EnvBase instance. The policy will
            be rolled out in each env.

        horizon (int): maximum number of steps to roll the agent out for

        use_goals (bool): if True, agent is goal-conditioned, so provide goal observations from env

        num_episodes (int): number of rollout episodes per environment

        render (bool): if True, render the rollout to the screen

        video_dir (str): if not None, dump rollout videos to this directory (one per environment)

        video_path (str): if not None, dump a single rollout video for all environments

        epoch (int): epoch number (used for video naming)

        video_skip (int): how often to write video frame

        terminate_on_success (bool): if True, terminate episode early as soon as a success is encountered

        verbose (bool): if True, print results of each rollout
    
    Returns:
        all_rollout_logs (dict): dictionary of rollout statistics (e.g. return, success rate, ...) 
            averaged across all rollouts 

        video_paths (dict): path to rollout videos for each environment
    """
    assert isinstance(policy, RolloutPolicy)

    all_rollout_logs = OrderedDict()

    # handle paths and create writers for video writing
    assert (video_path is None) or (video_dir is None), "rollout_with_stats: can't specify both video path and dir"
    write_video = (video_path is not None) or (video_dir is not None)
    video_paths = OrderedDict()
    video_writers = OrderedDict()
    if video_path is not None:
        # a single video is written for all envs
        video_paths = { k : video_path for k in envs }
        video_writer = imageio.get_writer(video_path, fps=20)
        video_writers = { k : video_writer for k in envs }
    if video_dir is not None:
        # video is written per env
        video_str = "_epoch_{}.mp4".format(epoch) if epoch is not None else ".mp4" 
        video_paths = { k : os.path.join(video_dir, "{}{}".format(k, video_str)) for k in envs }
        video_writers = { k : imageio.get_writer(video_paths[k], fps=20) for k in envs }

    for env_name, env in envs.items():
        env_video_writer = None
        if write_video:
            print("video writes to " + video_paths[env_name])
            env_video_writer = video_writers[env_name]

        print("rollout: env={}, horizon={}, use_goals={}, num_episodes={}".format(
            env.name, horizon, use_goals, num_episodes,
        ))
        rollout_logs = []
        iterator = range(num_episodes)
        if not verbose:
            iterator = LogUtils.custom_tqdm(iterator, total=num_episodes)

        num_success = 0
        for ep_i in iterator:
            rollout_timestamp = time.time()
            rollout_info = run_rollout(
                policy=policy,
                env=env,
                horizon=horizon,
                render=render,
                use_goals=use_goals,
                video_writer=env_video_writer,
                video_skip=video_skip,
                terminate_on_success=terminate_on_success,
            )
            rollout_info["time"] = time.time() - rollout_timestamp
            rollout_logs.append(rollout_info)
            num_success += rollout_info["Success_Rate"]
            if verbose:
                print("Episode {}, horizon={}, num_success={}".format(ep_i + 1, horizon, num_success))
                print(json.dumps(rollout_info, sort_keys=True, indent=4))

        if video_dir is not None:
            # close this env's video writer (next env has it's own)
            env_video_writer.close()

        # average metric across all episodes
        numeric_keys = [k for k, v in rollout_logs[0].items() if isinstance(v, numbers.Number)]
        nonnumeric_keys = list(set(rollout_logs[0].keys()).difference(set(numeric_keys)))
        rollout_logs = dict((k, [rollout_logs[i][k] for i in range(len(rollout_logs))]) for k in rollout_logs[0])
        rollout_logs_mean = dict((k, np.mean(v)) for k, v in rollout_logs.items() if k in numeric_keys)
        rollout_logs_mean["Time_Episode"] = np.sum(rollout_logs["time"]) / 60. # total time taken for rollouts in minutes
        # writing non-numeric logs to txt file
        for k in nonnumeric_keys:
            # rollout_logs_mean[k] = rollout_logs[k]
            fname = pathlib.Path(video_paths[env_name]).stem + f"_{k}.txt"
            fpath = pathlib.Path(video_paths[env_name]).parent / fname
            full_str = ""
            for i, s in enumerate(rollout_logs[k]):
                full_str += f"Episode {i}" + "\n" + "\n".join(s) + "\n\n"
            with open(fpath, 'w') as f:
                f.write(full_str)
        all_rollout_logs[env_name] = rollout_logs_mean

    if video_path is not None:
        # close video writer that was used for all envs
        video_writer.close()

    return all_rollout_logs, video_paths


def should_save_from_rollout_logs(
        all_rollout_logs,
        best_return,
        best_success_rate,
        epoch_ckpt_name,
        save_on_best_rollout_return,
        save_on_best_rollout_success_rate,
    ):
    """
    Helper function used during training to determine whether checkpoints and videos
    should be saved. It will modify input attributes appropriately (such as updating
    the best returns and success rates seen and modifying the epoch ckpt name), and
    returns a dict with the updated statistics.

    Args:
        all_rollout_logs (dict): dictionary of rollout results that should be consistent
            with the output of @rollout_with_stats

        best_return (dict): dictionary that stores the best average rollout return seen so far
            during training, for each environment

        best_success_rate (dict): dictionary that stores the best average success rate seen so far
            during training, for each environment

        epoch_ckpt_name (str): what to name the checkpoint file - this name might be modified
            by this function

        save_on_best_rollout_return (bool): if True, should save checkpoints that achieve a 
            new best rollout return

        save_on_best_rollout_success_rate (bool): if True, should save checkpoints that achieve a 
            new best rollout success rate

    Returns:
        save_info (dict): dictionary that contains updated input attributes @best_return,
            @best_success_rate, @epoch_ckpt_name, along with two additional attributes
            @should_save_ckpt (True if should save this checkpoint), and @ckpt_reason
            (string that contains the reason for saving the checkpoint)
    """
    should_save_ckpt = False
    ckpt_reason = None
    for env_name in all_rollout_logs:
        rollout_logs = all_rollout_logs[env_name]

        if rollout_logs["Return"] > best_return[env_name]:
            best_return[env_name] = rollout_logs["Return"]
            if save_on_best_rollout_return:
                # save checkpoint if achieve new best return
                epoch_ckpt_name += "_{}_return_{}".format(env_name, best_return[env_name])
                should_save_ckpt = True
                ckpt_reason = "return"

        if rollout_logs["Success_Rate"] > best_success_rate[env_name]:
            best_success_rate[env_name] = rollout_logs["Success_Rate"]
            if save_on_best_rollout_success_rate:
                # save checkpoint if achieve new best success rate
                epoch_ckpt_name += "_{}_success_{}".format(env_name, best_success_rate[env_name])
                should_save_ckpt = True
                ckpt_reason = "success"

    # return the modified input attributes
    return dict(
        best_return=best_return,
        best_success_rate=best_success_rate,
        epoch_ckpt_name=epoch_ckpt_name,
        should_save_ckpt=should_save_ckpt,
        ckpt_reason=ckpt_reason,
    )


def save_model(model, config, env_meta, shape_meta, ckpt_path, obs_normalization_stats=None):
    """
    Save model to a torch pth file.

    Args:
        model (Algo instance): model to save

        config (BaseConfig instance): config to save

        env_meta (dict): env metadata for this training run

        shape_meta (dict): shape metdata for this training run

        ckpt_path (str): writes model checkpoint to this path

        obs_normalization_stats (dict): optionally pass a dictionary for observation
            normalization. This should map observation keys to dicts
            with a "mean" and "std" of shape (1, ...) where ... is the default
            shape for the observation.
    """
    env_meta = deepcopy(env_meta)
    shape_meta = deepcopy(shape_meta)
    params = dict(
        model=model.serialize(),
        config=config.dump(),
        algo_name=config.algo_name,
        env_metadata=env_meta,
        shape_metadata=shape_meta,
    )
    if obs_normalization_stats is not None:
        assert config.train.hdf5_normalize_obs
        obs_normalization_stats = deepcopy(obs_normalization_stats)
        params["obs_normalization_stats"] = TensorUtils.to_list(obs_normalization_stats)
    torch.save(params, ckpt_path)
    print("save checkpoint to {}".format(ckpt_path))


# TODO(VS) remove commented code below
# def _create_affordance_grid(model, batch, policy_type="pick", policy_dim="2d"):
#     # Creating 3D affordance map. # TODO(VS) move this to bc.py if possible

#     if policy_dim == "3d":
#         raise NotImplementedError
#         # creating 3D action grid
#         actions = (np.mgrid[0:20, -10:10, 0:4]/20).transpose([1,2,3,0]).astype(np.float32) # shape (20, 20, 4, 3)
#         actions = actions.reshape([-1, 3])
#     else:
#         # creating 2D action grid
#         gridsize = 100
#         actions = (np.mgrid[0:gridsize, -gridsize//2:gridsize//2]/gridsize).transpose([2,1,0]).astype(np.float32) # shape (gridsize_x, gridsize_y, 2)
#         actions = actions.reshape([-1, 2])

#     # tiling grid to batch size
#     actions = torch.tensor(np.expand_dims(actions, 0))
#     batch_size = batch["obs"][list(batch["obs"].keys())[0]].shape[0]
#     actions = actions.tile([batch_size, 1, 1]).to(model.device)

#     import pdb; pdb.set_trace()
#     affordances, actions = model._create_aff_grid(batch, policy_type, policy_dim)

#     affordances = TensorUtils.detach(model.nets[f"{policy_type}_policy"](batch["obs"], actions, batch["goal_obs"])[0])

#     # affordances = torch.reshape(affordances, [batch_size, 10, 10, 10, 1])
#     # actions = actions.reshape([batch_size, 10, 10, 10, 3])

#     if policy_dim == "3d":
#         return torch.cat([actions, affordances], -1) # 3d only
#     else:
#         actions = actions.reshape([batch_size, gridsize, gridsize, 2])
#         return torch.cat([actions, torch.reshape(affordances, [batch_size, gridsize, gridsize, 1])], -1) # 2d only


def _create_affordance_grid(model, batch, policy_type="pick", policy_dim="2d"):
    # Creating 3D affordance map. # TODO(VS) move this to bc.py if possible

    if policy_dim == "3d":
        raise NotImplementedError
        # creating 3D action grid
        actions = (np.mgrid[0:20, -10:10, 0:4]/20).transpose([1,2,3,0]).astype(np.float32) # shape (20, 20, 4, 3)
        actions = actions.reshape([-1, 3])
    else:
        # obtaining 2D action grid and affordance from model._create_aff_grid()
        affordances, actions, img_recon = model._create_aff_grid(batch, policy_type, policy_dim)
        # concatenating actions and affordances before returning
        return torch.cat([actions, TensorUtils.detach(affordances).unsqueeze(-1)], -1), img_recon


def run_epoch(model, data_loader, epoch, validate=False, num_steps=None, debug=False):
    """
    Run an epoch of training or validation.

    Args:
        model (Algo instance): model to train

        data_loader (DataLoader instance): data loader that will be used to serve batches of data
            to the model

        epoch (int): epoch number

        validate (bool): whether this is a training epoch or validation epoch. This tells the model
            whether to do gradient steps or purely do forward passes.

        num_steps (int): if provided, this epoch lasts for a fixed number of batches (gradient steps),
            otherwise the epoch is a complete pass through the training dataset

    Returns:
        step_log_all (dict): dictionary of logged training metrics averaged across all batches
    """
    if debug:
        if not validate: return {}, None, None, None

    epoch_timestamp = time.time()
    if validate:
        model.set_eval()
    else:
        model.set_train()
    if num_steps is None:
        num_steps = len(data_loader)

    step_log_all = []
    timing_stats = dict(Data_Loading=[], Process_Batch=[], Train_Batch=[], Log_Info=[])
    start_time = time.time()

    data_loader_iter = iter(data_loader)
    affordances = {"pick":[], "place":[]}
    input_imgs = {}
    img_recons = {"pick":[], "place":[]}
    for _ in LogUtils.custom_tqdm(range(num_steps)):

        # load next batch from data loader
        try:
            t = time.time()
            batch = next(data_loader_iter)
        except StopIteration:
            # reset for next dataset pass
            data_loader_iter = iter(data_loader)
            t = time.time()
            batch = next(data_loader_iter)
        timing_stats["Data_Loading"].append(time.time() - t)

        # process batch for training
        t = time.time()
        input_batch = model.process_batch_for_training(batch)
        timing_stats["Process_Batch"].append(time.time() - t)

        # forward and backward pass
        t = time.time()
        info = model.train_on_batch(input_batch, epoch, validate=validate)
        timing_stats["Train_Batch"].append(time.time() - t)

        # tensorboard logging
        t = time.time()
        step_log = model.log_info(info)
        step_log_all.append(step_log)
        timing_stats["Log_Info"].append(time.time() - t)

        # obtain affordance grid
        if epoch % 1 == 0:# or validate:
            for ptype in affordances:
                aff, recon = _create_affordance_grid(model, input_batch, policy_type=ptype, policy_dim="2d")
                affordances[ptype].append(aff.detach().cpu().numpy())
                if recon is not None:
                    img_recons[ptype].append(recon.detach().cpu().numpy())

    # flatten and take the mean of the metrics
    step_log_dict = {}
    for i in range(len(step_log_all)):
        for k in step_log_all[i]:
            if k not in step_log_dict:
                step_log_dict[k] = []
            step_log_dict[k].append(step_log_all[i][k])
    step_log_all = dict((k, float(np.mean(v))) for k, v in step_log_dict.items())

    # add in timing stats
    for k in timing_stats:
        # sum across all training steps, and convert from seconds to minutes
        step_log_all["Time_{}".format(k)] = np.sum(timing_stats[k]) / 60.
    step_log_all["Time_Epoch"] = (time.time() - epoch_timestamp) / 60.

    # flatten affordance list
    if len(affordances[list(affordances.keys())[0]]) > 0:
        bs = len(input_batch["obs"][list(input_batch["obs"].keys())[0]]) # batch_size
        bs = min([bs, 4])
        for ptype in affordances:
            affordances[ptype] = np.concatenate(affordances[ptype], 0)[-bs:]
            if len(img_recons[ptype]) > 0:
                img_recons[ptype] = dict((k, np.concatenate(img_recons[ptype], 0)[-bs:]) for k in input_batch["obs"]) #TODO(VS) (multi-view experiments) get more recons for each image-type and log accordingly in inner dict.
            else:
                img_recons[ptype] = None
        input_imgs = dict((k, input_batch["obs"][k].detach().cpu().numpy()[-bs:]) for k in input_batch["obs"])

    else:
        affordances = None
        input_imgs = None
        img_recons = None

    return step_log_all, affordances, input_imgs, img_recons


def is_every_n_steps(interval, current_step, skip_zero=False):
    """
    Convenient function to check whether current_step is at the interval. 
    Returns True if current_step % interval == 0 and asserts a few corner cases (e.g., interval <= 0)
    
    Args:
        interval (int): target interval
        current_step (int): current step
        skip_zero (bool): whether to skip 0 (return False at 0)

    Returns:
        is_at_interval (bool): whether current_step is at the interval
    """
    if interval is None:
        return False
    assert isinstance(interval, int) and interval > 0
    assert isinstance(current_step, int) and current_step >= 0
    if skip_zero and current_step == 0:
        return False
    return current_step % interval == 0
