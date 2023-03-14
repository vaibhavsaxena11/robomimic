## python scripts/train.py --algo=bc --dataset=../datasets/robomimic_v0.1/proficient-human/lift/image_depth_xyz.hdf5
## python scripts/train.py --algo=bc --dataset=../datasets/robomimic_v0.1/proficient-human/lift/image_depth_xyzsegm.hdf5
## python scripts/train.py --algo=bc --dataset=../datasets/ravens/block-insertion/block_insertion.hdf5
## python scripts/train.py --debug --debug_rollout --algo=bc --dataset=../datasets/ravens/train10-test10/block-insertion/block_insertion.hdf5
## python scripts/train.py --debug --algo=bc --dataset=../datasets/ravens/train100-test10/block-insertion/block_insertion.hdf5
## python scripts/train.py --debug --algo=bc --dataset=../datasets/ravens/train500-test50/block-insertion/block_insertion.hdf5
## python scripts/train.py --debug --algo=bc --dataset=../datasets/ravens/train1000-test100/place-red-in-green/place_red_in_green.hdf5
## python scripts/train.py --debug --algo=bc --dataset=../datasets/ravens/train100-test50/place-red-in-green/place_red_in_green.hdf5
## python scripts/train.py --debug --algo=bc --dataset=../datasets/ravens/train100-test10/place-big-red-in-green/place_big_red_in_green.hdf5
## python scripts/train.py --debug --algo=bc --dataset=../datasets/ravens/train50-test10/sweeping-piles/sweeping_piles.hdf5


#XXX# srun --gres gpu:1 -p overcap -A overcap python scripts/train.py --algo=bc --dataset=../datasets/robomimic_v0.1/proficient-human/lift/image_depth_xyz.hdf5
## srun --gres gpu:1 -p overcap -A overcap python scripts/train.py --algo=bc --dataset=/srv/share/vsaxena33/work/3d-rep-manip/robomimic/datasets/robomimic_v0.1/proficient-human/lift/image_depth_xyz.hdf5
## srun --gres gpu:1 -p overcap -A overcap python scripts/train.py --algo=bc --dataset=/srv/share/vsaxena33/work/3d-rep-manip/robomimic/datasets/robomimic_v0.1/proficient-human/lift/image_depth_xyzsegm.hdf5

#XXX# srun --gres gpu:1 -p overcap -A overcap python scripts/train.py --algo=bc --dataset=../datasets/robomimic_v0.1/proficient-human/can/image_depth_xyz.hdf5
## srun --gres gpu:1 -p overcap -A overcap python scripts/train.py --algo=bc --dataset=/srv/share/vsaxena33/work/3d-rep-manip/robomimic/datasets/robomimic_v0.1/proficient-human/can/image_depth_xyz.hdf5
## srun --gres gpu:1 -p overcap -A overcap python scripts/train.py --algo=bc --dataset=/srv/share/vsaxena33/work/3d-rep-manip/robomimic/datasets/robomimic_v0.1/proficient-human/can/image_depth_xyzsegm.hdf5

## srun --gres gpu:1 -p overcap -A overcap python scripts/train.py --algo=bc --dataset=/srv/share/vsaxena33/work/3d-rep-manip/robomimic/datasets/robomimic_v0.1/proficient-human/square/image_depth_xyz.hdf5
## srun --gres gpu:1 -p overcap -A overcap python scripts/train.py --algo=bc --dataset=/srv/share/vsaxena33/work/3d-rep-manip/robomimic/datasets/robomimic_v0.1/proficient-human/square/image_depth_xyzsegm.hdf5

## srun --gres gpu:1 -p overcap -A overcap python scripts/train.py --algo=bc --dataset=/srv/share/vsaxena33/work/3d-rep-manip/robomimic/datasets/ravens/block-insertion/block_insertion.hdf5
## srun --gres gpu:1 -p overcap -A overcap python scripts/train.py --algo=bc --dataset=/srv/share/vsaxena33/work/3d-rep-manip/robomimic/datasets/ravens/train10-test10/block_insertion.hdf5
## srun --gres gpu:1 -p overcap -A overcap python scripts/train.py --algo=bc --dataset=/srv/share/vsaxena33/work/3d-rep-manip/robomimic/datasets/ravens/train100-test10/block_insertion.hdf5
## srun --gres gpu:1 -p overcap -A overcap python scripts/train.py --algo=bc --dataset=/srv/share/vsaxena33/work/3d-rep-manip/robomimic/datasets/ravens/train500-test50/block_insertion.hdf5

## srun --gres gpu:1 -p overcap -A overcap python scripts/train.py --algo=bc --dataset=/srv/share/vsaxena33/work/3d-rep-manip/robomimic/datasets/ravens/train1000-test100/place-red-in-green/place_red_in_green.hdf5
## srun --gres gpu:1 -p overcap -A overcap python scripts/train.py --algo=bc --dataset=/srv/share/vsaxena33/work/3d-rep-manip/robomimic/datasets/ravens/train100-test50/place_red_in_green.hdf5

## srun --gres gpu:1 -p overcap -A overcap python scripts/train.py --algo=bc --dataset=/srv/share/vsaxena33/work/3d-rep-manip/robomimic/datasets/ravens/train100-test10/place_big_red_in_green.hdf5

## srun --gres gpu:1 -p overcap -A overcap python scripts/train.py --algo=bc --dataset=/srv/share/vsaxena33/work/3d-rep-manip/robomimic/datasets/ravens/train50-test10/sweeping_piles.hdf5


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
import socket
import traceback

from collections import OrderedDict

import torch
from torch.utils.data import DataLoader

import robomimic
import robomimic.utils.train_utils as TrainUtils
import robomimic.utils.torch_utils as TorchUtils
import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.file_utils as FileUtils
from robomimic.config import config_factory
from robomimic.algo import algo_factory, RolloutPolicy
from robomimic.utils.log_utils import PrintLogger, DataLogger


def train(config, device, args=None):
    """
    Train a model using the algorithm.
    """

    # first set seeds
    np.random.seed(config.train.seed)
    torch.manual_seed(config.train.seed)

    print("\n============= New Training Run with Config =============")
    print(config)
    print("")
    log_dir, ckpt_dir, video_dir = TrainUtils.get_exp_dir(config)

    if config.experiment.logging.terminal_output_to_txt and not (args.debug or args.debug_rollout):
        # log stdout and stderr to a text file
        logger = PrintLogger(os.path.join(log_dir, 'log.txt'))
        sys.stdout = logger
        sys.stderr = logger

    # read config to set up metadata for observation modalities (e.g. detecting rgb observations)
    ObsUtils.initialize_obs_utils_with_config(config)

    # make sure the dataset exists
    dataset_path = os.path.expanduser(config.train.data)
    if not os.path.exists(dataset_path):
        raise Exception("Dataset at provided path {} not found!".format(dataset_path))

    # load basic metadata from training file
    print("\n============= Loaded Environment Metadata =============")
    env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path=config.train.data)
    shape_meta = FileUtils.get_shape_metadata_from_dataset(
        dataset_path=config.train.data,
        all_obs_keys=config.all_obs_keys,#None
        verbose=True
    )

    if env_meta["env_name"] == "Ravens":
        env_meta["env_kwargs"]["assets_root"] = os.path.join(robomimic.__path__[0], "../../ravens/ravens/environments/assets/") # TODO(VS) add this to config, and add code to include env_kwargs from config too

    if config.experiment.env is not None:
        env_meta["env_name"] = config.experiment.env
        print("=" * 30 + "\n" + "Replacing Env to {}\n".format(env_meta["env_name"]) + "=" * 30)

    # create environment
    envs = OrderedDict()
    if config.experiment.rollout.enabled:
        # create environments for validation runs
        env_names = [env_meta["env_name"]]

        if config.experiment.additional_envs is not None:
            for name in config.experiment.additional_envs:
                env_names.append(name)

        for env_name in env_names:
            ## NOTE(VS): env is setup by default to output ALL observations it can. They are filtered later in the encoder, based on what is provided during training.
            ## Only the images are a special case, which the env is setup to output only if they are present in the SequenceDataset during training as well.
            env = EnvUtils.create_env_from_metadata(
                env_meta=env_meta,
                env_name=env_name, 
                render=False, 
                render_offscreen=config.experiment.render_video,
                use_image_obs=shape_meta["use_images"], 
            )
            envs[env.name] = env
            print(envs[env.name])

    print("")

    # setup for a new training run
    data_logger = DataLogger(
        log_dir,
        log_tb=config.experiment.logging.log_tb,
    )
    model = algo_factory(
        algo_name=config.algo_name,
        config=config,
        obs_key_shapes=shape_meta["all_shapes"],
        ac_dim=shape_meta["ac_dim"],
        device=device,
    )

    if args.debug_rollout:
        ## TODO(VS) remove later; temporarily loading ckpt from path
        from robomimic.utils.file_utils import load_dict_from_checkpoint
        # model_ckpt = load_dict_from_checkpoint("/Users/vaibhav/work/3d-rep-manip/robomimic/bc_trained_models/test/20230124145427/models/model_epoch_1100.pth")["model"]
        # model_ckpt = load_dict_from_checkpoint("/Users/vaibhav/work/3d-rep-manip/robomimic/bc_trained_models/test/20230125130137/models/model_epoch_50.pth")["model"]
        # model_ckpt = load_dict_from_checkpoint("/Users/vaibhav/work/3d-rep-manip/robomimic/bc_trained_models/test/20230125142636/models/model_epoch_200.pth")["model"]
        # model_ckpt = load_dict_from_checkpoint("/Users/vaibhav/work/3d-rep-manip/robomimic/bc_trained_models/test/20230125142636/models/model_epoch_1400.pth")["model"]
        # model_ckpt = load_dict_from_checkpoint("/Users/vaibhav/work/3d-rep-manip/robomimic/bc_trained_models/test/20230125183250 - 2Dpickplace 256negex_negy/models/model_epoch_1750.pth")["model"]
        # model_ckpt = load_dict_from_checkpoint("/Users/vaibhav/work/3d-rep-manip/robomimic/bc_trained_models/test/20230201123243/models/model_epoch_100.pth")["model"]
        # model_ckpt = load_dict_from_checkpoint("/Users/vaibhav/work/3d-rep-manip/robomimic/bc_trained_models/test/20230131234557 - 2Dpickplace blockinsertion/models/model_epoch_950.pth")["model"]
        # model_ckpt = load_dict_from_checkpoint("/Users/vaibhav/work/3d-rep-manip/robomimic/bc_trained_models/test/20230204135031 - (server) 2Dpickplace bi front no-pool 256negex/models/model_epoch_100.pth")["model"]
        # model_ckpt = load_dict_from_checkpoint("/Users/vaibhav/work/3d-rep-manip/robomimic/bc_trained_models/test/20230206094807/models/model_epoch_50.pth")["model"]
        # model_ckpt = load_dict_from_checkpoint("/Users/vaibhav/work/3d-rep-manip/robomimic/bc_trained_models/test/20230206171330/models/model_epoch_350.pth")["model"]
        # model_ckpt = load_dict_from_checkpoint("/Users/vaibhav/work/3d-rep-manip/robomimic/bc_trained_models/test/20230206204225 - (server) afflite pring_front_tr1k resnet18+MLP aff_resnetMLP lr1e5 bs50/models/model_epoch_150.pth")["model"]

        # model_ckpt = load_dict_from_checkpoint("/Users/vaibhav/work/3d-rep-manip/robomimic/bc_trained_models/test/20230206212311 - (server) afflite bi_ortho_tr500 resnet18+MLP aff_resnetMLP lr1e5 bs50/models/model_epoch_950.pth")["model"] ## GOOD # set pad_images=False
        # model_ckpt = load_dict_from_checkpoint("/Users/vaibhav/work/3d-rep-manip/robomimic/bc_trained_models/test/20230207164605 - (server) afflite sweep_ortho_tr50 resnet18+MLP aff_resnetMLP+sigm lr1e5 bs10/models/model_epoch_2000.pth")["model"]
        
        # model_ckpt = load_dict_from_checkpoint("/Users/vaibhav/work/3d-rep-manip/robomimic/bc_trained_models/test/20230209215341 - (server) afflite bi_ortho_tr500 resnet18+MLP aff_resnetMLP+sigm_recon lr1e5 bs50/models/model_epoch_950.pth")["model"]
        # model_ckpt = load_dict_from_checkpoint("/Users/vaibhav/work/3d-rep-manip/robomimic/bc_trained_models/test/20230209214829 - (server) afflite bi_ortho_tr500 resnet18+MLP aff_resnetMLP+sigm_l2+nce+recon lr1e5 bs50/models/model_epoch_1050.pth")["model"]
        
        model_ckpt = load_dict_from_checkpoint("/Users/vaibhav/work/3d-rep-manip/robomimic/bc_trained_models/test/20230223172847/models/model_epoch_150.pth")["model"] ## GOOD # set pad_images=False

        model.deserialize(model_ckpt)


    #### Loading pretrained weights
    # # import pdb; pdb.set_trace()
    # if len(config.observation.modalities.obs.pcd) > 0:
    #     ndf_weights = torch.load('./ndf_robot/src/ndf_robot/model_weights/multi_category_weights.pth', map_location=torch.device('cpu'))
    #     weights = OrderedDict()
    #     for k, v in ndf_weights.items():
    #         if k.startswith("encoder"):
    #             weights['.'.join(k.split('.')[1:])] = v
    #     # import pdb; pdb.set_trace()
    #     if config.observation.modalities.obs.pcd[0] == "robot0_eye_in_hand_xyz":
    #         model.nets['policy'].nets.encoder.nets.obs.obs_nets.robot0_eye_in_hand_xyz.pointnet.load_state_dict(weights)
    #     elif config.observation.modalities.obs.pcd[0] == "agentview_xyz":
    #         model.nets['policy'].nets.encoder.nets.obs.obs_nets.agentview_xyz.pointnet.load_state_dict(weights)
    #     else:
    #         raise ValueError



    ### Computing model size
    # import pdb; pdb.set_trace()
    # # model_ = model.nets['policy'].nets.encoder.nets.obs.obs_nets.agentview_xyz.pointnet
    # model_ = model.nets['policy'].nets.encoder.nets.obs.obs_nets.agentview_xyz.pointnet
    # param_size = 0
    # for param in model_.parameters():
    #     param_size += param.nelement() * param.element_size()
    # buffer_size = 0
    # for buffer in model_.buffers():
    #     buffer_size += buffer.nelement() * buffer.element_size()

    # size_all_mb = (param_size + buffer_size) / 1024**2
    # print('model size: {:.3f}MB'.format(size_all_mb))
    # import pdb; pdb.set_trace()
    
    
    # save the config as a json file
    with open(os.path.join(log_dir, '..', 'config.json'), 'w') as outfile:
        json.dump(config, outfile, indent=4)

    print("\n============= Model Summary =============")
    print(model)  # print model summary
    print("")

    # load training data
    trainset, validset = TrainUtils.load_data_for_training(
        config, obs_keys=shape_meta["all_obs_keys"])
    train_sampler = trainset.get_dataset_sampler()
    print("\n============= Training Dataset =============")
    print(trainset)
    print("")

    # maybe retreve statistics for normalizing observations
    obs_normalization_stats = None
    if config.train.hdf5_normalize_obs:
        obs_normalization_stats = trainset.get_obs_normalization_stats()

    # initialize data loaders
    train_loader = DataLoader(
        dataset=trainset,
        sampler=train_sampler,
        batch_size=config.train.batch_size,
        shuffle=False,#(train_sampler is None), ##TODO(VS) remove False; added for debug
        num_workers=config.train.num_data_workers,
        drop_last=True
    )

    if config.experiment.validate:
        # cap num workers for validation dataset at 1
        num_workers = min(config.train.num_data_workers, 1)
        valid_sampler = validset.get_dataset_sampler()
        valid_loader = DataLoader(
            dataset=validset,
            sampler=valid_sampler,
            batch_size=config.train.batch_size,
            shuffle=False,#(valid_sampler is None), ##TODO(VS) remove False; added for debug
            num_workers=num_workers,
            drop_last=True
        )
    else:
        valid_loader = None

    # main training loop
    best_valid_loss = None
    best_return = {k: -np.inf for k in envs} if config.experiment.rollout.enabled else None
    best_success_rate = {k: -1. for k in envs} if config.experiment.rollout.enabled else None
    last_ckpt_time = time.time()

    # number of learning steps per epoch (defaults to a full dataset pass)
    train_num_steps = config.experiment.epoch_every_n_steps
    valid_num_steps = config.experiment.validation_epoch_every_n_steps

    for epoch in range(1, config.train.num_epochs + 1): # epoch numbers start at 1
        step_log, affordances, input_imgs, img_recons = TrainUtils.run_epoch(model=model, data_loader=train_loader, epoch=epoch, num_steps=train_num_steps, debug=args.debug_rollout)
        if affordances is not None:
            # data_logger.save_3d_affordance_grid(affordances, f"train_{epoch}")
            for k in affordances: data_logger.save_2d_affordances_and_images(affordances[k], input_imgs, img_recons[k], k, f"train_{epoch}")
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
                step_log, affordances, input_imgs, img_recons = TrainUtils.run_epoch(model=model, data_loader=valid_loader, epoch=epoch, validate=True, num_steps=valid_num_steps, debug=args.debug_rollout)
                if affordances is not None:
                    # data_logger.save_3d_affordance_grid(affordances, f"valid_{epoch}")
                    for k in affordances: data_logger.save_2d_affordances_and_images(affordances[k], input_imgs, img_recons[k], k, f"valid_{epoch}")
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
        # rollout_check=True ##TODO(VS) remove debug; added to shortcut to saving video
        if config.experiment.rollout.enabled and (epoch > config.experiment.rollout.warmstart) and rollout_check:

            # wrap model as a RolloutPolicy to prepare for rollouts
            rollout_model = RolloutPolicy(model, obs_normalization_stats=obs_normalization_stats)

            num_episodes = config.experiment.rollout.n
            all_rollout_logs, video_paths = TrainUtils.rollout_with_stats(
                policy=rollout_model,
                envs=envs,
                horizon=config.experiment.rollout.horizon,
                use_goals=config.use_goals,
                num_episodes=num_episodes,
                render=False,
                video_dir=video_dir if config.experiment.render_video else None,
                epoch=epoch,
                video_skip=config.experiment.get("video_skip", 5),
                terminate_on_success=config.experiment.rollout.terminate_on_success,
            )

            # summarize results from rollouts to tensorboard and terminal
            for env_name in all_rollout_logs:
                rollout_logs = all_rollout_logs[env_name]
                for k, v in rollout_logs.items():
                    if k.startswith("Time_"):
                        data_logger.record("Timing_Stats/Rollout_{}_{}".format(env_name, k[5:]), v, epoch)
                    else:
                        data_logger.record("Rollout/{}/{}".format(k, env_name), v, epoch, log_stats=True)

                print("\nEpoch {} Rollouts took {}s (avg) with results:".format(epoch, rollout_logs["time"]))
                print('Env: {}'.format(env_name))
                print(json.dumps(rollout_logs, sort_keys=True, indent=4))

            # checkpoint and video saving logic
            updated_stats = TrainUtils.should_save_from_rollout_logs(
                all_rollout_logs=all_rollout_logs,
                best_return=best_return,
                best_success_rate=best_success_rate,
                epoch_ckpt_name=epoch_ckpt_name,
                save_on_best_rollout_return=config.experiment.save.on_best_rollout_return,
                save_on_best_rollout_success_rate=config.experiment.save.on_best_rollout_success_rate,
            )
            best_return = updated_stats["best_return"]
            best_success_rate = updated_stats["best_success_rate"]
            epoch_ckpt_name = updated_stats["epoch_ckpt_name"]
            should_save_ckpt = (config.experiment.save.enabled and updated_stats["should_save_ckpt"]) or should_save_ckpt
            if updated_stats["ckpt_reason"] is not None:
                ckpt_reason = updated_stats["ckpt_reason"]

        # Only keep saved videos if the ckpt should be saved (but not because of validation score)
        should_save_video = (should_save_ckpt and (ckpt_reason != "valid")) or config.experiment.keep_all_videos
        if video_paths is not None and not should_save_video:
            for env_name in video_paths:
                os.remove(video_paths[env_name])

        # Save model checkpoints based on conditions (success rate, validation loss, etc)
        if should_save_ckpt:
            TrainUtils.save_model(
                model=model,
                config=config,
                env_meta=env_meta,
                shape_meta=shape_meta,
                ckpt_path=os.path.join(ckpt_dir, epoch_ckpt_name + ".pth"),
                obs_normalization_stats=obs_normalization_stats,
            )

        # Finally, log memory usage in MB
        process = psutil.Process(os.getpid())
        mem_usage = int(process.memory_info().rss / 1000000)
        data_logger.record("System/RAM Usage (MB)", mem_usage, epoch)
        print("\nEpoch {} Memory Usage: {} MB\n".format(epoch, mem_usage))

    # terminate logging
    data_logger.close()


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

    # ## adding for d4rl data
    # config.experiment.logging.log_tb = False
    # config.experiment.validate = False
    # config.observation.modalities.obs.low_dim = ["flat"]

    ## Adding for robosuite data ##
    # config.experiment.logging.log_tb = False
    config.observation.modalities.obs.low_dim = [
        "robot0_eef_pos",
        "robot0_eef_quat",
        "robot0_gripper_qpos",
        "object"
    ]

    ## Adding image modality data
    # config.observation.modalities.obs.rgb = ["agentview_image"] ## when using object or pcd, do not use image
    
    ## Removing gt object state
    config.observation.modalities.obs.low_dim = [ # removing gt object state
        "robot0_eef_pos",
        "robot0_eef_quat",
        "robot0_gripper_qpos",
    ]
    ## Adding pcd modality data, and 
    # config.observation.modalities.obs.pcd = ["robot0_eye_in_hand_xyz"] # Added new modality; each modality is encoded in a similar way (i.e. nets)
    # config.observation.modalities.obs.pcd = ["agentview_xyz"] # Added new modality; each modality is encoded in a similar way (i.e. nets)
    config.observation.encoder.pcd.core_class = "PointCloudCore"
    config.observation.encoder.pcd.core_kwargs = {"batch_norm": True, "channel_multiplier": 1, "pool": "max"}
    # config.observation.encoder.pcd.core_class = "ResnetPointnetCore"
    # config.observation.encoder.pcd.core_kwargs = {"k": 5, "c_dim": 256}
    config.observation.encoder.pcd.obs_randomizer_class = None
    config.observation.encoder.pcd.obs_randomizer_kwargs = None #TODO(VS) set kwargs for pointcloud
    ##

    ## Configs for Ravens dataset
    # config.experiment.rollout.enabled = False
    config.observation.modalities.obs.low_dim = []
    config.observation.modalities.obs.rgb = []
    config.observation.modalities.obs.rgb = ["color_ortho"] # ["color_front"] # 
    config.observation.modalities.obs.pcd = []
    # config.observation.modalities.obs.pcd = ["xyz_front"]
    config.observation.encoder.rgb.core_kwargs.pool_class = None
    config.observation.encoder.rgb.core_kwargs.pool_kwargs = {}
    config.observation.encoder.rgb.core_kwargs.backbone_class = "ResNet18Conv" #  "ResNet50Conv" # "UNetConv" # 
    config.observation.encoder.rgb.core_kwargs.backbone_kwargs.pretrained = False # True
    # config.observation.encoder.rgb.core_kwargs.backbone_kwargs.init_features = 32 # 8 # 32 # For UNetConv only, comment this line for other backbones


    #### Configs for Implicit Policy training
    config.algo.implicit.enabled = True
    config.algo.implicit.feat_dim = 100 # 10 # shape of decoder output (MLP) in MIMO_MLP #NOT BEING USED
    config.algo.implicit.affordance_layer_dims = (100, 100, 100, 100, 1) # (100, 100, 1) # 
    config.algo.implicit.pad_images = True # False ## When True, manually uncomment padding code in ImageModality.process_obs()
    config.algo.implicit.num_aug_rotated_images = 10 # 0 ## NOTE batch_size grows by this factor
    config.algo.implicit.L2explicit_loss_enabled = True  #True
    config.algo.implicit.NCEimplicit_loss_enabled = True # True
    config.algo.implicit.L2implicitRecon_loss_enabled = False # True
    config.algo.implicit.L2backboneRecon_loss_enabled = False # True ### ONLY True when corresponding encoder supports it
    config.algo.implicit.L2backboneRecon_loss_weight = 1. # 100.
    config.algo.implicit.stopgrad_act2enc = False # True # 
    config.algo.implicit.pos_enc_per_dim = None # 50 # 

    #### Configs for Rollout
    config.experiment.rollout.enabled = True # False
    config.experiment.rollout.rate = 50 # 1 # 
    config.experiment.rollout.n = 10 # 1
    config.experiment.rollout.horizon = 10
    config.experiment.video_skip = 1
    config.experiment.save.every_n_epochs = 50 # 10

    config.train.batch_size = 10 # 4 # 5 # 50 #### SEEM TO BE ABLE TO FIT 100 IMAGES TOTAL
    if args.debug or args.debug_rollout:
        config.train.batch_size = 5
    if args.debug_rollout:
        config.experiment.rollout.enabled = True
        config.experiment.rollout.rate = 1 # 5


    # optim params for the two policies in BC_Implicit
    config.unlock()

    config.algo.optim_params.pick_policy.learning_rate.initial = 1e-4 # 3e-4 # 1e-5 # 1e-6 # 1e-4      # pick_policy learning rate
    config.algo.optim_params.pick_policy.learning_rate.decay_factor = 0.1  # factor to decay LR by (if epoch schedule non-empty)
    config.algo.optim_params.pick_policy.learning_rate.epoch_schedule = [] # epochs where LR decay occurs
    config.algo.optim_params.pick_policy.regularization.L2 = 0.00          # L2 regularization strength

    config.algo.optim_params.place_policy.learning_rate.initial = 1e-4 # 3e-4 # 1e-5 # 1e-6 # 1e-4      # place_policy learning rate
    config.algo.optim_params.place_policy.learning_rate.decay_factor = 0.1  # factor to decay LR by (if epoch schedule non-empty)
    config.algo.optim_params.place_policy.learning_rate.epoch_schedule = [] # epochs where LR decay occurs
    config.algo.optim_params.place_policy.regularization.L2 = 0.00          # L2 regularization strength

    config.algo.optim_params.encoder.learning_rate.initial = 1e-4 # 3e-4 # 1e-5 # 1e-6 # 1e-4      # encoder learning rate
    config.algo.optim_params.encoder.learning_rate.decay_factor = 0.1  # factor to decay LR by (if epoch schedule non-empty)
    config.algo.optim_params.encoder.learning_rate.epoch_schedule = [] # epochs where LR decay occurs
    config.algo.optim_params.encoder.regularization.L2 = 0.00          # L2 regularization strength
    ####


    # get torch device
    device = TorchUtils.get_torch_device(try_to_use_cuda=config.train.cuda)

    # # maybe modify config for debugging purposes
    # if args.debug:
    #     # shrink length of training to test whether this run is likely to crash
    #     config.unlock()
    #     config.lock_keys()

    #     # train and validate (if enabled) for 3 gradient steps, for 2 epochs
    #     config.experiment.epoch_every_n_steps = 3
    #     config.experiment.validation_epoch_every_n_steps = 3
    #     config.train.num_epochs = 2

    #     # if rollouts are enabled, try 2 rollouts at end of each epoch, with 10 environment steps
    #     config.experiment.rollout.rate = 1
    #     config.experiment.rollout.n = 2
    #     config.experiment.rollout.horizon = 10

    #     # send output to a temporary directory
    #     config.train.output_dir = "/tmp/tmp_trained_models"

    # lock config to prevent further modifications and ensure missing keys raise errors
    config.lock()

    # catch error during training and print it
    res_str = "finished run successfully!"
    try:
        train(config, device=device, args=args)
    except Exception as e:
        res_str = "run failed with error:\n{}\n\n{}".format(e, traceback.format_exc())
    print(res_str)


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

    # debug mode
    parser.add_argument(
        "--debug",
        action='store_true',
        help="set this flag to run a quick training run for debugging purposes"
    )

    # debug mode
    parser.add_argument(
        "--debug_rollout",
        action='store_true',
        help="set this flag to run a quick training run for debugging purposes"
    )

    args = parser.parse_args()
    main(args)

