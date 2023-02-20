"""
Helper script to convert a dataset collected using ravens into an hdf5 compatible with
this repository. Takes a dataset path corresponding to the demos collected using ravens.
By default, the script also creates a 90-10 train-validation split.

Expects data at <src_folder>/<task>/<task>-<train|test>

Example usage:
    # puts processed dataset at provided root folder
    python convert_ravens.py --src_folder path/to/dataset/root --task <task-name>
    
    python scripts/conversion/convert_ravens.py --src_folder ./datasets/ravens/ --task block-insertion
    srun --gres gpu:1 -p overcap -A overcap python scripts/conversion/convert_ravens.py --src_folder /srv/share/vsaxena33/work/3d-rep-manip/robomimic/datasets/ravens --task block-insertion
    python scripts/conversion/convert_ravens.py --src_folder ./datasets/ravens/ --task place-red-in-green
"""
# python scripts/conversion/convert_ravens.py --folder ./datasets/ravens/ --task block-insertion
# python robomimic/scripts/conversion/convert_ravens.py --src_folder ./datasets/ravens/ --task place-red-in-green
# python robomimic/scripts/conversion/convert_ravens.py --src_folder ../ravens/data/train10-test10 --dst_folder ./datasets/ravens/train10-test10 --task block-insertion
# python robomimic/scripts/conversion/convert_ravens.py --src_folder ../ravens/data/train100-test10 --dst_folder ./datasets/ravens/train100-test10 --task block-insertion
# python robomimic/scripts/conversion/convert_ravens.py --src_folder ../ravens/data/train500-test50 --dst_folder ./datasets/ravens/train500-test50 --task block-insertion
# python robomimic/scripts/conversion/convert_ravens.py --src_folder ../ravens/data/train100-test10 --dst_folder ./datasets/ravens/train100-test10 --task place-big-red-in-green
# python robomimic/scripts/conversion/convert_ravens.py --src_folder ../ravens/data/train100-test50 --dst_folder ./datasets/ravens/train100-test50 --task place-red-in-green
# python robomimic/scripts/conversion/convert_ravens.py --src_folder ../ravens/data/train50-test10 --dst_folder ./datasets/ravens/train50-test10 --task sweeping-piles

import h5py
import json
import argparse
import os
import pickle
import numpy as np

import robomimic
import robomimic.envs.env_base as EB
from robomimic.utils.log_utils import custom_tqdm
from robomimic.utils.geometry import get_xyz_from_depth

from ravens.tasks import cameras


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--src_folder",
        type=str,
        help="path to ravens source dataset",
    )
    parser.add_argument(
        "--dst_folder",
        type=str,
        default=None,
        help="path to destination folder",
    )
    parser.add_argument(
        "--task",
        type=str,
        help="name of task",
    )
    args = parser.parse_args()

    # creating output file
    output_base_folder = args.dst_folder if args.dst_folder is not None else args.src_folder
    os.makedirs(os.path.join(output_base_folder, args.task), exist_ok=True)
    output_path = os.path.join(output_base_folder, args.task, "{}.hdf5".format(args.task.replace("-", "_")))
    f_sars = h5py.File(output_path, "w")
    f_sars_grp = f_sars.create_group("data")
    f_mask_grp = f_sars.create_group("mask")

    # camera idxs:: 0: front view, 1: looking over left shoulder; 2: looking over right shoulder
    camera_name2id = {"front": 0, "left": 1, "right": 2, "ortho": 3}

    # obtaining camera intrinsics for point cloud computation
    camera_configs = cameras.RealSenseD415.CONFIG

    split_to_path_map = {"train": "train", "valid": "test"}

    total_samples = 0
    num_traj = 0
    for split in ["train", "valid"]:
        data_path = os.path.join(args.src_folder, args.task, "-".join([args.task, split_to_path_map[split]]))
        all_actions = [os.path.join(data_path, "action", f) for f in os.listdir(os.path.join(data_path, "action"))]
        all_color = [os.path.join(data_path, "color", f) for f in os.listdir(os.path.join(data_path, "color"))]
        all_depth = [os.path.join(data_path, "depth", f) for f in os.listdir(os.path.join(data_path, "depth"))]
        all_info = [os.path.join(data_path, "info", f) for f in os.listdir(os.path.join(data_path, "info"))]
        all_reward = [os.path.join(data_path, "reward", f) for f in os.listdir(os.path.join(data_path, "reward"))]
        # TODO(VS) assert lengths of above lists are equal
        num_trajs = len(all_actions)

        mask_demos = [] # for storing demo names for this split in the hdf5

        print("\nConverting {} hdf5...".format(split))
        for idx in custom_tqdm(range(num_trajs)):
            traj = dict()
            ## TODO(VS) check if below works
            traj["actions"] = pickle.load(open(all_actions[idx], 'rb'))
            traj["colors"] = pickle.load(open(all_color[idx], 'rb')) # shape (traj_len, num_cams, H, W, 3) # H, W can be different for different cams
            traj["depths"] = pickle.load(open(all_depth[idx], 'rb')) # shape (traj_len, num_cams, H, W)  # H, W can be different for different cams
            # traj["infos"] = pickle.load(open(all_info[idx], 'rb'))
            traj["rewards"] = pickle.load(open(all_reward[idx], 'rb'))

            ## store trajectory

            # store observations (color and depth) and rewards
            ep_data_grp = f_sars_grp.create_group("demo_{}".format(num_traj))
            num_color_cams = len(traj["colors"][0])
            num_depth_cams = len(traj["depths"][0])
            for cam_name, cam_id in camera_name2id.items():
                # import pdb; pdb.set_trace()
                if cam_id < num_color_cams:
                    # ep_data_grp.create_dataset(f"obs/color_{cam_name}", data=np.array(traj["colors"][:-1, cam_id]))
                    # ep_data_grp.create_dataset(f"next_obs/color_{cam_name}", data=np.array(traj["colors"][1:, cam_id]))
                    ep_data_grp.create_dataset(f"obs/color_{cam_name}", data=np.stack([traj["colors"][i][cam_id] for i in range(len(traj["colors"]) - 1)], 0))
                    ep_data_grp.create_dataset(f"next_obs/color_{cam_name}", data=np.stack([traj["colors"][i][cam_id] for i in range(1, len(traj["colors"]))], 0))
                if cam_id < num_depth_cams:
                    # ep_data_grp.create_dataset(f"obs/depth_{cam_name}", data=np.array(traj["depths"][:-1, cam_id]))
                    # ep_data_grp.create_dataset(f"next_obs/depth_{cam_name}", data=np.array(traj["depths"][1:, cam_id]))
                    ep_data_grp.create_dataset(f"obs/depth_{cam_name}", data=np.stack([traj["depths"][i][cam_id] for i in range(len(traj["depths"]) - 1)], 0))
                    ep_data_grp.create_dataset(f"next_obs/depth_{cam_name}", data=np.stack([traj["depths"][i][cam_id] for i in range(1, len(traj["depths"]))], 0))
            ep_data_grp.create_dataset("rewards", data=np.array(traj["rewards"][1:]))
            


            # store point-cloud (xyz), only for front camera to save space
            cam_name = "front"; cam_id = camera_name2id[cam_name]
            xyzs = []
            depths = [traj["depths"][i][cam_id] for i in range(len(traj["depths"]))]
            for depth in depths: # looping over trajectory length
                xyz = get_xyz_from_depth(
                    depth, # depth map
                    np.reshape(camera_configs[cam_id]["intrinsics"], (3, 3)), # intrinsics
                    camera_configs[cam_id]["image_size"][0], # camera height
                    camera_configs[cam_id]["image_size"][1], # camera width
                )
                xyzs.append(xyz)
            ep_data_grp.create_dataset(f"obs/xyz_{cam_name}", data=xyzs[:-1])
            ep_data_grp.create_dataset(f"next_obs/xyz_{cam_name}", data=xyzs[1:])

            # ## TODO(VS) remove temporary plotting code
            # import matplotlib.pyplot as plt
            # import imageio
            # xyz = np.reshape(xyzs[0], [-1, 3])
            # fig = plt.figure()
            # ax = fig.add_subplot(projection='3d')
            # ax.scatter(xyz[:,0], xyz[:,1], xyz[:,2])
            # ax.set_xlabel('X')
            # ax.set_ylabel('Y')
            # ax.set_zlabel('Z')
            # plt.savefig(f'ravens_{cam_name}_xyz.png')
            # imageio.imsave(f'ravens_{cam_name}_image.png', traj["colors"][0, cam_id])
            # imageio.imsave(f'ravens_{cam_name}_depth.png', traj["depths"][0, cam_id])
            # import pdb; pdb.set_trace()


            # store actions
            pose0_tra = []
            pose0_rot = []
            pose1_tra = []
            pose1_rot = []
            for act in traj["actions"]:
                if act is None: # last action of trajectory
                    break
                pose0_tra.append(act["pose0"][0])
                pose0_rot.append(act["pose0"][1])
                pose1_tra.append(act["pose1"][0])
                pose1_rot.append(act["pose1"][1])
            # ep_data_grp.create_dataset("actions/pose0_tra", data=np.array(pose0_tra))
            # ep_data_grp.create_dataset("actions/pose0_rot", data=np.array(pose0_rot))
            # ep_data_grp.create_dataset("actions/pose1_tra", data=np.array(pose1_tra))
            # ep_data_grp.create_dataset("actions/pose1_rot", data=np.array(pose1_rot))

            # appending all tra and rot poses, along with pose0 and pose1, together into one 14-dim action
            actions = np.concatenate([np.array(pose0_tra), np.array(pose0_rot), np.array(pose1_tra), np.array(pose1_rot)], -1)
            ep_data_grp.create_dataset("actions", data=actions)

            ## TODO(VS) check if infos are useful, otherwise ignore
            # ep_data_grp.create_dataset("infos", data=traj["infos"]) ## TODO(VS) need to store dict in hdf5

            ep_data_grp.attrs["num_samples"] = len(traj["actions"]) - 1 # ignoring last obs (no action)

            mask_demos.append("demo_{}".format(num_traj)) # for later adding to hdf5

            total_samples += len(traj["actions"])
            num_traj += 1

        f_mask_grp.create_dataset(split, data=np.array(mask_demos, dtype="S"))

    # store env meta - edit env_kwargs below based on the env config using which the data was collected
    env_meta = dict(
        type=EB.EnvType.RAVENS_TYPE,
        env_name="Ravens",
        env_kwargs={
            "task": args.task,
            "continuous": False,
            "disp": False,
            "shared_memory": False,
            "hz": 480,
        }, ## TODO(VS) fill in after writing env_ravens.py
    )
    ## TODO(VS) store env kwargs
    # if "env_args" in f["data"].attrs:
    #     del f["data"].attrs["env_args"]
    f_sars["data"].attrs["env_args"] = json.dumps(env_meta, indent=4)

    # print("====== Stored env meta ======")
    # print(f["data"].attrs["env_args"])

    # add total samples to global metadata
    f_sars["data"].attrs["total"] = total_samples

    print("Wrote {} trajectories to new converted hdf5 at {}\n".format(num_traj, output_path))

    f_sars.close()
