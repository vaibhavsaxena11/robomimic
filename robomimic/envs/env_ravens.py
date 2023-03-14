## NOTE(VS) Followed instructions here: https://robomimic.github.io/docs/modules/environments.html#implement-an-environment-wrapper
import numpy as np

import robomimic.envs.env_base as EB
import robomimic.utils.obs_utils as ObsUtils
from robomimic.utils.geometry import get_xyz_from_depth

from ravens import tasks
from ravens.environments.environment import ContinuousEnvironment
from ravens.environments.environment import Environment
from ravens.tasks import cameras

class EnvRavens(EB.EnvBase):
    """Wrapper class for Ravens environments (https://github.com/google-research/ravens)"""
    def __init__(
        self, 
        env_name,
        render=False,
        render_offscreen=False,
        use_image_obs=False, 
        postprocess_visual_obs=True, 
        task=None,
        mode="train", # train or valid; only used for AssemblingKits task
        continuous=False,
        **kwargs,
    ):
        env_cls = ContinuousEnvironment if continuous else Environment
        self.env = env_cls(**kwargs)
        # if task.endswith("ortho"): task = task[:-6]
        self.task = tasks.names[task](continuous=continuous)
        self.env.set_task(self.task)
        self.total_reward = 0
        self.camera_configs = cameras.RealSenseD415.CONFIG

        self.oracle = self.task.oracle(self.env, steps_per_seg=3) # steps_per_seg only used if env is ContinuousEnvironment
        # usage: self.oracle.act(self.env._get_obs(), None)

        self._kwargs = kwargs
        self._env_name = env_name
        self.postprocess_visual_obs = postprocess_visual_obs

    def step(self, action):
        # Take a step in the environment with an input action, return (observation, reward, done, info).
        act = {"pose0": (action[:3], action[3:7]), "pose1":(action[7:10], action[10:14])}
        obs, reward, done, info = self.env.step(act)
        self.total_reward += reward
        cmap, hmap, mask = self.task.get_true_image(self.env)
        obs['color'] = (*obs['color'], cmap)
        obs['depth'] = (*obs['depth'], hmap)
        return self.get_observation(obs), reward, done, info

    def reset(self):
        # Reset the environment, return observation
        self.total_reward = 0
        di = self.env.reset()
        cmap, hmap, mask = self.task.get_true_image(self.env)
        di['color'] = (*di['color'], cmap)
        di['depth'] = (*di['depth'], hmap)
        return self.get_observation(di)

    def render(self, mode="rgb_array", height=None, width=None, **kwargs):
        """
        Render from simulation to either an on-screen window or off-screen to RGB array.

        Args:
            mode (str): pass "human" for on-screen rendering or "rgb_array" for off-screen rendering
            height (int): height of image to render - only used if mode is "rgb_array"
            width (int): width of image to render - only used if mode is "rgb_array"
            camera_name (str): camera name to use for rendering
        """
        if "height" in kwargs.keys() or "width" in kwargs.keys():
            print("WARNING: image height or width passed to render was ignored. Rendering default image size.")
        if mode == "rgb_array":
            color = self.env.render() # renders only the color image from the first camera
            return color
        else:
            raise NotImplementedError('Only rgb_array implemented') ## TODO(VS) render to screen

    def get_observation(self, di=None):
        """
        Get current environment observation dictionary.

        Args:
            di (dict): current raw observation dictionary from ravens to wrap and provide 
                as a dictionary. If not provided, will be queried from ravens.
        """
        # Return the current environment observation as a dictionary, unless obs is not None. This function should process the raw environment observation to align with the input expected by the policy model. For example, it should cast an image observation to float with value range 0-1 and shape format [C, H, W].
        
        if di is None:
            di = self.env._get_obs() # contains keys "color" and "depth"
            cmap, hmap, mask = self.task.get_true_image(self.env)
            di['color'] = (*di['color'], cmap)
            di['depth'] = (*di['depth'], hmap)
        
        ret = {}
        # Setting keys in ret to match with demo dataset
        camera_name2id = {"front": 0, "left": 1, "right": 2, "ortho": 3}
        for k in ObsUtils.OBS_KEYS_TO_MODALITIES:
            # assuming keys are "<mod>_<cam>", i.e. "<color/depth/xyz>_<front/left/right>", others will be ignored 
            if len(k.split("_")) != 2: continue
            mod, cam = k.split("_")
            if mod in ["color", "depth"]:
                ret[k] = di[mod][camera_name2id[cam]]
            elif mod == "xyz":
                ret[k] = get_xyz_from_depth(
                    di["depth"][camera_name2id[cam]], 
                    np.reshape(self.camera_configs[camera_name2id[cam]]["intrinsics"], (3, 3)), # intrinsics
                    self.camera_configs[camera_name2id[cam]]["image_size"][0], # camera height
                    self.camera_configs[camera_name2id[cam]]["image_size"][1], # camera width
                )
            else:
                continue
                # raise ValueError(f"Observation {k} neither output by ravens nor computed in this wrapper.")

        # Processing observations (rgb and pcd)
        if self.postprocess_visual_obs:
            for k in ret:
                if ObsUtils.key_is_obs_modality(key=k, obs_modality="rgb") or ObsUtils.key_is_obs_modality(key=k, obs_modality="pcd"):
                    ret[k] = ObsUtils.process_obs(obs=ret[k], obs_key=k)
        return ret
        

    def is_success(self):
        """
        Check if the task condition(s) is reached. Should return a dictionary
        { str: bool } with at least a "task" key for the overall task success,
        and additional optional keys corresponding to other task criteria.
        """
        # if self.total_reward > 0.99:
        #     return { "task" : True }
        # return { "task" : False }
        return { "task" : self.task.done() }

    def is_done(self):
        return self.task.done() # task returns done=true when success

    def get_reward(self):
        pass

    def serialize(self):
        # Aggregate and return all information needed to re-instantiate this environment in a dictionary. This is the same as @env_meta - environment metadata stored in hdf5 datasets and used in robomimic/utils/env_utils.py.
        pass

    @classmethod
    def create_for_data_processing(cls):
        # (Optional) A class method that initialize an environment for data-postprocessing purposes, which includes extracting observations, labeling dense / sparse rewards, and annotating dones in transitions. This function should at least designate the list of observation modalities that are image / low-dimensional observations by calling robomimic.utils.obs_utils.initialize_obs_utils_with_obs_specs().
        pass

    def get_goal(self):
        # (Optional) Get goal for a goal-conditional task
        pass

    def set_goal(self, goal):
        # (optional) Set goal with external specification
        pass

    def get_state(self):
        # (Optional) This function should return the underlying state of a simulated environment. Should be compatible with reset_to.
        pass

    def reset_to(self, state):
        # (Optional) Reset to a specific simulator state. Useful for reproducing results.
        pass

    @property
    def name(self):
        """
        Returns name of environment name (str).
        """
        return self._env_name

    @property
    def type(self):
        """
        Returns environment type (int) for this kind of environment.
        This helps identify this env class.
        """
        return EB.EnvType.RAVENS_TYPE

    @property
    def action_dimension(self):
        pass

    @property
    def rollout_exceptions(self):
        """
        Return tuple of exceptions to except when doing rollouts. This is useful to ensure
        that the entire training run doesn't crash because of a bad policy that causes unstable
        simulation computations.
        """
        return () #TODO(VS)