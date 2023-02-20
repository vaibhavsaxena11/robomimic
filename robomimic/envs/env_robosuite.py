"""
This file contains the robosuite environment wrapper that is used
to provide a standardized environment API for training policies and interacting
with metadata present in datasets.
"""
import json
import numpy as np
from copy import deepcopy

import mujoco_py
import robosuite
from robosuite.utils.mjcf_utils import postprocess_model_xml

import robomimic.utils.obs_utils as ObsUtils
import robomimic.envs.env_base as EB
from robomimic.utils.geometry import get_xyz_from_depth


class EnvRobosuite(EB.EnvBase):
    """Wrapper class for robosuite environments (https://github.com/ARISE-Initiative/robosuite)"""
    def __init__(
        self, 
        env_name, 
        render=False, 
        render_offscreen=False, 
        use_image_obs=False, 
        postprocess_visual_obs=True, 
        **kwargs,
    ):
        """
        Args:
            env_name (str): name of environment. Only needs to be provided if making a different
                environment from the one in @env_meta.

            render (bool): if True, environment supports on-screen rendering

            render_offscreen (bool): if True, environment supports off-screen rendering. This
                is forced to be True if @env_meta["use_images"] is True.

            use_image_obs (bool): if True, environment is expected to render rgb image observations
                on every env.step call. Set this to False for efficiency reasons, if image
                observations are not required.

            postprocess_visual_obs (bool): if True, postprocess image observations
                to prepare for learning. This should only be False when extracting observations
                for saving to a dataset (to save space on RGB images for example).
        """
        self.postprocess_visual_obs = postprocess_visual_obs

        # robosuite version check
        self._is_v1 = (robosuite.__version__.split(".")[0] == "1")
        if self._is_v1:
            assert (int(robosuite.__version__.split(".")[1]) >= 2), "only support robosuite v0.3 and v1.2+"

        kwargs = deepcopy(kwargs)

        # update kwargs based on passed arguments
        update_kwargs = dict(
            has_renderer=render,
            has_offscreen_renderer=(render_offscreen or use_image_obs),
            ignore_done=True,
            use_object_obs=True,
            use_camera_obs=use_image_obs,
            # camera_depths=False, ## TODO(VS) check why this was always set to False by default
        )
        kwargs.update(update_kwargs)

        if self._is_v1:
            if kwargs["has_offscreen_renderer"]:
                # ensure that we select the correct GPU device for rendering by testing for EGL rendering
                # NOTE: this package should be installed from this link (https://github.com/StanfordVL/egl_probe)
                import egl_probe
                valid_gpu_devices = egl_probe.get_available_devices()
                if len(valid_gpu_devices) > 0:
                    kwargs["render_gpu_device_id"] = valid_gpu_devices[0]
        else:
            # make sure gripper visualization is turned off (we almost always want this for learning)
            kwargs["gripper_visualization"] = False
            del kwargs["camera_depths"]
            kwargs["camera_depth"] = False # rename kwarg

        self._env_name = env_name
        self._init_kwargs = deepcopy(kwargs)
        self.env = robosuite.make(self._env_name, **kwargs)

        if self._is_v1:
            # Make sure joint position observations and eef vel observations are active
            for ob_name in self.env.observation_names:
                if ("joint_pos" in ob_name) or ("eef_vel" in ob_name):
                    self.env.modify_observable(observable_name=ob_name, attribute="active", modifier=True)

    def step(self, action):
        """
        Step in the environment with an action.

        Args:
            action (np.array): action to take

        Returns:
            observation (dict): new observation dictionary
            reward (float): reward for this step
            done (bool): whether the task is done
            info (dict): extra information
        """
        obs, r, done, info = self.env.step(action)
        obs = self.get_observation(obs)
        return obs, r, self.is_done(), info

    def reset(self):
        """
        Reset environment.

        Returns:
            observation (dict): initial observation dictionary.
        """
        di = self.env.reset()
        return self.get_observation(di)

    def reset_to(self, state):
        """
        Reset to a specific simulator state.

        Args:
            state (dict): current simulator state that contains one or more of:
                - states (np.ndarray): initial state of the mujoco environment
                - model (str): mujoco scene xml
        
        Returns:
            observation (dict): observation dictionary after setting the simulator state (only
                if "states" is in @state)
        """
        should_ret = False
        if "model" in state:
            self.reset()
            xml = postprocess_model_xml(state["model"])
            self.env.reset_from_xml_string(xml)
            self.env.sim.reset()
            if not self._is_v1:
                # hide teleop visualization after restoring from model
                self.env.sim.model.site_rgba[self.env.eef_site_id] = np.array([0., 0., 0., 0.])
                self.env.sim.model.site_rgba[self.env.eef_cylinder_id] = np.array([0., 0., 0., 0.])
        if "states" in state:
            self.env.sim.set_state_from_flattened(state["states"])
            self.env.sim.forward()
            should_ret = True

        if "goal" in state:
            self.set_goal(**state["goal"])
        if should_ret:
            # only return obs if we've done a forward call - otherwise the observations will be garbage
            return self.get_observation()
        return None

    ## NOTE(VS) code from https://github.com/ARISE-Initiative/robosuite/blob/master/robosuite/utils/mjcf_utils.py#L852
    ## helper function to get instance ids from geom ids
    ## TODO(VS) get this function away from here! This exists in robosuite's master but not in offline_study branch.
    def get_ids(self, sim, elements, element_type="geom", inplace=False):
        """
        Grabs the mujoco IDs for each element in @elements, corresponding to the specified @element_type.
        Args:
            sim (MjSim): Active mujoco simulation object
            elements (str or list or dict): Element(s) to convert into IDs. Note that the return type corresponds to
                @elements type, where each element name is replaced with the ID
            element_type (str): The type of element to grab ID for. Options are {geom, body, site}
            inplace (bool): If False, will create a copy of @elements to prevent overwriting the original data structure
        Returns:
            str or list or dict: IDs corresponding to @elements.
        """
        from collections.abc import Iterable
        if not inplace:
            # Copy elements first so we don't write to the underlying object
            elements = deepcopy(elements)
        # Choose what to do based on elements type
        if isinstance(elements, str):
            # We simply return the value of this single element
            assert element_type in {
                "geom",
                "body",
                "site",
            }, f"element_type must be either geom, body, or site. Got: {element_type}"
            if element_type == "geom":
                elements = sim.model.geom_name2id(elements)
            elif element_type == "body":
                elements = sim.model.body_name2id(elements)
            else:  # site
                elements = sim.model.site_name2id(elements)
        elif isinstance(elements, dict):
            # Iterate over each element in dict and recursively repeat
            for name, ele in elements:
                elements[name] = self.get_ids(sim=sim, elements=ele, element_type=element_type, inplace=True)
        else:  # We assume this is an iterable array
            assert isinstance(elements, Iterable), "Elements must be iterable for get_id!"
            elements = [self.get_ids(sim=sim, elements=ele, element_type=element_type, inplace=True) for ele in elements]

        return elements

    ## NOTE(VS) code from https://github.com/ARISE-Initiative/robosuite/blob/master/robosuite/models/tasks/task.py#L91
    ## TODO(VS) get this function away from here! This exists in robosuite's master but not in offline_study branch.
    def generate_id_mappings(self, sim):
        """
        Generates IDs mapping class instances to set of (visual) geom IDs corresponding to that class instance
        Args:
            sim (MjSim): Current active mujoco simulation object
        """
        self._instances_to_ids = {}
        self._geom_ids_to_instances = {}
        self._site_ids_to_instances = {}
        self._classes_to_ids = {}
        self._geom_ids_to_classes = {}
        self._site_ids_to_classes = {}

        models = [model for model in self.env.model.mujoco_objects] ##
        def _get_robot_models(robot):
            models = [robot.mount] if robot.mount is not None else [] ## NOTE(VS) scraped from https://github.com/ARISE-Initiative/robosuite/blob/master/robosuite/models/robots/robot_model.py#L190
            return models + list(robot.grippers.values()) ## NOTE(VS) scraped from https://github.com/ARISE-Initiative/robosuite/blob/master/robosuite/models/robots/manipulators/manipulator_model.py#L83
        for robot in self.env.model.mujoco_robots: ##
            # models += [robot] + robot.models ##
            models += [robot] + _get_robot_models(robot) ##

        # Parse all mujoco models from robots and objects
        for model in models:
            # Grab model class name and visual IDs
            cls = str(type(model)).split("'")[1].split(".")[-1]
            inst = model.name
            id_groups = [
                self.get_ids(sim=sim, elements=model.visual_geoms + model.contact_geoms, element_type="geom"),
                self.get_ids(sim=sim, elements=model.sites, element_type="site"),
            ]
            group_types = ("geom", "site")
            ids_to_instances = (self._geom_ids_to_instances, self._site_ids_to_instances)
            ids_to_classes = (self._geom_ids_to_classes, self._site_ids_to_classes)

            # Add entry to mapping dicts

            # Instances should be unique
            assert inst not in self._instances_to_ids, f"Instance {inst} already registered; should be unique"
            self._instances_to_ids[inst] = {}

            # Classes may not be unique
            if cls not in self._classes_to_ids:
                self._classes_to_ids[cls] = {group_type: [] for group_type in group_types}

            for ids, group_type, ids_to_inst, ids_to_cls in zip(
                id_groups, group_types, ids_to_instances, ids_to_classes
            ):
                # Add geom, site ids
                self._instances_to_ids[inst][group_type] = ids
                self._classes_to_ids[cls][group_type] += ids

                # Add reverse mappings as well
                for idn in ids:
                    assert idn not in ids_to_inst, f"ID {idn} already registered; should be unique"
                    ids_to_inst[idn] = inst
                    ids_to_cls[idn] = cls

    def render(self, mode="human", height=None, width=None, camera_name="agentview"):
        """
        Render from simulation to either an on-screen window or off-screen to RGB array.

        Args:
            mode (str): pass "human" for on-screen rendering or "rgb_array" for off-screen rendering
            height (int): height of image to render - only used if mode is "rgb_array"
            width (int): width of image to render - only used if mode is "rgb_array"
            camera_name (str): camera name to use for rendering
        """
        if mode == "human":
            cam_id = self.env.sim.model.camera_name2id(camera_name)
            self.env.viewer.set_camera(cam_id)
            return self.env.render()
        elif mode == "rgb_array":
            return self.env.sim.render(height=height, width=width, camera_name=camera_name)[::-1]
        else:
            raise NotImplementedError("mode={} is not implemented".format(mode))

    def get_observation(self, di=None, segmentation=True): 
        ## TODO(VS) (needs fixing for rollout testing) when calling this during rollout, set segmentation= whatever was in the dataset 
        ## currently, validation rollouts when dataset does not contain segmentation masks, will also segment out point clouds
        ## this will still give out 3 channels, i.e. no error during training, but the policy is used to seeing full point cloud
        """
        Get current environment observation dictionary.

        Args:
            di (dict): current raw observation dictionary from robosuite to wrap and provide 
                as a dictionary. If not provided, will be queried from robosuite.
        """
        if di is None:
            di = self.env._get_observations(force_update=True) if self._is_v1 else self.env._get_observation()
        ret = {}
        for k in di:
            if (k in ObsUtils.OBS_KEYS_TO_MODALITIES) and ObsUtils.key_is_obs_modality(key=k, obs_modality="rgb"):
                ret[k] = di[k][::-1]
                if self.postprocess_visual_obs:
                    ret[k] = ObsUtils.process_obs(obs=ret[k], obs_key=k)

        # "object" key contains object information
        ret["object"] = np.array(di["object-state"])

        if self._is_v1:
            for robot in self.env.robots:
                # add all robot-arm-specific observations. Note the (k not in ret) check
                # ensures that we don't accidentally add robot wrist images a second time
                pf = robot.robot_model.naming_prefix
                for k in di:
                    if k.startswith(pf) and (k not in ret) and (not k.endswith("proprio-state")):
                        ret[k] = np.array(di[k])
        else:
            # minimal proprioception for older versions of robosuite
            ret["proprio"] = np.array(di["robot-state"])
            ret["eef_pos"] = np.array(di["eef_pos"])
            ret["eef_quat"] = np.array(di["eef_quat"])
            ret["gripper_qpos"] = np.array(di["gripper_qpos"])

        # Adding depth and 3D (x,y,z) points to observation dict from each camera.
        #NOTE(VS)# obs keys are: any keys in ObsUtils.OBS_KEYS_TO_MODALITIES i.e. are images, object-state, and anything 
        ## that comes from robot{i} (in that order), hence agentview_depth is skipped until here, but included in the next block
        ## TODO(VS) cleanup comment
        for cam_name, cam_height, cam_width in zip(self.env.camera_names, self.env.camera_heights, self.env.camera_widths):
            if f"{cam_name}_depth" in di:
                if f"{cam_name}_depth" not in ret:
                    ret[f"{cam_name}_depth"] = di[f"{cam_name}_depth"]
                ret[f"{cam_name}_xyz"] = get_xyz_from_depth(
                    self.get_real_depth_map(ret[f"{cam_name}_depth"]).squeeze(), 
                    self.get_camera_intrinsic_matrix(cam_name, cam_height, cam_width), 
                    cam_height, cam_width
                )
                # if self.postprocess_visual_obs:
                #     ret[f"{cam_name}_xyz"] = ObsUtils.process_obs(obs=ret[f"{cam_name}_xyz"], obs_modality='pcd')
        #TODO(VS) maybe remove robot0_eye_in_hand_xyz and _depth from returned dict so as to reduce dataset size?
        
        if segmentation:
            self.generate_id_mappings(self.env.sim)
            ## NOTE(VS): most code from here: https://github.com/ARISE-Initiative/robosuite/blob/master/robosuite/environments/robot_env.py#L446
            from robosuite.utils.mjcf_utils import IMAGE_CONVENTION_MAPPING
            import robosuite.utils.macros as macros
            convention = IMAGE_CONVENTION_MAPPING[macros.IMAGE_CONVENTION]
            cam_s = "instance" # "class"
            if cam_s == "instance":
                name2id = {inst: i for i, inst in enumerate(list(self._instances_to_ids.keys()))} ##
                mapping = {idn: name2id[inst] for idn, inst in self._geom_ids_to_instances.items()} ##
                # print(name2id) # TODO(VS) cleanup
                # print(mapping)
            elif cam_s == "class":
                name2id = {cls: i for i, cls in enumerate(list(self._classes_to_ids.keys()))}
                mapping = {idn: name2id[cls] for idn, cls in self._geom_ids_to_classes.items()}
            else:  # element
                # No additional mapping needed
                mapping = None
                assert mapping is not None
    
            for cam_name, cam_height, cam_width in zip(self.env.camera_names, self.env.camera_heights, self.env.camera_widths):
                seg = self.env.sim.render(
                    camera_name=cam_name,
                    width=cam_width,
                    height=cam_height,
                    depth=False,
                    segmentation=True,
                )
                seg = np.expand_dims(seg[::convention, :, 1], axis=-1)
                # Map raw IDs to grouped IDs if we're using instance or class-level segmentation
                seg = (
                    np.fromiter(map(lambda x: mapping.get(x, -1), seg.flatten()), dtype=np.int32).reshape(
                        cam_height, cam_width, 1
                    )
                    + 1
                ) # seg mask with 0 entry refers to points not belonging to any model (object or robot)
                seg = (seg>0).astype(np.float)
                # Concatenate seg mask as the last channel of point cloud
                ret[f"{cam_name}_xyz"] = np.concatenate([ret[f"{cam_name}_xyz"], seg], -1)
        else:
            for cam_name, cam_height, cam_width in zip(self.env.camera_names, self.env.camera_heights, self.env.camera_widths):
                ret[f"{cam_name}_xyz"] = np.concatenate([ret[f"{cam_name}_xyz"], np.ones([cam_height, cam_width, 1])], -1)
            
        if self.postprocess_visual_obs:
            for cam_name in self.env.camera_names:
                ret[f"{cam_name}_xyz"] = ObsUtils.process_obs(obs=ret[f"{cam_name}_xyz"], obs_modality='pcd')

        return ret

    def get_real_depth_map(self, depth_map):
        """
        Source: https://github.com/RoboTurk-Platform/robomimic-internal/blob/benchmark/batchRL/envs/env_robosuite.py#L342
        TODO(VS) doc string
        TODO(VS) add unit tests

        By default, robosuite will return a depth map that is normalized in [0, 1]. This
        helper function converts the map so that the entries correspond to actual distances.
        (see https://github.com/deepmind/dm_control/blob/master/dm_control/mujoco/engine.py#L742)
        """
        # NOTE: we assume this hasn't happened yet in robosuite internally, and assert
        #       that all entries are in [0, 1]
        assert np.all(depth_map >= 0.) and np.all(depth_map <= 1.)
        extent = self.env.sim.model.stat.extent
        far = self.env.sim.model.vis.map.zfar * extent
        near = self.env.sim.model.vis.map.znear * extent
        return near / (1. - depth_map * (1. - near / far))

    def get_camera_intrinsic_matrix(self, camera_name, camera_height, camera_width):
        """
        Source: https://github.com/RoboTurk-Platform/robomimic-internal/blob/benchmark/batchRL/envs/env_robosuite.py#L296
        TODO(VS) this code exists in robosuite/master/utils/camera_utils.py; see if it can go in in the offline_study branch to prevent code duplication
        TODO(VS) doc string
        TODO(VS) add unit tests
        
        Obtains camera internal matrix from other parameters. A 3X3 matrix.
        """
        cam_id = self.env.sim.model.camera_name2id(camera_name)
        fovy = self.env.sim.model.cam_fovy[cam_id]
        f = 0.5 * camera_height / np.tan(fovy * np.pi / 360)
        K = np.array([[f, 0, camera_width / 2], [0, f, camera_height / 2], [0, 0, 1]])
        return K

    def get_state(self):
        """
        Get current environment simulator state as a dictionary. Should be compatible with @reset_to.
        """
        xml = self.env.sim.model.get_xml() # model xml file
        state = np.array(self.env.sim.get_state().flatten()) # simulator state
        return dict(model=xml, states=state)

    def get_reward(self):
        """
        Get current reward.
        """
        return self.env.reward()

    def get_goal(self):
        """
        Get goal observation. Not all environments support this.
        """
        return self.get_observation(self.env._get_goal())

    def set_goal(self, **kwargs):
        """
        Set goal observation with external specification. Not all environments support this.
        """
        return self.env.set_goal(**kwargs)

    def is_done(self):
        """
        Check if the task is done (not necessarily successful).
        """

        # Robosuite envs always rollout to fixed horizon.
        return False

    def is_success(self):
        """
        Check if the task condition(s) is reached. Should return a dictionary
        { str: bool } with at least a "task" key for the overall task success,
        and additional optional keys corresponding to other task criteria.
        """
        succ = self.env._check_success()
        if isinstance(succ, dict):
            assert "task" in succ
            return succ
        return { "task" : succ }

    @property
    def action_dimension(self):
        """
        Returns dimension of actions (int).
        """
        return self.env.action_spec[0].shape[0]

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
        return EB.EnvType.ROBOSUITE_TYPE

    def serialize(self):
        """
        Save all information needed to re-instantiate this environment in a dictionary.
        This is the same as @env_meta - environment metadata stored in hdf5 datasets,
        and used in utils/env_utils.py.
        """
        return dict(env_name=self.name, type=self.type, env_kwargs=deepcopy(self._init_kwargs))

    @classmethod
    def create_for_data_processing(
        cls, 
        env_name, 
        camera_names, 
        camera_height, 
        camera_width, 
        reward_shaping, 
        **kwargs,
    ):
        """
        Create environment for processing datasets, which includes extracting
        observations, labeling dense / sparse rewards, and annotating dones in
        transitions. 

        Args:
            env_name (str): name of environment
            camera_names (list of str): list of camera names that correspond to image observations
            camera_height (int): camera height for all cameras
            camera_width (int): camera width for all cameras
            reward_shaping (bool): if True, use shaped environment rewards, else use sparse task completion rewards
        """
        is_v1 = (robosuite.__version__.split(".")[0] == "1")
        has_camera = (len(camera_names) > 0)

        new_kwargs = {
            "reward_shaping": reward_shaping,
        }

        if has_camera:
            if is_v1:
                new_kwargs["camera_names"] = list(camera_names)
                new_kwargs["camera_heights"] = camera_height
                new_kwargs["camera_widths"] = camera_width
            else:
                assert len(camera_names) == 1
                if has_camera:
                    new_kwargs["camera_name"] = camera_names[0]
                    new_kwargs["camera_height"] = camera_height
                    new_kwargs["camera_width"] = camera_width

        kwargs.update(new_kwargs)

        # also initialize obs utils so it knows which modalities are image modalities
        image_modalities = list(camera_names)
        if is_v1:
            image_modalities = ["{}_image".format(cn) for cn in camera_names]
        elif has_camera:
            # v0.3 only had support for one image, and it was named "rgb"
            assert len(image_modalities) == 1
            image_modalities = ["rgb"]
        obs_modality_specs = {
            "obs": {
                "low_dim": [], # technically unused, so we don't have to specify all of them
                "rgb": image_modalities,
            }
        }
        ObsUtils.initialize_obs_utils_with_obs_specs(obs_modality_specs)

        # note that @postprocess_visual_obs is False since this env's images will be written to a dataset
        return cls(
            env_name=env_name,
            render=False, 
            render_offscreen=has_camera, 
            use_image_obs=has_camera, 
            postprocess_visual_obs=False,
            **kwargs,
        )

    @property
    def rollout_exceptions(self):
        """
        Return tuple of exceptions to except when doing rollouts. This is useful to ensure
        that the entire training run doesn't crash because of a bad policy that causes unstable
        simulation computations.
        """
        return (mujoco_py.builder.MujocoException)

    def __repr__(self):
        """
        Pretty-print env description.
        """
        return self.name + "\n" + json.dumps(self._init_kwargs, sort_keys=True, indent=4)
