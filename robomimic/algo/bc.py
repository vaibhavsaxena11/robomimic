"""
Implementation of Behavioral Cloning (BC).
"""
import numpy as np
from collections import OrderedDict
import kornia
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

import robomimic.models.base_nets as BaseNets
import robomimic.models.obs_nets as ObsNets
import robomimic.models.policy_nets as PolicyNets
import robomimic.models.vae_nets as VAENets
import robomimic.utils.loss_utils as LossUtils
import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.torch_utils as TorchUtils
import robomimic.utils.obs_utils as ObsUtils

from robomimic.algo import register_algo_factory_func, PolicyAlgo


@register_algo_factory_func("bc")
def algo_config_to_class(algo_config):
    """
    Maps algo config to the BC algo class to instantiate, along with additional algo kwargs.

    Args:
        algo_config (Config instance): algo config

    Returns:
        algo_class: subclass of Algo
        algo_kwargs (dict): dictionary of additional kwargs to pass to algorithm
    """

    # note: we need the check below because some configs import BCConfig and exclude
    # some of these options
    gaussian_enabled = ("gaussian" in algo_config and algo_config.gaussian.enabled)
    gmm_enabled = ("gmm" in algo_config and algo_config.gmm.enabled)
    vae_enabled = ("vae" in algo_config and algo_config.vae.enabled)
    implicit_enabled = ("implicit" in algo_config and algo_config.implicit.enabled)

    if algo_config.rnn.enabled:
        if gmm_enabled:
            return BC_RNN_GMM, {}
        return BC_RNN, {}
    assert sum([gaussian_enabled, gmm_enabled, vae_enabled, implicit_enabled]) <= 1
    if gaussian_enabled:
        return BC_Gaussian, {}
    if gmm_enabled:
        return BC_GMM, {}
    if vae_enabled:
        return BC_VAE, {}
    if implicit_enabled:
        return BC_Implicit, {}
    return BC, {}


class BC(PolicyAlgo):
    """
    Normal BC training.
    """
    def _create_networks(self):
        """
        Creates networks and places them into @self.nets.
        """
        self.nets = nn.ModuleDict()
        self.nets["policy"] = PolicyNets.ActorNetwork(
            obs_shapes=self.obs_shapes,
            goal_shapes=self.goal_shapes,
            ac_dim=self.ac_dim,
            mlp_layer_dims=self.algo_config.actor_layer_dims,
            encoder_kwargs=ObsUtils.obs_encoder_kwargs_from_config(self.obs_config.encoder),
        )
        self.nets = self.nets.float().to(self.device)

    def process_batch_for_training(self, batch):
        """
        Processes input batch from a data loader to filter out
        relevant information and prepare the batch for training.

        Args:
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader

        Returns:
            input_batch (dict): processed and filtered batch that
                will be used for training 
        """
        input_batch = dict()
        ## NOTE(VS): picking first item in the sequence for training BC
        ## expected sequence length 1
        input_batch["obs"] = {k: batch["obs"][k][:, 0, :] for k in batch["obs"]}
        input_batch["goal_obs"] = batch.get("goal_obs", None) # goals may not be present
        input_batch["actions"] = batch["actions"][:, 0, :]
        return TensorUtils.to_device(TensorUtils.to_float(input_batch), self.device)

    def train_on_batch(self, batch, epoch, validate=False):
        """
        Training on a single batch of data.

        Args:
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader and filtered by @process_batch_for_training

            epoch (int): epoch number - required by some Algos that need
                to perform staged training and early stopping

            validate (bool): if True, don't perform any learning updates.

        Returns:
            info (dict): dictionary of relevant inputs, outputs, and losses
                that might be relevant for logging
        """
        with TorchUtils.maybe_no_grad(no_grad=validate):
            info = super(BC, self).train_on_batch(batch, epoch, validate=validate)
            predictions = self._forward_training(batch)
            losses = self._compute_losses(predictions, batch)

            info["predictions"] = TensorUtils.detach(predictions)
            info["losses"] = TensorUtils.detach(losses)

            if not validate:
                step_info = self._train_step(losses)
                info.update(step_info)

        return info

    def _forward_training(self, batch):
        """
        Internal helper function for BC algo class. Compute forward pass
        and return network outputs in @predictions dict.

        Args:
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader and filtered by @process_batch_for_training

        Returns:
            predictions (dict): dictionary containing network outputs
        """
        predictions = OrderedDict()
        actions = self.nets["policy"](obs_dict=batch["obs"], goal_dict=batch["goal_obs"])
        predictions["actions"] = actions
        return predictions

    def _compute_losses(self, predictions, batch):
        """
        Internal helper function for BC algo class. Compute losses based on
        network outputs in @predictions dict, using reference labels in @batch.

        Args:
            predictions (dict): dictionary containing network outputs, from @_forward_training
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader and filtered by @process_batch_for_training

        Returns:
            losses (dict): dictionary of losses computed over the batch
        """
        losses = OrderedDict()
        a_target = batch["actions"]
        actions = predictions["actions"]
        losses["l2_loss"] = nn.MSELoss()(actions, a_target)
        losses["l1_loss"] = nn.SmoothL1Loss()(actions, a_target)
        # cosine direction loss on eef delta position
        losses["cos_loss"] = LossUtils.cosine_loss(actions[..., :3], a_target[..., :3])

        action_losses = [
            self.algo_config.loss.l2_weight * losses["l2_loss"],
            self.algo_config.loss.l1_weight * losses["l1_loss"],
            self.algo_config.loss.cos_weight * losses["cos_loss"],
        ]
        action_loss = sum(action_losses)
        losses["action_loss"] = action_loss
        return losses

    def _train_step(self, losses):
        """
        Internal helper function for BC algo class. Perform backpropagation on the
        loss tensors in @losses to update networks.

        Args:
            losses (dict): dictionary of losses computed over the batch, from @_compute_losses
        """

        # gradient step
        info = OrderedDict()
        policy_grad_norms = TorchUtils.backprop_for_loss(
            net=self.nets["policy"],
            optim=self.optimizers["policy"],
            loss=losses["action_loss"],
        )
        info["policy_grad_norms"] = policy_grad_norms
        return info

    def log_info(self, info):
        """
        Process info dictionary from @train_on_batch to summarize
        information to pass to tensorboard for logging.

        Args:
            info (dict): dictionary of info

        Returns:
            loss_log (dict): name -> summary statistic
        """
        log = super(BC, self).log_info(info)
        log["Loss"] = info["losses"]["action_loss"].item()
        if "l2_loss" in info["losses"]:
            log["L2_Loss"] = info["losses"]["l2_loss"].item()
        if "l1_loss" in info["losses"]:
            log["L1_Loss"] = info["losses"]["l1_loss"].item()
        if "cos_loss" in info["losses"]:
            log["Cosine_Loss"] = info["losses"]["cos_loss"].item()
        if "policy_grad_norms" in info:
            log["Policy_Grad_Norms"] = info["policy_grad_norms"]
        return log

    def get_action(self, obs_dict, goal_dict=None):
        """
        Get policy action outputs.

        Args:
            obs_dict (dict): current observation
            goal_dict (dict): (optional) goal

        Returns:
            action (torch.Tensor): action tensor
        """
        assert not self.nets.training
        return self.nets["policy"](obs_dict, goal_dict=goal_dict)


class BC_Implicit(BC):
    """
    Implicit Behavior Cloning.
    """
    def _create_networks(self):
        """
        Creates networks and places them into @self.nets.
        """
        self.nets = nn.ModuleDict()
        # self.nets["policy"] = PolicyNets.AffordanceNetwork(
        #     obs_shapes=self.obs_shapes,
        #     goal_shapes=self.goal_shapes,
        #     feat_dim=self.algo_config.implicit.feat_dim, # added 4 for predicting orientation directly from the MIMO_MLP
        #     ac_implicit_dim=3,#self.ac_dim,#2,#3, #TODO(VS) make use of self.ac_dim for ac_implicit_dim and ac_explicit_dim
        #     ac_explicit_dim=4,
        #     mlp_layer_dims=self.algo_config.actor_layer_dims,
        #     affordance_layer_dims=self.algo_config.implicit.affordance_layer_dims,
        #     activation=nn.Sigmoid, #nn.Identity,
        #     encoder_kwargs=ObsUtils.obs_encoder_kwargs_from_config(self.obs_config.encoder),
        # )
        self.ac_implicit_dim = 2 # 3
        self.nets["pick_policy"] = PolicyNets.AffordanceNetworkLite(
            obs_shapes=self.obs_shapes,
            goal_shapes=self.goal_shapes,
            feat_dim=self.algo_config.implicit.feat_dim, # added 4 for predicting orientation directly from the MIMO_MLP
            ac_implicit_dim=self.ac_implicit_dim,#3,#self.ac_dim,#2,#3, #TODO(VS) make use of self.ac_dim for ac_implicit_dim and ac_explicit_dim
            ac_explicit_dim=4,
            mlp_layer_dims=self.algo_config.actor_layer_dims,
            affordance_layer_dims=self.algo_config.implicit.affordance_layer_dims,
            activation=nn.Sigmoid,#nn.ReLU,#, #nn.Identity,
            # activation=nn.ReLU,#nn.ReLU,#, #nn.Identity,
            # pad_images=self.algo_config.implicit.pad_images,
            encoder_kwargs=ObsUtils.obs_encoder_kwargs_from_config(self.obs_config.encoder),
        )
        self.nets["place_policy"] = PolicyNets.AffordanceNetworkLite(
            obs_shapes=self.obs_shapes,
            goal_shapes=self.goal_shapes,
            feat_dim=self.algo_config.implicit.feat_dim, # added 4 for predicting orientation directly from the MIMO_MLP
            ac_implicit_dim=self.ac_implicit_dim,#3,#self.ac_dim,#2,#3, #TODO(VS) make use of self.ac_dim for ac_implicit_dim and ac_explicit_dim
            ac_explicit_dim=4,
            mlp_layer_dims=self.algo_config.actor_layer_dims,
            affordance_layer_dims=self.algo_config.implicit.affordance_layer_dims,
            activation=nn.Sigmoid,#nn.ReLU,#, #nn.Identity,
            # activation=nn.ReLU,#nn.ReLU,#, #nn.Identity,
            # pad_images=self.algo_config.implicit.pad_images,
            encoder_kwargs=ObsUtils.obs_encoder_kwargs_from_config(self.obs_config.encoder),
        )
        self.nets = self.nets.float().to(self.device)

    def train_on_batch(self, batch, epoch, validate=False):
        """
        Training on a single batch of data.

        Args:
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader and filtered by @process_batch_for_training

            epoch (int): epoch number - required by some Algos that need
                to perform staged training and early stopping

            validate (bool): if True, don't perform any learning updates.

        Returns:
            info (dict): dictionary of relevant inputs, outputs, and losses
                that might be relevant for logging
        """
        with TorchUtils.maybe_no_grad(no_grad=validate):
            info = super(BC, self).train_on_batch(batch, epoch, validate=validate)
            losses = self._compute_losses(batch)

            info["losses"] = TensorUtils.detach(losses)

            if not validate:
                step_info = self._train_step(losses)
                info.update(step_info)

        return info

    def _create_aff_grid(self, batch, policy_type, policy_dim="2d"):
        """
        Helper for _compute_implicit_repro_loss()
        """
        if policy_dim == "3d": #TODO(VS)
            return NotImplementedError
            # creating 3D action grid
            actions = (np.mgrid[0:20, -10:10, 0:4]/20).transpose([1,2,3,0]).astype(np.float32) # shape (20, 20, 4, 3)
            actions = actions.reshape([-1, 3])
        else:
            # returning affordance probe 
            if self.algo_config.implicit.pad_images:
                # creating 2D action grid of size (180, 60, 2)

                # # below is for padded images on the x dimension
                gridsize_j, gridsize_i = 160, 160
                actions = (np.mgrid[0:gridsize_i, 0:gridsize_j]).transpose([2,1,0]).astype(np.float32) # shape (gridsize_j, gridsize_i, 2)
                actions[..., 0] = 0.1 + 0.8*(actions[..., 0]/gridsize_i) # x in [0.1, 0.9]
                actions[..., 1] = -0.5 + (actions[..., 1]/gridsize_j) # y in [-0.5, 0.5]
                actions = actions.reshape([-1, 2])
            else: 
                gridsize_j, gridsize_i = 160, 80
                actions = (np.mgrid[0:gridsize_i, 0:gridsize_j]).transpose([2,1,0]).astype(np.float32) # shape (gridsize_j, gridsize_i, 2)
                actions[..., 0] = 0.3 + 0.4*(actions[..., 0]/gridsize_i) # x in [0.3, 0.7]
                actions[..., 1] = -0.5 + (actions[..., 1]/gridsize_j) # y in [-0.5, 0.5]
                actions = actions.reshape([-1, 2])

        # tiling grid to batch size
        actions = torch.tensor(np.expand_dims(actions, 0))
        batch_size = batch["obs"][list(batch["obs"].keys())[0]].shape[0]
        actions = actions.tile([batch_size, 1, 1]).to(self.device)
        affordances = self.nets[f"{policy_type}_policy"](batch["obs"], actions, batch["goal_obs"])[0]

        if policy_dim == "3d": #TODO(VS)
            # return torch.cat([actions, affordances], -1) # 3d only #TODO(VS)
            return NotImplementedError
        else:
            return torch.reshape(affordances, [batch_size, gridsize_j, gridsize_i]), torch.reshape(actions, [batch_size, gridsize_j, gridsize_i, 2])

    def _compute_implicit_repro_loss(self, batch, losses):
        """
        Helper for _compute_losses().
        Compute MSE loss between the image generated using the affordance map, and the input image.
        """
        for ac in ["pick", "place"]:
            aff, _ = self._create_aff_grid(batch, policy_type=ac, policy_dim=f"{self.ac_implicit_dim}d")

            imgs = torch.mean(batch["obs"][list(batch["obs"].keys())[0]], 1) # avg across all channels
            losses[f"{ac}_affordance_recon_loss"] = nn.MSELoss()(aff, imgs) # summing all channels
            #TODO(VS) handle more obs keys?

            losses[f"{ac}_policy_action_loss"] += losses[f"{ac}_affordance_recon_loss"]
            losses["action_loss"] += losses[f"{ac}_affordance_recon_loss"]

    def _rotate_inputs(self, in_batch, pivot, n_rotations, reverse=False):
        """
        Used for data augmentation in _compute_losses()
        """
        angles = []
        for i in range(n_rotations):
            theta = i * 2 * 180 / n_rotations
            angles.append(theta)

        batch_size = in_batch.shape[0]
        rot_x_list = []
        for i, angle in enumerate(angles):
            x = in_batch#[i].unsqueeze(0)

            # create transformation (rotation)
            alpha: float = angle if not reverse else (-1.0 * angle)  # in degrees
            angle: torch.tensor = torch.ones(batch_size) * alpha

            # define the scale factor
            scale: torch.tensor = torch.ones(batch_size, 2)

            # define the rotation center
            center: torch.tensor = torch.ones(batch_size, 2)
            center[..., 0] = pivot[1]
            center[..., 1] = pivot[0]


            # compute the transformation matrix
            M: torch.tensor = kornia.geometry.transform.get_rotation_matrix2d(center, angle, scale)

            # apply the transformation to original image
            if len(x.shape) == 4: # rotate image
                # define the rotation center
                center: torch.tensor = torch.ones(batch_size, 2)
                center[..., 0] = pivot[1]
                center[..., 1] = pivot[0]
                # compute the transformation matrix
                M: torch.tensor = kornia.geometry.transform.get_rotation_matrix2d(center, angle, scale)

                _, _, h, w = x.shape
                x_warped: torch.tensor = kornia.geometry.transform.warp_affine(x.float(), M.to(x.device), dsize=(h, w))
                # x_warped = x_warped #TODO(VS) check
            elif len(x.shape) == 2: # rotate vector
                # define the rotation center
                center: torch.tensor = torch.ones(batch_size, 2)
                center[..., 0] = pivot[0]
                center[..., 1] = pivot[1]
                # compute the transformation matrix
                M: torch.tensor = kornia.geometry.transform.get_rotation_matrix2d(center, angle, scale)

                b, _ = x.shape
                # import pdb; pdb.set_trace()
                x = torch.cat([x, torch.ones([b, 1]).to(x.device)], -1).unsqueeze(-1)
                x_warped = torch.bmm(M.to(x.device), x).squeeze(-1)
            else:
                raise NotImplementedError
            rot_x_list.append(x_warped)
        return torch.cat(rot_x_list, 0)

    def _compute_losses(self, batch):
        """
        Internal helper function for BC algo class. Compute losses based on
        network outputs in @predictions dict, using reference labels in @batch.

        Args:
            predictions (dict): dictionary containing network outputs, from @_forward_training
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader and filtered by @process_batch_for_training

        Returns:
            losses (dict): dictionary of losses computed over the batch
        """
        ### TODO(VS) cleanup

        ### rotating images
        batch = copy.deepcopy(batch) # TODO(VS) this should make sure that the original batch doesn't change, check!! this affects affordance logging
        n_rotations = 10
        for k in batch["obs"]:
            if k in ObsUtils.OBS_MODALITIES_TO_KEYS["rgb"]:
                img_rot_pivot = np.array(batch["obs"][k].shape[-2:]) // 2
                batch["obs"][k] = self._rotate_inputs(batch["obs"][k], img_rot_pivot, n_rotations)
        ###
        ### rotating actions
        act_rot_pivot = [0.5, 0] # hard-coded center (x,y) of table
        if self.ac_implicit_dim == 2:
            rotated_pick_tra = self._rotate_inputs(batch["actions"][:, :self.ac_implicit_dim], act_rot_pivot, n_rotations)
            rotated_plac_tra = self._rotate_inputs(batch["actions"][:, 7:7+self.ac_implicit_dim], act_rot_pivot, n_rotations)
            batch["actions"] = torch.tile(batch["actions"], [n_rotations, 1])
            batch["actions"][:, :self.ac_implicit_dim] = rotated_pick_tra
            batch["actions"][:, 7:7+self.ac_implicit_dim] = rotated_plac_tra
        else:
            # TODO(VS) rotation augmentation only works for 2D rotations, do for 3D rotations too
            raise NotImplementedError
        # TODO(VS) test rotated actions by plotting on rotated images
        ###

        # ###TODO(VS)(cleanup; remove) testing plots for rotated actions
        # import pdb; pdb.set_trace()
        # idx=5
        # import matplotlib.pyplot as plt
        # plt.imshow(batch['obs']['color_ortho'][idx].transpose(0,1).transpose(1,2))
        # act_x, act_y = batch['actions'][idx][:2][0], batch['actions'][idx][:2][1] # pick
        # i, j = int(40 + 80*((act_x - 0.3)/0.4)), int(80 + act_y*80) # pick
        # plt.plot((i,), (j,), 'o') # pick
        # act_x, act_y = batch['actions'][idx][7:7+2][0], batch['actions'][idx][7:7+2][1] # place
        # i, j = int(40 + 80*((act_x - 0.3)/0.4)), int(80 + act_y*80) # place
        # plt.plot((i,), (j,), 'x') # place
        # plt.show()
        # ###

        losses = OrderedDict()
        a_target = batch["actions"]

        a_target_pick_tra = a_target[:, :self.ac_implicit_dim]
        a_target_pick_ori = a_target[:, 3:7]
        a_target_plac_tra = a_target[:, 7:7+self.ac_implicit_dim]
        a_target_plac_ori = a_target[:, 10:]

        losses["correct_action_affordance_loss"] = 0.
        losses["wrong_action_affordance_loss"] = 0.
        losses["affordance_loss"] = 0.
        losses["l2_loss"] = 0.
        losses["action_loss"] = 0.
        for ac in ["pick", "place"]:
            if ac == "pick":
                a_implicit_target = a_target_pick_tra
                a_explicit_target = a_target_pick_ori
            else:
                a_implicit_target = a_target_plac_tra
                a_explicit_target = a_target_plac_ori

            # a_target = a_target[..., :2] ## 2D actions
            # # a_target[..., 0] = 0.5
            # # a_target[..., 1] = 0.5
            # # a_target = torch.zeros([a_target.shape[0], 2, a_target.shape[1]])
            # # a_target[:, 0, 0] = 0.3; a_target[:, 0, 1] = 0.3
            # # a_target[:, 1, 0] = 0.8; a_target[:, 1, 1] = 0.8

            affordances, act_explicit = self.nets[f"{ac}_policy"](batch["obs"], a_implicit_target, batch["goal_obs"])
            assert len(affordances.shape) == 2, affordances.shape
            affordances = affordances.unsqueeze(1)
            losses[f"correct_{ac}_action_affordance_loss"] = torch.mean(affordances)
            losses["correct_action_affordance_loss"] += losses[f"correct_{ac}_action_affordance_loss"]

            # losses["affordance_loss"] = nn.BCEWithLogitsLoss()(affordances, torch.ones([a_implicit_target.shape[0], 1]).to(self.device)) # BCE Loss


            #### Wrong actions logic
            ## obtaining negative action examples for infoNCE loss
            ## insight: robot always moves in +ve x, but both +/- y
            num_wrong_exs = 256 # 512 # 100 # 100 10 # 256 # 
            wrong_actions_posy = 1*torch.rand([a_implicit_target.shape[0], num_wrong_exs//2, a_implicit_target.shape[1]]).to(self.device)
            wrong_actions_negy = 1*torch.rand([a_implicit_target.shape[0], num_wrong_exs//2, a_implicit_target.shape[1]]).to(self.device)
            wrong_actions_negy[:,:,1] *= -1
            wrong_actions = torch.cat([wrong_actions_posy, wrong_actions_negy], 1)
            # import pdb; pdb.set_trace()
            # import matplotlib.pyplot as plt
            # plt.figure()
            # plt.scatter(wrong_actions[0,:,0], wrong_actions[0,:,1], c='red')
            # plt.scatter(a_implicit_target[:,0], a_implicit_target[:,1], c='blue')
            # plt.show()

            # # wrong_actions[:,:,2] = torch.zeros(wrong_actions.shape[:2]).to(self.device) # setting z=0 for all wrong actions, hopefully learns faster since most actions are on the surface
            
            # ## delta around positive actions is negative
            # num_wrong_exs = 100 # don't go too large (hunch)
            # # wrong_actions_delta = torch.rand([a_implicit_target.shape[0], num_wrong_exs, a_implicit_target.shape[1]]).to(self.device)/5.0 - 0.1 # \in [-0.1, 0.1]
            # wrong_actions_delta = 4*torch.rand([a_implicit_target.shape[0], num_wrong_exs, a_implicit_target.shape[1]]).to(self.device)/10.0 - 0.2 # \in [-0.2, 0.2]
            # wrong_actions = torch.tile(torch.unsqueeze(a_implicit_target, 1), [1, num_wrong_exs, 1]) + wrong_actions_delta
            ####


            # XXX # another way of computing wrong actions
            # action_deltas = torch.rand([a_implicit_target.shape[0], num_wrong_exs, a_implicit_target.shape[1]]).to(self.device)/5.0 - 0.1
            # wrong_actions = torch.tile(torch.unsqueeze(a_target_pick_tra, 1), [1, num_wrong_exs, 1]) + action_deltas
            # XXX #

            wrong_actions_affordances, _ = self.nets[f"{ac}_policy"](batch["obs"], wrong_actions, batch["goal_obs"])
            all_affordances = torch.cat([affordances, wrong_actions_affordances], 1)

            losses[f"wrong_{ac}_action_affordance_loss"] = torch.mean(torch.logsumexp(wrong_actions_affordances, 1))
            # losses["wrong_action_affordance_loss"] = torch.mean(torch.mean(wrong_actions_affordances, 1))

            losses["wrong_action_affordance_loss"] += losses[f"wrong_{ac}_action_affordance_loss"]

            # Computing total affordance loss on implicit actions. (Affordance is negative energy.)
            losses[f"{ac}_affordance_loss"] = torch.mean(- torch.sum(affordances, 1) + torch.logsumexp(all_affordances, 1)) # sum over correct-affordances to support multiple correct actions
            # losses[f"{ac}_affordance_loss"] = torch.mean(-affordances + torch.mean(wrong_actions_affordances, 1))
            losses["affordance_loss"] += losses[f"{ac}_affordance_loss"]

            # Computing loss on explicit actions.
            losses[f"{ac}_l2_loss"] = nn.MSELoss()(act_explicit, a_explicit_target)
            losses["l2_loss"] += losses[f"{ac}_l2_loss"]

            losses[f"{ac}_policy_action_loss"] = 0.
            if self.algo_config.implicit.L2explicit_loss_enabled: # for training EXPLICIT part of the policy
                losses[f"{ac}_policy_action_loss"] += losses[f"{ac}_l2_loss"]
            if self.algo_config.implicit.NCEimplicit_loss_enabled: # for training IMPLICIT part of the policy
                losses[f"{ac}_policy_action_loss"] += losses[f"{ac}_affordance_loss"]

            losses["action_loss"] += losses[f"{ac}_policy_action_loss"]

        if self.algo_config.implicit.L2recon_loss_enabled: # also for training IMPLICIT part of the policy
            self._compute_implicit_repro_loss(batch, losses)
        return losses

    def _train_step(self, losses):
        """
        Internal helper function for BC algo class. Perform backpropagation on the
        loss tensors in @losses to update networks.

        Args:
            losses (dict): dictionary of losses computed over the batch, from @_compute_losses
        """

        # gradient step
        info = OrderedDict()
        for k in self.nets:
            grad_norms = TorchUtils.backprop_for_loss(
                net=self.nets[k],
                optim=self.optimizers[k],
                loss=losses[f"{k}_action_loss"],
            )
            info[f"{k}_grad_norms"] = grad_norms
        return info

    def log_info(self, info):
        """
        Process info dictionary from @train_on_batch to summarize
        information to pass to tensorboard for logging.

        Args:
            info (dict): dictionary of info

        Returns:
            loss_log (dict): name -> summary statistic
        """
        log = super(BC, self).log_info(info)
        log["Loss"] = info["losses"]["action_loss"].item()
        if "l2_loss" in info["losses"]:
            log["L2_Loss"] = info["losses"]["l2_loss"].item()
        if "l1_loss" in info["losses"]:
            log["L1_Loss"] = info["losses"]["l1_loss"].item()
        if "cos_loss" in info["losses"]:
            log["Cosine_Loss"] = info["losses"]["cos_loss"].item()
        if "affordance_loss" in info["losses"]:
            log["Affordance_Loss"] = info["losses"]["affordance_loss"].item()
        if "correct_action_affordance_loss" in info["losses"]:
            log["Correct_Action_Affordance_Loss"] = info["losses"]["correct_action_affordance_loss"].item()
        if "wrong_action_affordance_loss" in info["losses"]:
            log["Wrong_Action_Affordance_Loss"] = info["losses"]["wrong_action_affordance_loss"].item()

        # Logging pick and place losses if they exist.
        # Pick
        if "pick_l2_loss" in info["losses"]:
            log["Pick_L2_Loss"] = info["losses"]["pick_l2_loss"].item()
        if "pick_affordance_loss" in info["losses"]:
            log["Pick_Affordance_Loss"] = info["losses"]["pick_affordance_loss"].item()
        if "correct_pick_action_affordance_loss" in info["losses"]:
            log["Correct_Pick_Action_Affordance_Loss"] = info["losses"]["correct_pick_action_affordance_loss"].item()
        if "wrong_pick_action_affordance_loss" in info["losses"]:
            log["Wrong_Pick_Action_Affordance_Loss"] = info["losses"]["wrong_pick_action_affordance_loss"].item()
        if "pick_affordance_recon_loss" in info["losses"]:
            log["Pick_Affordance_Recon_Loss"] = info["losses"]["pick_affordance_recon_loss"].item()
        if "pick_policy_action_loss" in info["losses"]:
            log["Pick_Action_Loss"] = info["losses"]["pick_policy_action_loss"].item()
        # Place
        if "place_l2_loss" in info["losses"]:
            log["Place_L2_Loss"] = info["losses"]["place_l2_loss"].item()
        if "place_affordance_loss" in info["losses"]:
            log["Place_Affordance_Loss"] = info["losses"]["place_affordance_loss"].item()
        if "correct_place_action_affordance_loss" in info["losses"]:
            log["Correct_Place_Action_Affordance_Loss"] = info["losses"]["correct_place_action_affordance_loss"].item()
        if "wrong_place_action_affordance_loss" in info["losses"]:
            log["Wrong_Place_Action_Affordance_Loss"] = info["losses"]["wrong_place_action_affordance_loss"].item()
        if "place_affordance_recon_loss" in info["losses"]:
            log["Place_Affordance_Recon_Loss"] = info["losses"]["place_affordance_recon_loss"].item()
        if "place_policy_action_loss" in info["losses"]:
            log["Place_Action_Loss"] = info["losses"]["place_policy_action_loss"].item()
        
        if "policy_grad_norms" in info:
            log["Policy_Grad_Norms"] = info["policy_grad_norms"]
        if "pick_policy_grad_norms" in info:
            log["Pick_Policy_Grad_Norms"] = info["pick_policy_grad_norms"]
        if "place_policy_grad_norms" in info:
            log["Place_Policy_Grad_Norms"] = info["place_policy_grad_norms"]

        return log

    def get_action(self, obs_dict, goal_dict=None):
        """
        Get policy action outputs.

        Args:
            obs_dict (dict): current observation
            goal_dict (dict): (optional) goal

        Returns:
            action (torch.Tensor): action tensor
        """
        ### TODO(VS) cleanup

        def plot_aff(aff):
            import pylab
            import matplotlib.pyplot as plt
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            colmap = pylab.cm.ScalarMappable(cmap=pylab.cm.viridis)
            colmap.set_array(aff[:,3])
            ax.scatter(aff[:,0], aff[:,1], aff[:,2], c=pylab.cm.viridis((aff[:,3]-min(aff[:,3]))/(max(aff[:,3])-min(aff[:,3]))), marker='o')
            fig.colorbar(colmap)
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            plt.show()
        def plot_aff2d(aff):
            import matplotlib.pyplot as plt
            fig = plt.figure()
            plt.imshow(aff, cmap='viridis')
            plt.colorbar()
            ax = plt.gca()
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            plt.show()

        assert not self.nets.training

        batch_size = obs_dict[tuple(obs_dict.keys())[0]].shape[0]
        actions = torch.zeros([batch_size, self.ac_dim]).to(self.device)
        # import pdb; pdb.set_trace()
        for ac in ["pick", "place"]:
            #####
            # import pdb; pdb.set_trace()
            aff = self._create_aff_grid({"obs": obs_dict,  "goal_obs": None}, policy_type=ac, policy_dim=f"{self.ac_implicit_dim}d")
            # plot_aff2d(aff[0].detach().numpy())
            # plot_aff2d(obs_dict["color_ortho"][0].transpose(0,1).transpose(1,2))

            ### old plotting, but still good for action selection ###
            from robomimic.utils.train_utils import _create_affordance_grid
            aff = _create_affordance_grid(self, {"obs": obs_dict, "goal_obs": None}, policy_type=ac, policy_dim=f"{self.ac_implicit_dim}d")
            # plot_aff2d(aff[0,...,-1])
            ###

            act = torch.mean( aff[aff[...,-1]==torch.max(aff[...,-1])][..., :self.ac_implicit_dim].unsqueeze(0), 1 ) # avg of (x,y) indices where aff==max # assumed batch_size=1
            # plot_aff(aff[0,torch.where(aff[0,:,2]==0)[0],:])
            # max_loc_z0 = aff[0,torch.where(aff[0,:,2]==0)[0],:][torch.argmax(aff[0, torch.where(aff[0,:,2]==0)[0], -1])]
            # act = aff[0,torch.argmax(aff[0,:,-1]),:3].unsqueeze(0)
            #####
            

            # act = torch.rand([batch_size, self.ac_implicit_dim]).to(self.device)
            # act[:,-1] = 0.0 # manually setting initial z=0

            affordance, _ = self.nets[f"{ac}_policy"](obs_dict, act, goal_dict=goal_dict)
            
            n_iters_mcmc = 10
            for _ in range(n_iters_mcmc):
                new_actions = act + torch.rand([batch_size, self.ac_implicit_dim]).to(self.device)
                new_affordance, _ = self.nets[f"{ac}_policy"](obs_dict, new_actions, goal_dict=goal_dict)
                choose_next_detr = new_affordance > affordance
                # choose_next_rand = torch.rand([batch_size]) < torch.exp(new_affordance - affordance) # TODO(VS)
                choose_next_rand = torch.tensor([[False]])
                choose_next = (choose_next_detr or choose_next_rand).float()
                act = choose_next*new_actions + (1-choose_next)*act
                affordance = choose_next*new_affordance + (1-choose_next)*affordance
            
            if ac == "pick":
                _, act_explicit = self.nets[f"{ac}_policy"](obs_dict, new_actions, goal_dict=goal_dict)
                actions[:,:self.ac_implicit_dim] = act
                if self.ac_implicit_dim == 2:
                    actions[:,2] = 0.04 # hard-coded pick height
                actions[:,3:7] = act_explicit
            else:
                _, act_explicit = self.nets[f"{ac}_policy"](obs_dict, new_actions, goal_dict=goal_dict)
                actions[:,7:7+self.ac_implicit_dim] = act
                if self.ac_implicit_dim == 2:
                    actions[:,9] = 0.02 # hard-coded place height
                actions[:,10:] = act_explicit

            # import pdb; pdb.set_trace()

        return actions


class BC_Gaussian(BC):
    """
    BC training with a Gaussian policy.
    """
    def _create_networks(self):
        """
        Creates networks and places them into @self.nets.
        """
        assert self.algo_config.gaussian.enabled

        self.nets = nn.ModuleDict()
        self.nets["policy"] = PolicyNets.GaussianActorNetwork(
            obs_shapes=self.obs_shapes,
            goal_shapes=self.goal_shapes,
            ac_dim=self.ac_dim,
            mlp_layer_dims=self.algo_config.actor_layer_dims,
            fixed_std=self.algo_config.gaussian.fixed_std,
            init_std=self.algo_config.gaussian.init_std,
            std_limits=(self.algo_config.gaussian.min_std, 7.5),
            std_activation=self.algo_config.gaussian.std_activation,
            low_noise_eval=self.algo_config.gaussian.low_noise_eval,
            encoder_kwargs=ObsUtils.obs_encoder_kwargs_from_config(self.obs_config.encoder),
        )

        self.nets = self.nets.float().to(self.device)

    def _forward_training(self, batch):
        """
        Internal helper function for BC algo class. Compute forward pass
        and return network outputs in @predictions dict.

        Args:
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader and filtered by @process_batch_for_training

        Returns:
            predictions (dict): dictionary containing network outputs
        """
        dists = self.nets["policy"].forward_train(
            obs_dict=batch["obs"], 
            goal_dict=batch["goal_obs"],
        )

        # make sure that this is a batch of multivariate action distributions, so that
        # the log probability computation will be correct
        assert len(dists.batch_shape) == 1
        log_probs = dists.log_prob(batch["actions"])

        predictions = OrderedDict(
            log_probs=log_probs,
        )
        return predictions

    def _compute_losses(self, predictions, batch):
        """
        Internal helper function for BC algo class. Compute losses based on
        network outputs in @predictions dict, using reference labels in @batch.

        Args:
            predictions (dict): dictionary containing network outputs, from @_forward_training
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader and filtered by @process_batch_for_training

        Returns:
            losses (dict): dictionary of losses computed over the batch
        """

        # loss is just negative log-likelihood of action targets
        action_loss = -predictions["log_probs"].mean()
        return OrderedDict(
            log_probs=-action_loss,
            action_loss=action_loss,
        )

    def log_info(self, info):
        """
        Process info dictionary from @train_on_batch to summarize
        information to pass to tensorboard for logging.

        Args:
            info (dict): dictionary of info

        Returns:
            loss_log (dict): name -> summary statistic
        """
        log = PolicyAlgo.log_info(self, info)
        log["Loss"] = info["losses"]["action_loss"].item()
        log["Log_Likelihood"] = info["losses"]["log_probs"].item() 
        if "policy_grad_norms" in info:
            log["Policy_Grad_Norms"] = info["policy_grad_norms"]
        return log


class BC_GMM(BC_Gaussian):
    """
    BC training with a Gaussian Mixture Model policy.
    """
    def _create_networks(self):
        """
        Creates networks and places them into @self.nets.
        """
        assert self.algo_config.gmm.enabled

        self.nets = nn.ModuleDict()
        self.nets["policy"] = PolicyNets.GMMActorNetwork(
            obs_shapes=self.obs_shapes,
            goal_shapes=self.goal_shapes,
            ac_dim=self.ac_dim,
            mlp_layer_dims=self.algo_config.actor_layer_dims,
            num_modes=self.algo_config.gmm.num_modes,
            min_std=self.algo_config.gmm.min_std,
            std_activation=self.algo_config.gmm.std_activation,
            low_noise_eval=self.algo_config.gmm.low_noise_eval,
            encoder_kwargs=ObsUtils.obs_encoder_kwargs_from_config(self.obs_config.encoder),
        )

        self.nets = self.nets.float().to(self.device)


class BC_VAE(BC):
    """
    BC training with a VAE policy.
    """
    def _create_networks(self):
        """
        Creates networks and places them into @self.nets.
        """
        self.nets = nn.ModuleDict()
        self.nets["policy"] = PolicyNets.VAEActor(
            obs_shapes=self.obs_shapes,
            goal_shapes=self.goal_shapes,
            ac_dim=self.ac_dim,
            device=self.device,
            encoder_kwargs=ObsUtils.obs_encoder_kwargs_from_config(self.obs_config.encoder),
            **VAENets.vae_args_from_config(self.algo_config.vae),
        )
        
        self.nets = self.nets.float().to(self.device)

    def train_on_batch(self, batch, epoch, validate=False):
        """
        Update from superclass to set categorical temperature, for categorical VAEs.
        """
        if self.algo_config.vae.prior.use_categorical:
            temperature = self.algo_config.vae.prior.categorical_init_temp - epoch * self.algo_config.vae.prior.categorical_temp_anneal_step
            temperature = max(temperature, self.algo_config.vae.prior.categorical_min_temp)
            self.nets["policy"].set_gumbel_temperature(temperature)
        return super(BC_VAE, self).train_on_batch(batch, epoch, validate=validate)

    def _forward_training(self, batch):
        """
        Internal helper function for BC algo class. Compute forward pass
        and return network outputs in @predictions dict.

        Args:
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader and filtered by @process_batch_for_training

        Returns:
            predictions (dict): dictionary containing network outputs
        """
        vae_inputs = dict(
            actions=batch["actions"],
            obs_dict=batch["obs"],
            goal_dict=batch["goal_obs"],
            freeze_encoder=batch.get("freeze_encoder", False),
        )

        vae_outputs = self.nets["policy"].forward_train(**vae_inputs)
        predictions = OrderedDict(
            actions=vae_outputs["decoder_outputs"],
            kl_loss=vae_outputs["kl_loss"],
            reconstruction_loss=vae_outputs["reconstruction_loss"],
            encoder_z=vae_outputs["encoder_z"],
        )
        if not self.algo_config.vae.prior.use_categorical:
            with torch.no_grad():
                encoder_variance = torch.exp(vae_outputs["encoder_params"]["logvar"])
            predictions["encoder_variance"] = encoder_variance
        return predictions

    def _compute_losses(self, predictions, batch):
        """
        Internal helper function for BC algo class. Compute losses based on
        network outputs in @predictions dict, using reference labels in @batch.

        Args:
            predictions (dict): dictionary containing network outputs, from @_forward_training
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader and filtered by @process_batch_for_training

        Returns:
            losses (dict): dictionary of losses computed over the batch
        """

        # total loss is sum of reconstruction and KL, weighted by beta
        kl_loss = predictions["kl_loss"]
        recons_loss = predictions["reconstruction_loss"]
        action_loss = recons_loss + self.algo_config.vae.kl_weight * kl_loss
        return OrderedDict(
            recons_loss=recons_loss,
            kl_loss=kl_loss,
            action_loss=action_loss,
        )

    def log_info(self, info):
        """
        Process info dictionary from @train_on_batch to summarize
        information to pass to tensorboard for logging.

        Args:
            info (dict): dictionary of info

        Returns:
            loss_log (dict): name -> summary statistic
        """
        log = PolicyAlgo.log_info(self, info)
        log["Loss"] = info["losses"]["action_loss"].item()
        log["KL_Loss"] = info["losses"]["kl_loss"].item()
        log["Reconstruction_Loss"] = info["losses"]["recons_loss"].item()
        if self.algo_config.vae.prior.use_categorical:
            log["Gumbel_Temperature"] = self.nets["policy"].get_gumbel_temperature()
        else:
            log["Encoder_Variance"] = info["predictions"]["encoder_variance"].mean().item()
        if "policy_grad_norms" in info:
            log["Policy_Grad_Norms"] = info["policy_grad_norms"]
        return log


class BC_RNN(BC):
    """
    BC training with an RNN policy.
    """
    def _create_networks(self):
        """
        Creates networks and places them into @self.nets.
        """
        self.nets = nn.ModuleDict()
        self.nets["policy"] = PolicyNets.RNNActorNetwork(
            obs_shapes=self.obs_shapes,
            goal_shapes=self.goal_shapes,
            ac_dim=self.ac_dim,
            mlp_layer_dims=self.algo_config.actor_layer_dims,
            encoder_kwargs=ObsUtils.obs_encoder_kwargs_from_config(self.obs_config.encoder),
            **BaseNets.rnn_args_from_config(self.algo_config.rnn),
        )

        self._rnn_hidden_state = None
        self._rnn_horizon = self.algo_config.rnn.horizon
        self._rnn_counter = 0
        self._rnn_is_open_loop = self.algo_config.rnn.get("open_loop", False)

        self.nets = self.nets.float().to(self.device)

    def process_batch_for_training(self, batch):
        """
        Processes input batch from a data loader to filter out
        relevant information and prepare the batch for training.

        Args:
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader

        Returns:
            input_batch (dict): processed and filtered batch that
                will be used for training
        """
        input_batch = dict()
        input_batch["obs"] = batch["obs"]
        input_batch["goal_obs"] = batch.get("goal_obs", None) # goals may not be present
        input_batch["actions"] = batch["actions"]

        if self._rnn_is_open_loop:
            # replace the observation sequence with one that only consists of the first observation.
            # This way, all actions are predicted "open-loop" after the first observation, based
            # on the rnn hidden state.
            n_steps = batch["actions"].shape[1]
            obs_seq_start = TensorUtils.index_at_time(batch["obs"], ind=0)
            input_batch["obs"] = TensorUtils.unsqueeze_expand_at(obs_seq_start, size=n_steps, dim=1)

        return TensorUtils.to_device(TensorUtils.to_float(input_batch), self.device)

    def get_action(self, obs_dict, goal_dict=None):
        """
        Get policy action outputs.

        Args:
            obs_dict (dict): current observation
            goal_dict (dict): (optional) goal

        Returns:
            action (torch.Tensor): action tensor
        """
        assert not self.nets.training

        if self._rnn_hidden_state is None or self._rnn_counter % self._rnn_horizon == 0:
            batch_size = list(obs_dict.values())[0].shape[0]
            self._rnn_hidden_state = self.nets["policy"].get_rnn_init_state(batch_size=batch_size, device=self.device)

            if self._rnn_is_open_loop:
                # remember the initial observation, and use it instead of the current observation
                # for open-loop action sequence prediction
                self._open_loop_obs = TensorUtils.clone(TensorUtils.detach(obs_dict))

        obs_to_use = obs_dict
        if self._rnn_is_open_loop:
            # replace current obs with last recorded obs
            obs_to_use = self._open_loop_obs

        self._rnn_counter += 1
        action, self._rnn_hidden_state = self.nets["policy"].forward_step(
            obs_to_use, goal_dict=goal_dict, rnn_state=self._rnn_hidden_state)
        return action

    def reset(self):
        """
        Reset algo state to prepare for environment rollouts.
        """
        self._rnn_hidden_state = None
        self._rnn_counter = 0


class BC_RNN_GMM(BC_RNN):
    """
    BC training with an RNN GMM policy.
    """
    def _create_networks(self):
        """
        Creates networks and places them into @self.nets.
        """
        assert self.algo_config.gmm.enabled
        assert self.algo_config.rnn.enabled

        self.nets = nn.ModuleDict()
        self.nets["policy"] = PolicyNets.RNNGMMActorNetwork(
            obs_shapes=self.obs_shapes,
            goal_shapes=self.goal_shapes,
            ac_dim=self.ac_dim,
            mlp_layer_dims=self.algo_config.actor_layer_dims,
            num_modes=self.algo_config.gmm.num_modes,
            min_std=self.algo_config.gmm.min_std,
            std_activation=self.algo_config.gmm.std_activation,
            low_noise_eval=self.algo_config.gmm.low_noise_eval,
            encoder_kwargs=ObsUtils.obs_encoder_kwargs_from_config(self.obs_config.encoder),
            **BaseNets.rnn_args_from_config(self.algo_config.rnn),
        )

        self._rnn_hidden_state = None
        self._rnn_horizon = self.algo_config.rnn.horizon
        self._rnn_counter = 0
        self._rnn_is_open_loop = self.algo_config.rnn.get("open_loop", False)

        self.nets = self.nets.float().to(self.device)

    def _forward_training(self, batch):
        """
        Internal helper function for BC algo class. Compute forward pass
        and return network outputs in @predictions dict.

        Args:
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader and filtered by @process_batch_for_training

        Returns:
            predictions (dict): dictionary containing network outputs
        """
        dists = self.nets["policy"].forward_train(
            obs_dict=batch["obs"], 
            goal_dict=batch["goal_obs"],
        )

        # make sure that this is a batch of multivariate action distributions, so that
        # the log probability computation will be correct
        assert len(dists.batch_shape) == 2 # [B, T]
        log_probs = dists.log_prob(batch["actions"])

        predictions = OrderedDict(
            log_probs=log_probs,
        )
        return predictions

    def _compute_losses(self, predictions, batch):
        """
        Internal helper function for BC algo class. Compute losses based on
        network outputs in @predictions dict, using reference labels in @batch.

        Args:
            predictions (dict): dictionary containing network outputs, from @_forward_training
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader and filtered by @process_batch_for_training

        Returns:
            losses (dict): dictionary of losses computed over the batch
        """

        # loss is just negative log-likelihood of action targets
        action_loss = -predictions["log_probs"].mean()
        return OrderedDict(
            log_probs=-action_loss,
            action_loss=action_loss,
        )

    def log_info(self, info):
        """
        Process info dictionary from @train_on_batch to summarize
        information to pass to tensorboard for logging.

        Args:
            info (dict): dictionary of info

        Returns:
            loss_log (dict): name -> summary statistic
        """
        log = PolicyAlgo.log_info(self, info)
        log["Loss"] = info["losses"]["action_loss"].item()
        log["Log_Likelihood"] = info["losses"]["log_probs"].item() 
        if "policy_grad_norms" in info:
            log["Policy_Grad_Norms"] = info["policy_grad_norms"]
        return log
