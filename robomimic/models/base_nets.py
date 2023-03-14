"""
Contains torch Modules that correspond to basic network building blocks, like 
MLP, RNN, and CNN backbones.
"""

import sys
import math
import abc
import numpy as np
import textwrap
from copy import deepcopy
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models as vision_models

import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.obs_utils as ObsUtils
from robomimic.utils.python_utils import extract_class_init_kwargs_from_dict


CONV_ACTIVATIONS = {
    "relu": nn.ReLU,
    "None": None,
    None: None,
}


def rnn_args_from_config(rnn_config):
    """
    Takes a Config object corresponding to RNN settings
    (for example `config.algo.rnn` in BCConfig) and extracts
    rnn kwargs for instantiating rnn networks.
    """
    return dict(
        rnn_hidden_dim=rnn_config.hidden_dim,
        rnn_num_layers=rnn_config.num_layers,
        rnn_type=rnn_config.rnn_type,
        rnn_kwargs=dict(rnn_config.kwargs),
    )


class Module(torch.nn.Module):
    """
    Base class for networks. The only difference from torch.nn.Module is that it
    requires implementing @output_shape.
    """
    @abc.abstractmethod
    def output_shape(self, input_shape=None):
        """
        Function to compute output shape from inputs to this module. 

        Args:
            input_shape (iterable of int): shape of input. Does not include batch dimension.
                Some modules may not need this argument, if their output does not depend 
                on the size of the input, or if they assume fixed size input.

        Returns:
            out_shape ([int]): list of integers corresponding to output shape
        """
        raise NotImplementedError


class Sequential(torch.nn.Sequential, Module):
    """
    Compose multiple Modules together (defined above).
    """
    def __init__(self, *args):
        for arg in args:
            assert isinstance(arg, Module)
        torch.nn.Sequential.__init__(self, *args)

    def output_shape(self, input_shape=None):
        """
        Function to compute output shape from inputs to this module. 

        Args:
            input_shape (iterable of int): shape of input. Does not include batch dimension.
                Some modules may not need this argument, if their output does not depend 
                on the size of the input, or if they assume fixed size input.

        Returns:
            out_shape ([int]): list of integers corresponding to output shape
        """
        out_shape = input_shape
        for module in self:
            out_shape = module.output_shape(out_shape)
        return out_shape


class Parameter(Module):
    """
    A class that is a thin wrapper around a torch.nn.Parameter to make for easy saving
    and optimization.
    """
    def __init__(self, init_tensor):
        """
        Args:
            init_tensor (torch.Tensor): initial tensor
        """
        super(Parameter, self).__init__()
        self.param = torch.nn.Parameter(init_tensor)

    def output_shape(self, input_shape=None):
        """
        Function to compute output shape from inputs to this module. 

        Args:
            input_shape (iterable of int): shape of input. Does not include batch dimension.
                Some modules may not need this argument, if their output does not depend 
                on the size of the input, or if they assume fixed size input.

        Returns:
            out_shape ([int]): list of integers corresponding to output shape
        """
        return list(self.param.shape)

    def forward(self, inputs=None):
        """
        Forward call just returns the parameter tensor.
        """
        return self.param


class Unsqueeze(Module):
    """
    Trivial class that unsqueezes the input. Useful for including in a nn.Sequential network
    """
    def __init__(self, dim):
        super(Unsqueeze, self).__init__()
        self.dim = dim

    def output_shape(self, input_shape=None):
        assert input_shape is not None
        return input_shape + [1] if self.dim == -1 else input_shape[:self.dim + 1] + [1] + input_shape[self.dim + 1:]

    def forward(self, x):
        return x.unsqueeze(dim=self.dim)


class Squeeze(Module):
    """
    Trivial class that squeezes the input. Useful for including in a nn.Sequential network
    """

    def __init__(self, dim):
        super(Squeeze, self).__init__()
        self.dim = dim

    def output_shape(self, input_shape=None):
        assert input_shape is not None
        return input_shape[:self.dim] + input_shape[self.dim+1:] if input_shape[self.dim] == 1 else input_shape

    def forward(self, x):
        return x.squeeze(dim=self.dim)


class ResNetMLP(Module):
    """
    Basic ResNet block.

    Forward returns the following:
        output_activation ( model(input) + activation(layer_func(input)) )
    """
    def __init__(
        self, 
        model, 
        input_dim,
        output_dim, 
        layer_func=nn.Linear,
        layer_func_kwargs=None,
        activation=nn.ReLU,
        output_activation=None,
    ):
        """
        Args:
            input_dim (int): dimension of inputs

            output_dim (int): dimension of outputs

            layer_func: func for processing input (single layer) - defaults to Linear

            layer_func_kwargs (dict): kwargs for @layer_func

            activation: non-linearity after layer_func - defaults to ReLU

            output_activation: if provided, applies the provided non-linearity after adding residual
        """
        super(ResNetMLP, self).__init__()
        self.model = model
        self._input_dim = input_dim
        self._output_dim = output_dim

        if layer_func_kwargs is None:
            layer_func_kwargs = dict()

        # single layer MLP for converting input to dim output_dim.
        layers = [layer_func(input_dim, output_dim, **layer_func_kwargs)]
        if activation is not None:
            layers.append(activation())
        self._model = nn.Sequential(*layers)
        self.output_activation = output_activation

    def output_shape(self, input_shape=None):
        [self._output_dim]

    def forward(self, inputs):
        out =  self.model(inputs) + self._model(inputs)
        if self.output_activation is not None:
            out = self.output_activation(out)
        return out


class MLP(Module):
    """
    Base class for simple Multi-Layer Perceptrons.
    """
    def __init__(
        self,
        input_dim,
        output_dim,
        layer_dims=(),
        layer_func=nn.Linear,
        layer_func_kwargs=None,
        activation=nn.ReLU,
        dropouts=None,
        normalization=False,
        output_activation=None,
        residual=False,
    ):
        """
        Args:
            input_dim (int): dimension of inputs

            output_dim (int): dimension of outputs

            layer_dims ([int]): sequence of integers for the hidden layers sizes

            layer_func: mapping per layer - defaults to Linear

            layer_func_kwargs (dict): kwargs for @layer_func

            activation: non-linearity per layer - defaults to ReLU

            dropouts ([float]): if not None, adds dropout layers with the corresponding probabilities
                after every layer. Must be same size as @layer_dims.

            normalization (bool): if True, apply layer normalization after each layer

            output_activation: if provided, applies the provided non-linearity to the output layer
        """
        super(MLP, self).__init__()
        layers = []
        dim = input_dim
        if layer_func_kwargs is None:
            layer_func_kwargs = dict()
        if dropouts is not None:
            assert(len(dropouts) == len(layer_dims))
        for i, l in enumerate(layer_dims):
            layers.append(layer_func(dim, l, **layer_func_kwargs))
            if normalization:
                layers.append(nn.LayerNorm(l))
            layers.append(activation())
            if dropouts is not None and dropouts[i] > 0.:
                layers.append(nn.Dropout(dropouts[i]))
            dim = l
        layers.append(layer_func(dim, output_dim))
        if residual:
            model = nn.Sequential(*layers)
            layers = [ResNetMLP(model, input_dim, output_dim, layer_func=layer_func, layer_func_kwargs=layer_func_kwargs, activation=activation, output_activation=None)]
        if output_activation is not None:
            layers.append(output_activation())
        self._layer_func = layer_func
        self.nets = layers
        self._model = nn.Sequential(*layers)

        self._layer_dims = layer_dims
        self._input_dim = input_dim
        self._output_dim = output_dim
        self._dropouts = dropouts
        self._act = activation
        self._output_act = output_activation

    def output_shape(self, input_shape=None):
        """
        Function to compute output shape from inputs to this module. 

        Args:
            input_shape (iterable of int): shape of input. Does not include batch dimension.
                Some modules may not need this argument, if their output does not depend 
                on the size of the input, or if they assume fixed size input.

        Returns:
            out_shape ([int]): list of integers corresponding to output shape
        """
        return [self._output_dim]

    def forward(self, inputs):
        """
        Forward pass.
        """
        return self._model(inputs)

    def __repr__(self):
        """Pretty print network."""
        header = str(self.__class__.__name__)
        act = None if self._act is None else self._act.__name__
        output_act = None if self._output_act is None else self._output_act.__name__

        indent = ' ' * 4
        msg = "input_dim={}\noutput_dim={}\nlayer_dims={}\nlayer_func={}\ndropout={}\nact={}\noutput_act={}".format(
            self._input_dim, self._output_dim, self._layer_dims,
            self._layer_func.__name__, self._dropouts, act, output_act
        )
        msg = textwrap.indent(msg, indent)
        msg = header + '(\n' + msg + '\n)'
        return msg


class RNN_Base(Module):
    """
    A wrapper class for a multi-step RNN and a per-step network.
    """
    def __init__(
        self,
        input_dim,
        rnn_hidden_dim,
        rnn_num_layers,
        rnn_type="LSTM",  # [LSTM, GRU]
        rnn_kwargs=None,
        per_step_net=None,
    ):
        """
        Args:
            input_dim (int): dimension of inputs

            rnn_hidden_dim (int): RNN hidden dimension

            rnn_num_layers (int): number of RNN layers

            rnn_type (str): [LSTM, GRU]

            rnn_kwargs (dict): kwargs for the torch.nn.LSTM / GRU

            per_step_net: a network that runs per time step on top of the RNN output
        """
        super(RNN_Base, self).__init__()
        self.per_step_net = per_step_net
        if per_step_net is not None:
            assert isinstance(per_step_net, Module), "RNN_Base: per_step_net is not instance of Module"

        assert rnn_type in ["LSTM", "GRU"]
        rnn_cls = nn.LSTM if rnn_type == "LSTM" else nn.GRU
        rnn_kwargs = rnn_kwargs if rnn_kwargs is not None else {}
        rnn_is_bidirectional = rnn_kwargs.get("bidirectional", False)

        self.nets = rnn_cls(
            input_size=input_dim,
            hidden_size=rnn_hidden_dim,
            num_layers=rnn_num_layers,
            batch_first=True,
            **rnn_kwargs,
        )

        self._hidden_dim = rnn_hidden_dim
        self._num_layers = rnn_num_layers
        self._rnn_type = rnn_type
        self._num_directions = int(rnn_is_bidirectional) + 1 # 2 if bidirectional, 1 otherwise

    @property
    def rnn_type(self):
        return self._rnn_type

    def get_rnn_init_state(self, batch_size, device):
        """
        Get a default RNN state (zeros)
        Args:
            batch_size (int): batch size dimension

            device: device the hidden state should be sent to.

        Returns:
            hidden_state (torch.Tensor or tuple): returns hidden state tensor or tuple of hidden state tensors
                depending on the RNN type
        """
        h_0 = torch.zeros(self._num_layers * self._num_directions, batch_size, self._hidden_dim).to(device)
        if self._rnn_type == "LSTM":
            c_0 = torch.zeros(self._num_layers * self._num_directions, batch_size, self._hidden_dim).to(device)
            return h_0, c_0
        else:
            return h_0

    def output_shape(self, input_shape):
        """
        Function to compute output shape from inputs to this module. 

        Args:
            input_shape (iterable of int): shape of input. Does not include batch dimension.
                Some modules may not need this argument, if their output does not depend 
                on the size of the input, or if they assume fixed size input.

        Returns:
            out_shape ([int]): list of integers corresponding to output shape
        """

        # infer time dimension from input shape and add to per_step_net output shape
        if self.per_step_net is not None:
            out = self.per_step_net.output_shape(input_shape[1:])
            if isinstance(out, dict):
                out = {k: [input_shape[0]] + out[k] for k in out}
            else:
                out = [input_shape[0]] + out
        else:
            out = [input_shape[0], self._num_layers * self._hidden_dim]
        return out

    def forward(self, inputs, rnn_init_state=None, return_state=False):
        """
        Forward a sequence of inputs through the RNN and the per-step network.

        Args:
            inputs (torch.Tensor): tensor input of shape [B, T, D], where D is the RNN input size

            rnn_init_state: rnn hidden state, initialize to zero state if set to None

            return_state (bool): whether to return hidden state

        Returns:
            outputs: outputs of the per_step_net

            rnn_state: return rnn state at the end if return_state is set to True
        """
        assert inputs.ndimension() == 3  # [B, T, D]
        batch_size, seq_length, inp_dim = inputs.shape
        if rnn_init_state is None:
            rnn_init_state = self.get_rnn_init_state(batch_size, device=inputs.device)

        outputs, rnn_state = self.nets(inputs, rnn_init_state)
        if self.per_step_net is not None:
            outputs = TensorUtils.time_distributed(outputs, self.per_step_net)

        if return_state:
            return outputs, rnn_state
        else:
            return outputs

    def forward_step(self, inputs, rnn_state):
        """
        Forward a single step input through the RNN and per-step network, and return the new hidden state.
        Args:
            inputs (torch.Tensor): tensor input of shape [B, D], where D is the RNN input size

            rnn_state: rnn hidden state, initialize to zero state if set to None

        Returns:
            outputs: outputs of the per_step_net

            rnn_state: return the new rnn state
        """
        assert inputs.ndimension() == 2
        inputs = TensorUtils.to_sequence(inputs)
        outputs, rnn_state = self.forward(
            inputs,
            rnn_init_state=rnn_state,
            return_state=True,
        )
        return outputs[:, 0], rnn_state


"""
================================================
Visual Backbone Networks
================================================
"""
class ConvBase(Module):
    """
    Base class for ConvNets.
    """
    def __init__(self):
        super(ConvBase, self).__init__()

    # dirty hack - re-implement to pass the buck onto subclasses from ABC parent
    def output_shape(self, input_shape):
        """
        Function to compute output shape from inputs to this module. 

        Args:
            input_shape (iterable of int): shape of input. Does not include batch dimension.
                Some modules may not need this argument, if their output does not depend 
                on the size of the input, or if they assume fixed size input.

        Returns:
            out_shape ([int]): list of integers corresponding to output shape
        """
        raise NotImplementedError

    def forward(self, inputs):
        x = self.nets(inputs)
        if list(self.output_shape(list(inputs.shape)[1:])) != list(x.shape)[1:]:
            raise ValueError('Size mismatch: expect size %s, but got size %s' % (
                str(self.output_shape(list(inputs.shape)[1:])), str(list(x.shape)[1:]))
            )
        return x


class ResNet18Conv(ConvBase):
    """
    A ResNet18 block that can be used to process input images.
    """
    def __init__(
        self,
        input_channel=3,
        pretrained=False,
        input_coord_conv=False,
    ):
        """
        Args:
            input_channel (int): number of input channels for input images to the network.
                If not equal to 3, modifies first conv layer in ResNet to handle the number
                of input channels.
            pretrained (bool): if True, load pretrained weights for all ResNet layers.
            input_coord_conv (bool): if True, use a coordinate convolution for the first layer
                (a convolution where input channels are modified to encode spatial pixel location)
        """
        super(ResNet18Conv, self).__init__()
        net = vision_models.resnet18(pretrained=pretrained)

        if input_coord_conv:
            net.conv1 = CoordConv2d(input_channel, 64, kernel_size=7, stride=2, padding=3, bias=False)
        elif input_channel != 3:
            net.conv1 = nn.Conv2d(input_channel, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # cut the last fc layer
        self._input_coord_conv = input_coord_conv
        self._input_channel = input_channel
        self.nets = torch.nn.Sequential(*(list(net.children())[:-2]))

    def output_shape(self, input_shape):
        """
        Function to compute output shape from inputs to this module. 

        Args:
            input_shape (iterable of int): shape of input. Does not include batch dimension.
                Some modules may not need this argument, if their output does not depend 
                on the size of the input, or if they assume fixed size input.

        Returns:
            out_shape ([int]): list of integers corresponding to output shape
        """
        assert(len(input_shape) == 3)
        out_h = int(math.ceil(input_shape[1] / 32.))
        out_w = int(math.ceil(input_shape[2] / 32.))
        return [512, out_h, out_w]

    def __repr__(self):
        """Pretty print network."""
        header = '{}'.format(str(self.__class__.__name__))
        return header + '(input_channel={}, input_coord_conv={})'.format(self._input_channel, self._input_coord_conv)


class ResNet50Conv(ConvBase):
    """
    A ResNet50 block that can be used to process input images.
    """
    def __init__(
        self,
        input_channel=3,
        pretrained=False,
        input_coord_conv=False,
    ):
        """
        Args:
            input_channel (int): number of input channels for input images to the network.
                If not equal to 3, modifies first conv layer in ResNet to handle the number
                of input channels.
            pretrained (bool): if True, load pretrained weights for all ResNet layers.
            input_coord_conv (bool): if True, use a coordinate convolution for the first layer
                (a convolution where input channels are modified to encode spatial pixel location)
        """
        super(ResNet50Conv, self).__init__()
        net = vision_models.resnet50(pretrained=pretrained)

        if input_coord_conv:
            net.conv1 = CoordConv2d(input_channel, 64, kernel_size=7, stride=2, padding=3, bias=False)
        elif input_channel != 3:
            net.conv1 = nn.Conv2d(input_channel, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # cut the last fc layer
        self._input_coord_conv = input_coord_conv
        self._input_channel = input_channel
        self.nets = torch.nn.Sequential(*(list(net.children())[:-2]))

    def output_shape(self, input_shape):
        """
        Function to compute output shape from inputs to this module. 

        Args:
            input_shape (iterable of int): shape of input. Does not include batch dimension.
                Some modules may not need this argument, if their output does not depend 
                on the size of the input, or if they assume fixed size input.

        Returns:
            out_shape ([int]): list of integers corresponding to output shape
        """
        assert(len(input_shape) == 3)
        out_h = int(math.ceil(input_shape[1] / 32.))
        out_w = int(math.ceil(input_shape[2] / 32.))
        return [2048, out_h, out_w]

    def __repr__(self):
        """Pretty print network."""
        header = '{}'.format(str(self.__class__.__name__))
        return header + '(input_channel={}, input_coord_conv={})'.format(self._input_channel, self._input_coord_conv)


class UNetConv(ConvBase):
    def __init__(
        self,
        input_channel=3,
        pretrained=False, #TODO(VS)
        init_features=32,
        recon_enabled=False,
    ):
        """
        Architecture from: https://github.com/mateuszbuda/brain-segmentation-pytorch
        """
        super(UNetConv, self).__init__()
        self._input_channel = input_channel
        self._output_channel = input_channel
        self.init_features = init_features
        self.recon_enabled = recon_enabled

        # self.net = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet', in_channels=input_channel, out_channels=input_channel, init_features=self.init_features, pretrained=pretrained)
        self.net = self._create_network()

    def _create_network(self):
        features = self.init_features
        in_channels = self._input_channel
        out_channels = self._output_channel

        self.encoder1 = UNetConv._block(in_channels, features, name="enc1")
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = UNetConv._block(features, features * 2, name="enc2")
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3 = UNetConv._block(features * 2, features * 4, name="enc3")
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder4 = UNetConv._block(features * 4, features * 8, name="enc4")
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = UNetConv._block(features * 8, features * 16, name="bottleneck")

        self.upconv4 = nn.ConvTranspose2d(
            features * 16, features * 8, kernel_size=2, stride=2
        )
        self.decoder4 = UNetConv._block((features * 8) * 2, features * 8, name="dec4")
        self.upconv3 = nn.ConvTranspose2d(
            features * 8, features * 4, kernel_size=2, stride=2
        )
        self.decoder3 = UNetConv._block((features * 4) * 2, features * 4, name="dec3")
        self.upconv2 = nn.ConvTranspose2d(
            features * 4, features * 2, kernel_size=2, stride=2
        )
        self.decoder2 = UNetConv._block((features * 2) * 2, features * 2, name="dec2")
        self.upconv1 = nn.ConvTranspose2d(
            features * 2, features, kernel_size=2, stride=2
        )
        self.decoder1 = UNetConv._block(features * 2, features, name="dec1")

        self.conv = nn.Conv2d(
            in_channels=features, out_channels=out_channels, kernel_size=1
        )

    @staticmethod
    def _block(in_channels, features, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    # (name + "norm1", nn.BatchNorm2d(num_features=features)),
                    (name + "relu1", nn.ReLU(inplace=True)),
                    (
                        name + "conv2",
                        nn.Conv2d(
                            in_channels=features,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    # (name + "norm2", nn.BatchNorm2d(num_features=features)),
                    (name + "relu2", nn.ReLU(inplace=True)),
                ]
            )
        )

    def output_shape(self, input_shape):
        """
        Function to compute output shape from inputs to this module. 

        Args:
            input_shape (iterable of int): shape of input. Does not include batch dimension.
                Some modules may not need this argument, if their output does not depend 
                on the size of the input, or if they assume fixed size input.

        Returns:
            out_shape ([int]): list of integers corresponding to output shape
        """
        assert(len(input_shape) == 3)
        out_h = int(math.ceil(input_shape[1] / 16.))
        out_w = int(math.ceil(input_shape[2] / 16.))
        return [self.init_features * 16, out_h, out_w]

    def forward(self, inputs):
        enc1 = self.encoder1(inputs)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))

        bottleneck = self.bottleneck(self.pool4(enc4))

        if not self.recon_enabled:
            return {"feats": bottleneck, "recon": None}

        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)
        recon = torch.sigmoid(self.conv(dec1))

        return {"feats": bottleneck, "recon": recon}

    def __repr__(self):
        """Pretty print network."""
        header = '{}'.format(str(self.__class__.__name__))
        return header + '(input_channel={})'.format(self._input_channel)


class CoordConv2d(nn.Conv2d, Module):
    """
    2D Coordinate Convolution

    Source: An Intriguing Failing of Convolutional Neural Networks and the CoordConv Solution
    https://arxiv.org/abs/1807.03247
    (e.g. adds 2 channels per input feature map corresponding to (x, y) location on map)
    """
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        padding_mode='zeros',
        coord_encoding='position',
    ):
        """
        Args:
            in_channels: number of channels of the input tensor [C, H, W]
            out_channels: number of output channels of the layer
            kernel_size: convolution kernel size
            stride: conv stride
            padding: conv padding
            dilation: conv dilation
            groups: conv groups
            bias: conv bias
            padding_mode: conv padding mode
            coord_encoding: type of coordinate encoding. currently only 'position' is implemented
        """

        assert(coord_encoding in ['position'])
        self.coord_encoding = coord_encoding
        if coord_encoding == 'position':
            in_channels += 2  # two extra channel for positional encoding
            self._position_enc = None  # position encoding
        else:
            raise Exception("CoordConv2d: coord encoding {} not implemented".format(self.coord_encoding))
        nn.Conv2d.__init__(
            self,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode
        )

    def output_shape(self, input_shape):
        """
        Function to compute output shape from inputs to this module. 

        Args:
            input_shape (iterable of int): shape of input. Does not include batch dimension.
                Some modules may not need this argument, if their output does not depend 
                on the size of the input, or if they assume fixed size input.

        Returns:
            out_shape ([int]): list of integers corresponding to output shape
        """

        # adds 2 to channel dimension
        return [input_shape[0] + 2] + input_shape[1:]

    def forward(self, input):
        b, c, h, w = input.shape
        if self.coord_encoding == 'position':
            if self._position_enc is None:
                pos_y, pos_x = torch.meshgrid(torch.arange(h), torch.arange(w))
                pos_y = pos_y.float().to(input.device) / float(h)
                pos_x = pos_x.float().to(input.device) / float(w)
                self._position_enc = torch.stack((pos_y, pos_x)).unsqueeze(0)
            pos_enc = self._position_enc.expand(b, -1, -1, -1)
            input = torch.cat((input, pos_enc), dim=1)
        return super(CoordConv2d, self).forward(input)


class ShallowConv(ConvBase):
    """
    A shallow convolutional encoder from https://rll.berkeley.edu/dsae/dsae.pdf
    """
    def __init__(self, input_channel=3, output_channel=32):
        super(ShallowConv, self).__init__()
        self._input_channel = input_channel
        self._output_channel = output_channel
        self.nets = nn.Sequential(
            torch.nn.Conv2d(input_channel, 64, kernel_size=7, stride=2, padding=3),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 32, kernel_size=1, stride=1, padding=0),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
        )

    def output_shape(self, input_shape):
        """
        Function to compute output shape from inputs to this module. 

        Args:
            input_shape (iterable of int): shape of input. Does not include batch dimension.
                Some modules may not need this argument, if their output does not depend 
                on the size of the input, or if they assume fixed size input.

        Returns:
            out_shape ([int]): list of integers corresponding to output shape
        """
        assert(len(input_shape) == 3)
        assert(input_shape[0] == self._input_channel)
        out_h = int(math.floor(input_shape[1] / 2.))
        out_w = int(math.floor(input_shape[2] / 2.))
        return [self._output_channel, out_h, out_w]


class Conv1dBase(Module):
    """
    Base class for stacked Conv1d layers.

    Args:
        input_channel (int): Number of channels for inputs to this network
        activation (None or str): Per-layer activation to use. Defaults to "relu". Valid options are
            currently {relu, None} for no activation
        conv_kwargs (dict): Specific nn.Conv1D args to use, in list form, where the ith element corresponds to the
            argument to be passed to the ith Conv1D layer.
            See https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html for specific possible arguments.

            e.g.: common values to use:
                out_channels (list of int): Output channel size for each sequential Conv1d layer
                kernel_size (list of int): Kernel sizes for each sequential Conv1d layer
                stride (list of int): Stride sizes for each sequential Conv1d layer
    """
    def __init__(
        self,
        input_channel=1,
        activation="relu",
        **conv_kwargs,
    ):
        super(Conv1dBase, self).__init__()

        # Get activation requested
        activation = CONV_ACTIVATIONS[activation]

        # Make sure out_channels and kernel_size are specified
        for kwarg in ("out_channels", "kernel_size"):
            assert kwarg in conv_kwargs, f"{kwarg} must be specified in Conv1dBase kwargs!"

        # Generate network
        self.n_layers = len(conv_kwargs["out_channels"])
        layers = OrderedDict()
        for i in range(self.n_layers):
            layer_kwargs = {k: v[i] for k, v in conv_kwargs.items()}
            layers[f'conv{i}'] = nn.Conv1d(
                in_channels=input_channel,
                **layer_kwargs,
            )
            if activation is not None:
                layers[f'act{i}'] = activation()
            input_channel = layer_kwargs["out_channels"]

        # Store network
        self.nets = nn.Sequential(layers)

    def output_shape(self, input_shape):
        """
        Function to compute output shape from inputs to this module.

        Args:
            input_shape (iterable of int): shape of input. Does not include batch dimension.
                Some modules may not need this argument, if their output does not depend
                on the size of the input, or if they assume fixed size input.

        Returns:
            out_shape ([int]): list of integers corresponding to output shape
        """
        channels, length = input_shape
        for i in range(self.n_layers):
            net = getattr(self.nets, f"conv{i}")
            channels = net.out_channels
            length = int((length + 2 * net.padding[0] - net.dilation[0] * (net.kernel_size[0] - 1) - 1) / net.stride[0]) + 1
        return [channels, length]

    def forward(self, inputs):
        x = self.nets(inputs)
        if list(self.output_shape(list(inputs.shape)[1:])) != list(x.shape)[1:]:
            raise ValueError('Size mismatch: expect size %s, but got size %s' % (
                str(self.output_shape(list(inputs.shape)[1:])), str(list(x.shape)[1:]))
            )
        return x


"""
================================================
Pooling Networks
================================================
"""
class SpatialSoftmax(ConvBase):
    """
    Spatial Softmax Layer.

    Based on Deep Spatial Autoencoders for Visuomotor Learning by Finn et al.
    https://rll.berkeley.edu/dsae/dsae.pdf
    """
    def __init__(
        self,
        input_shape,
        num_kp=None,
        temperature=1.,
        learnable_temperature=False,
        output_variance=False,
        noise_std=0.0,
    ):
        """
        Args:
            input_shape (list): shape of the input feature (C, H, W)
            num_kp (int): number of keypoints (None for not use spatialsoftmax)
            temperature (float): temperature term for the softmax.
            learnable_temperature (bool): whether to learn the temperature
            output_variance (bool): treat attention as a distribution, and compute second-order statistics to return
            noise_std (float): add random spatial noise to the predicted keypoints
        """
        super(SpatialSoftmax, self).__init__()
        assert len(input_shape) == 3
        self._in_c, self._in_h, self._in_w = input_shape # (C, H, W)

        if num_kp is not None:
            self.nets = torch.nn.Conv2d(self._in_c, num_kp, kernel_size=1)
            self._num_kp = num_kp
        else:
            self.nets = None
            self._num_kp = self._in_c
        self.learnable_temperature = learnable_temperature
        self.output_variance = output_variance
        self.noise_std = noise_std

        if self.learnable_temperature:
            # temperature will be learned
            temperature = torch.nn.Parameter(torch.ones(1) * temperature, requires_grad=True)
            self.register_parameter('temperature', temperature)
        else:
            # temperature held constant after initialization
            temperature = torch.nn.Parameter(torch.ones(1) * temperature, requires_grad=False)
            self.register_buffer('temperature', temperature)

        pos_x, pos_y = np.meshgrid(
                np.linspace(-1., 1., self._in_w),
                np.linspace(-1., 1., self._in_h)
                )
        pos_x = torch.from_numpy(pos_x.reshape(1, self._in_h * self._in_w)).float()
        pos_y = torch.from_numpy(pos_y.reshape(1, self._in_h * self._in_w)).float()
        self.register_buffer('pos_x', pos_x)
        self.register_buffer('pos_y', pos_y)

        self.kps = None

    def __repr__(self):
        """Pretty print network."""
        header = format(str(self.__class__.__name__))
        return header + '(num_kp={}, temperature={}, noise={})'.format(
            self._num_kp, self.temperature.item(), self.noise_std)

    def output_shape(self, input_shape):
        """
        Function to compute output shape from inputs to this module. 

        Args:
            input_shape (iterable of int): shape of input. Does not include batch dimension.
                Some modules may not need this argument, if their output does not depend 
                on the size of the input, or if they assume fixed size input.

        Returns:
            out_shape ([int]): list of integers corresponding to output shape
        """
        assert(len(input_shape) == 3)
        assert(input_shape[0] == self._in_c)
        return [self._num_kp, 2]

    def forward(self, feature):
        """
        Forward pass through spatial softmax layer. For each keypoint, a 2D spatial 
        probability distribution is created using a softmax, where the support is the 
        pixel locations. This distribution is used to compute the expected value of 
        the pixel location, which becomes a keypoint of dimension 2. K such keypoints
        are created.

        Returns:
            out (torch.Tensor or tuple): mean keypoints of shape [B, K, 2], and possibly
                keypoint variance of shape [B, K, 2, 2] corresponding to the covariance
                under the 2D spatial softmax distribution
        """
        assert(feature.shape[1] == self._in_c)
        assert(feature.shape[2] == self._in_h)
        assert(feature.shape[3] == self._in_w)
        if self.nets is not None:
            feature = self.nets(feature)

        # [B, K, H, W] -> [B * K, H * W] where K is number of keypoints
        feature = feature.reshape(-1, self._in_h * self._in_w)
        # 2d softmax normalization
        attention = F.softmax(feature / self.temperature, dim=-1)
        # [1, H * W] x [B * K, H * W] -> [B * K, 1] for spatial coordinate mean in x and y dimensions
        expected_x = torch.sum(self.pos_x * attention, dim=1, keepdim=True)
        expected_y = torch.sum(self.pos_y * attention, dim=1, keepdim=True)
        # stack to [B * K, 2]
        expected_xy = torch.cat([expected_x, expected_y], 1)
        # reshape to [B, K, 2]
        feature_keypoints = expected_xy.view(-1, self._num_kp, 2)

        if self.training:
            noise = torch.randn_like(feature_keypoints) * self.noise_std
            feature_keypoints += noise

        if self.output_variance:
            # treat attention as a distribution, and compute second-order statistics to return
            expected_xx = torch.sum(self.pos_x * self.pos_x * attention, dim=1, keepdim=True)
            expected_yy = torch.sum(self.pos_y * self.pos_y * attention, dim=1, keepdim=True)
            expected_xy = torch.sum(self.pos_x * self.pos_y * attention, dim=1, keepdim=True)
            var_x = expected_xx - expected_x * expected_x
            var_y = expected_yy - expected_y * expected_y
            var_xy = expected_xy - expected_x * expected_y
            # stack to [B * K, 4] and then reshape to [B, K, 2, 2] where last 2 dims are covariance matrix
            feature_covar = torch.cat([var_x, var_xy, var_xy, var_y], 1).reshape(-1, self._num_kp, 2, 2)
            feature_keypoints = (feature_keypoints, feature_covar)

        if isinstance(feature_keypoints, tuple):
            self.kps = (feature_keypoints[0].detach(), feature_keypoints[1].detach())
        else:
            self.kps = feature_keypoints.detach()
        return feature_keypoints


class SpatialMeanPool(Module):
    """
    Module that averages inputs across all spatial dimensions (dimension 2 and after),
    leaving only the batch and channel dimensions.
    """
    def __init__(self, input_shape):
        super(SpatialMeanPool, self).__init__()
        assert len(input_shape) == 3 # [C, H, W]
        self.in_shape = input_shape

    def output_shape(self, input_shape=None):
        """
        Function to compute output shape from inputs to this module. 

        Args:
            input_shape (iterable of int): shape of input. Does not include batch dimension.
                Some modules may not need this argument, if their output does not depend 
                on the size of the input, or if they assume fixed size input.

        Returns:
            out_shape ([int]): list of integers corresponding to output shape
        """
        return list(self.in_shape[:1]) # [C, H, W] -> [C]

    def forward(self, inputs):
        """Forward pass - average across all dimensions except batch and channel."""
        return TensorUtils.flatten(inputs, begin_axis=2).mean(dim=2)


class FeatureAggregator(Module):
    """
    Helpful class for aggregating features across a dimension. This is useful in 
    practice when training models that break an input image up into several patches
    since features can be extraced per-patch using the same encoder and then 
    aggregated using this module.
    """
    def __init__(self, dim=1, agg_type="avg"):
        super(FeatureAggregator, self).__init__()
        self.dim = dim
        self.agg_type = agg_type

    def set_weight(self, w):
        assert self.agg_type == "w_avg"
        self.agg_weight = w

    def clear_weight(self):
        assert self.agg_type == "w_avg"
        self.agg_weight = None

    def output_shape(self, input_shape):
        """
        Function to compute output shape from inputs to this module. 

        Args:
            input_shape (iterable of int): shape of input. Does not include batch dimension.
                Some modules may not need this argument, if their output does not depend 
                on the size of the input, or if they assume fixed size input.

        Returns:
            out_shape ([int]): list of integers corresponding to output shape
        """
        # aggregates on @self.dim, so it is removed from the output shape 
        return list(input_shape[:self.dim]) + list(input_shape[self.dim+1:])

    def forward(self, x):
        """Forward pooling pass."""
        if self.agg_type == "avg":
            # mean-pooling
            return torch.mean(x, dim=1)
        if self.agg_type == "w_avg":
            # weighted mean-pooling
            return torch.sum(x * self.agg_weight, dim=1)
        raise Exception("unexpected agg type: {}".forward(self.agg_type))


"""
================================================
Encoder Core Networks (Abstract class)
================================================
"""
class EncoderCore(Module):
    """
    Abstract class used to categorize all cores used to encode observations
    """
    def __init__(self, input_shape):
        self.input_shape = input_shape
        super(EncoderCore, self).__init__()

    def __init_subclass__(cls, **kwargs):
        """
        Hook method to automatically register all valid subclasses so we can keep track of valid observation encoders
        in a global dict.

        This global dict stores mapping from observation encoder network name to class.
        We keep track of these registries to enable automated class inference at runtime, allowing
        users to simply extend our base encoder class and refer to that class in string form
        in their config, without having to manually register their class internally.
        This also future-proofs us for any additional encoder classes we would
        like to add ourselves.
        """
        ObsUtils.register_encoder_core(cls)


"""
================================================
Visual Core Networks (Backbone + Pool)
================================================
"""
class VisualCore(EncoderCore, ConvBase):
    """
    A network block that combines a visual backbone network with optional pooling
    and linear layers.
    """
    def __init__(
        self,
        input_shape,
        backbone_class,
        backbone_kwargs,
        pool_class=None,
        pool_kwargs=None,
        flatten=True,
        feature_dimension=None,
    ):
        """
        Args:
            input_shape (tuple): shape of input (not including batch dimension)
            backbone_class (str): class name for the visual backbone network (e.g.: ResNet18)
            backbone_kwargs (dict): kwargs for the visual backbone network
            pool_class (str): class name for the visual feature pooler (optional)
                Common options are "SpatialSoftmax" and "SpatialMeanPool"
            pool_kwargs (dict): kwargs for the visual feature pooler (optional)
            flatten (bool): whether to flatten the visual feature
            feature_dimension (int): if not None, add a Linear layer to
                project output into a desired feature dimension
        """
        super(VisualCore, self).__init__(input_shape=input_shape)
        self.flatten = flatten

        # add input channel dimension to visual core inputs
        backbone_kwargs["input_channel"] = input_shape[0]

        # extract only relevant kwargs for this specific backbone
        backbone_kwargs = extract_class_init_kwargs_from_dict(cls=eval(backbone_class), dic=backbone_kwargs, copy=True)

        # visual backbone
        assert isinstance(backbone_class, str)
        self.backbone = eval(backbone_class)(**backbone_kwargs)

        assert isinstance(self.backbone, ConvBase)

        feat_shape = self.backbone.output_shape(input_shape)
        net_list = [self.backbone]

        # maybe make pool net
        if pool_class is not None:
            assert isinstance(pool_class, str)
            # feed output shape of backbone to pool net
            if pool_kwargs is None:
                pool_kwargs = dict()
            # extract only relevant kwargs for this specific backbone
            pool_kwargs["input_shape"] = feat_shape
            pool_kwargs = extract_class_init_kwargs_from_dict(cls=eval(pool_class), dic=pool_kwargs, copy=True)
            self.pool = eval(pool_class)(**pool_kwargs)
            assert isinstance(self.pool, Module)

            feat_shape = self.pool.output_shape(feat_shape)
            net_list.append(self.pool)
        else:
            self.pool = None

        # flatten layer
        if self.flatten:
            net_list.append(torch.nn.Flatten(start_dim=1, end_dim=-1))

        # maybe linear layer
        self.feature_dimension = feature_dimension
        if feature_dimension is not None:
            assert self.flatten
            linear = torch.nn.Linear(int(np.prod(feat_shape)), feature_dimension)
            net_list.append(linear)

        self.nets = nn.Sequential(*net_list)

    def output_shape(self, input_shape):
        """
        Function to compute output shape from inputs to this module. 

        Args:
            input_shape (iterable of int): shape of input. Does not include batch dimension.
                Some modules may not need this argument, if their output does not depend 
                on the size of the input, or if they assume fixed size input.

        Returns:
            out_shape ([int]): list of integers corresponding to output shape
        """
        if self.feature_dimension is not None:
            # linear output
            return [self.feature_dimension]
        feat_shape = self.backbone.output_shape(input_shape)
        if self.pool is not None:
            # pool output
            feat_shape = self.pool.output_shape(feat_shape)
        # backbone + flat output
        if self.flatten:
            return [np.prod(feat_shape)]
        else:
            return feat_shape

    def forward(self, inputs):
        """
        Forward pass through visual core.
        """
        ndim = len(self.input_shape)
        assert tuple(inputs.shape)[-ndim:] == tuple(self.input_shape)
        return super(VisualCore, self).forward(inputs)

    def __repr__(self):
        """Pretty print network."""
        header = '{}'.format(str(self.__class__.__name__))
        msg = ''
        indent = ' ' * 2
        msg += textwrap.indent(
            "\ninput_shape={}\noutput_shape={}".format(self.input_shape, self.output_shape(self.input_shape)), indent)
        msg += textwrap.indent("\nbackbone_net={}".format(self.backbone), indent)
        msg += textwrap.indent("\npool_net={}".format(self.pool), indent)
        msg = header + '(' + msg + '\n)'
        return msg

## TODO(VS) cleanup comments
## PointNet (https://arxiv.org/pdf/1612.00593.pdf) ##
## Source: https://github.com/fxia22/pointnet.pytorch/blob/f0c2430b0b1529e3f76fb5d6cd6ca14be763d975/pointnet/model.py
class STN3d(nn.Module):
    """ Helper class for PointNetFeat. Predicts 3x3 input transformation matrices. """
    def __init__(self, batch_norm=True, channel_multiplier=1):
        super(STN3d, self).__init__()
        # input shape: (B, 3, N)
        self.channel_multiplier = channel_multiplier
        self.conv1 = torch.nn.Conv1d(3, 64*channel_multiplier, 1) # out shape: (B, 64, N)
        self.conv2 = torch.nn.Conv1d(64*channel_multiplier, 128*channel_multiplier, 1) # out shape: (B, 128, N)
        self.conv3 = torch.nn.Conv1d(128*channel_multiplier, 1024*channel_multiplier, 1) # out shape: (B, 1024, N)
        self.fc1 = nn.Linear(1024*channel_multiplier, 512*channel_multiplier)
        self.fc2 = nn.Linear(512*channel_multiplier, 256*channel_multiplier)
        self.fc3 = nn.Linear(256*channel_multiplier, 9)
        self.relu = nn.ReLU()
        self.batch_norm = batch_norm

        self.act1 = lambda x: self.relu(x)
        self.act2 = lambda x: self.relu(x)
        self.act3 = lambda x: self.relu(x)
        self.act4 = lambda x: self.relu(x)
        self.act5 = lambda x: self.relu(x)
        if self.batch_norm:
            self.bn1 = nn.BatchNorm1d(64*channel_multiplier)
            self.bn2 = nn.BatchNorm1d(128*channel_multiplier)
            self.bn3 = nn.BatchNorm1d(1024*channel_multiplier)
            self.bn4 = nn.BatchNorm1d(512*channel_multiplier)
            self.bn5 = nn.BatchNorm1d(256*channel_multiplier)
            self.act1 = lambda x: self.bn1(self.relu(x))
            self.act2 = lambda x: self.bn2(self.relu(x))
            self.act3 = lambda x: self.bn3(self.relu(x))
            self.act4 = lambda x: self.bn4(self.relu(x))
            self.act5 = lambda x: self.bn5(self.relu(x))

    def forward(self, x):
        # input shape: (B, 3, N)
        batchsize = x.size()[0]
        x = self.act1(self.conv1(x)) # out shape: (B, 64, N)
        x = self.act2(self.conv2(x)) # out shape: (B, 128, N)
        x = self.act3(self.conv3(x)) # out shape: (B, 1024, N)
        x = torch.max(x, 2, keepdim=True)[0] # max-pool across points dimension; out shape: (B, 1024, 1)
        x = x.view(-1, 1024*self.channel_multiplier) # out shape: (B, 1024)

        x = self.act4(self.fc1(x)) # out shape: (B, 512)
        x = self.act5(self.fc2(x)) # out shape: (B, 256)
        x = self.fc3(x) # out shape: (B, 9)

        iden = torch.autograd.Variable(torch.from_numpy(np.array([1,0,0,0,1,0,0,0,1]).astype(np.float32))).view(1,9).repeat(batchsize, 1)
        iden = iden.to(x.device)
        x = x + iden
        x = x.view(-1, 3, 3)
        return x


class STNkd(nn.Module):
    """ Helper class for PointNetFeat. Predicts feature transformation matrices. """
    def __init__(self, k=64, batch_norm=True, channel_multiplier=1):
        super(STNkd, self).__init__()
        self.channel_multiplier = channel_multiplier
        self.conv1 = torch.nn.Conv1d(k, 64*self.channel_multiplier, 1)
        self.conv2 = torch.nn.Conv1d(64*self.channel_multiplier, 128*self.channel_multiplier, 1)
        self.conv3 = torch.nn.Conv1d(128*self.channel_multiplier, 1024*self.channel_multiplier, 1)
        self.fc1 = nn.Linear(1024*self.channel_multiplier, 512*self.channel_multiplier)
        self.fc2 = nn.Linear(512*self.channel_multiplier, 256*self.channel_multiplier)
        self.fc3 = nn.Linear(256*self.channel_multiplier, k*k)
        self.relu = nn.ReLU()
        self.batch_norm = batch_norm

        self.act1 = lambda x: self.relu(x)
        self.act2 = lambda x: self.relu(x)
        self.act3 = lambda x: self.relu(x)
        self.act4 = lambda x: self.relu(x)
        self.act5 = lambda x: self.relu(x)
        if self.batch_norm:
            self.bn1 = nn.BatchNorm1d(64*channel_multiplier)
            self.bn2 = nn.BatchNorm1d(128*channel_multiplier)
            self.bn3 = nn.BatchNorm1d(1024*channel_multiplier)
            self.bn4 = nn.BatchNorm1d(512*channel_multiplier)
            self.bn5 = nn.BatchNorm1d(256*channel_multiplier)
            self.act1 = lambda x: self.bn1(self.relu(x))
            self.act2 = lambda x: self.bn2(self.relu(x))
            self.act3 = lambda x: self.bn3(self.relu(x))
            self.act4 = lambda x: self.bn4(self.relu(x))
            self.act5 = lambda x: self.bn5(self.relu(x))

        self.k = k

    def forward(self, x):
        batchsize = x.size()[0]
        x = self.act1(self.conv1(x))
        x = self.act2(self.conv2(x))
        x = self.act3(self.conv3(x))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024*self.channel_multiplier)

        x = self.act4(self.fc1(x))
        x = self.act5(self.fc2(x))
        x = self.fc3(x)

        iden = torch.nn.Variable(torch.from_numpy(np.eye(self.k).flatten().astype(np.float32))).view(1,self.k*self.k).repeat(batchsize,1)
        iden = iden.to(x.device)
        x = x + iden
        x = x.view(-1, self.k, self.k)
        return x

class PointNetFeat(nn.Module):
    """ Helper class for PointNet. Predicts features from inputs. """
    def __init__(self, global_feat_only=True, feature_transform=False, batch_norm=True, channel_multiplier=1, pool="max"):
        super(PointNetFeat, self).__init__()
        self.channel_multiplier = channel_multiplier
        self.conv1 = torch.nn.Conv1d(3, 64*self.channel_multiplier, 1)
        self.conv2 = torch.nn.Conv1d(64*self.channel_multiplier, 128*self.channel_multiplier, 1)
        self.conv3 = torch.nn.Conv1d(128*self.channel_multiplier, 1024, 1)
        self.relu = nn.ReLU()
        self.batch_norm = batch_norm
        self.pool = pool
        assert self.pool in ["max", "avg"], f"Pooling method {self.pool} not supported."
        self.stn = STN3d(batch_norm=self.batch_norm, channel_multiplier=self.channel_multiplier) # input transformation class

        self.act1 = lambda x: self.relu(x)
        self.act2 = lambda x: self.relu(x)
        self.act3 = lambda x: x
        if self.batch_norm:
            self.bn1 = nn.BatchNorm1d(64)
            self.bn2 = nn.BatchNorm1d(128)
            self.bn3 = nn.BatchNorm1d(1024)
            self.act1 = lambda x: self.bn1(self.relu(x))
            self.act2 = lambda x: self.bn2(self.relu(x))
            self.act3 = lambda x: self.bn3(x)


        self.global_feat = global_feat_only
        self.feature_transform = feature_transform
        if self.feature_transform:
            self.fstn = STNkd(k=64, batch_norm=self.batch_norm, channel_multiplier=self.channel_multiplier) # feature transformation class

    def forward(self, x):
        n_pts = x.size()[2] # input shape (B, 3, n_pts)
        trans = self.stn(x) # (B, 3, 3) input transformation matrices
        x = x.transpose(2, 1) # (B, 3, N) -> (B, N, 3)
        x = torch.bmm(x, trans) # (B, N, 3)
        x = x.transpose(2, 1) # after input transform; (B, N, 3) -> (B, 3, N)
        x = self.act1(self.conv1(x)) # final feature shape (B, 64, N)

        if self.feature_transform:
            trans_feat = self.fstn(x) # (B, 64, 64) feature transformation matrices
            x = x.transpose(2, 1) # (B, 64, N) -> (B, N, 64)
            x = torch.bmm(x, trans_feat) # (B, N, 64)
            x = x.transpose(2, 1) # (B, N, 64) -> (B, 64, N); after feature transform
        else:
            trans_feat = None

        pointfeat = x # final feature shape (B, 64, N)
        x = self.act2(self.conv2(x)) # out shape (B, 128, N)
        x = self.act3(self.conv3(x)) # out shape (B, 1024, N)
        if self.pool == "max":
            x = torch.max(x, 2, keepdim=True)[0] # max-pool across points dimension; shape (B, 1024, 1)
        elif self.pool == "avg":
            x = torch.mean(x, 2, keepdim=True) # avg-pool across points dimension; shape (B, 1024, 1)
        x = x.view(-1, 1024) # shape (B, 1024)
        if self.global_feat:
            return x, trans, trans_feat
        else:
            # concatenating the max-pooled feature with the previously computed point-wise features
            x = x.view(-1, 1024, 1).repeat(1, 1, n_pts)
            return torch.cat([x, pointfeat], 1), trans, trans_feat

## TODO(VS) remove classifier code
# class PointNetClassifier(nn.Module):
#     def __init__(self, classes=2, feature_transform=False):
#         super(PointNetClassifier, self).__init__()
#         self.feature_transform = feature_transform
#         self.feat = PointNetFeat(global_feat=True, feature_transform=feature_transform)
#         self.fc1 = nn.Linear(1024, 512)
#         self.fc2 = nn.Linear(512, 256)
#         self.fc3 = nn.Linear(256, classes)
#         self.dropout = nn.Dropout(p=0.3)
#         self.bn1 = nn.BatchNorm1d(512)
#         self.bn2 = nn.BatchNorm1d(256)
#         self.relu = nn.ReLU()

#     def forward(self, x):
#         x, trans, trans_feat = self.feat(x)
#         x = F.relu(self.bn1(self.fc1(x)))
#         x = F.relu(self.bn2(self.dropout(self.fc2(x))))
#         x = self.fc3(x)
#         return F.log_softmax(x, dim=1), trans, trans_feat

## TODO(VS) refactor STN3d, STNkd, PointNetFeat, PointNetCore into a single PointNet class
class PointCloudCore(EncoderCore):
    def __init__(self, input_shape, batch_norm=True, channel_multiplier=1, pool="max"):
        super(PointCloudCore, self).__init__(input_shape)
        self.input_shape = input_shape
        self.batch_norm = batch_norm
        self.channel_multiplier = channel_multiplier
        self.pool = pool
        self.pointnet = PointNetFeat(global_feat_only=True, feature_transform=False, batch_norm=self.batch_norm, channel_multiplier=self.channel_multiplier, pool=self.pool)

    def output_shape(self, input_shape):
        return [1024]

    def forward(self, inputs): #TODO(VS)
        feats, _, _ = self.pointnet(inputs)
        return feats
    
    ## TODO(VS)
    # def __repr__(self):
    #     raise NotImplementedError


import robomimic.ndf_robot.src.ndf_robot.model as ndf_model
class ResnetPointnetCore(EncoderCore):
    ''' DGCNN-based VNN encoder network with ResNet blocks.

    Args:
        c_dim (int): dimension of latent code c
        dim (int): input points dimension
        hidden_dim (int): hidden dimension of the network
    '''

    def __init__(self, input_shape, c_dim=128, dim=3, hidden_dim=128, k=20, meta_output=None):
        super().__init__(input_shape)
        self._input_shape = input_shape
        self.c_dim = c_dim
        self.pointnet = ndf_model.VNN_ResnetPointnet(c_dim=c_dim, dim=dim, hidden_dim=hidden_dim, k=k, meta_output=meta_output)
        
    def output_shape(self, input_shape):
        return [self.c_dim*3]

    def forward(self, inputs):
        output = self.pointnet(inputs)
        batch_size = output.shape[0]
        return output.reshape([batch_size, -1])


"""
================================================
Scan Core Networks (Conv1D Sequential + Pool)
================================================
"""
class ScanCore(EncoderCore, ConvBase):
    """
    A network block that combines a Conv1D backbone network with optional pooling
    and linear layers.
    """
    def __init__(
        self,
        input_shape,
        conv_kwargs,
        conv_activation="relu",
        pool_class=None,
        pool_kwargs=None,
        flatten=True,
        feature_dimension=None,
    ):
        """
        Args:
            input_shape (tuple): shape of input (not including batch dimension)
            conv_kwargs (dict): kwargs for the conv1d backbone network. Should contain lists for the following values:
                out_channels (int)
                kernel_size (int)
                stride (int)
                ...
            conv_activation (str or None): Activation to use between conv layers. Default is relu.
                Currently, valid options are {relu}
            pool_class (str): class name for the visual feature pooler (optional)
                Common options are "SpatialSoftmax" and "SpatialMeanPool"
            pool_kwargs (dict): kwargs for the visual feature pooler (optional)
            flatten (bool): whether to flatten the network output
            feature_dimension (int): if not None, add a Linear layer to
                project output into a desired feature dimension (note: flatten must be set to True!)
        """
        super(ScanCore, self).__init__(input_shape=input_shape)
        self.flatten = flatten
        self.feature_dimension = feature_dimension

        # Generate backbone network
        self.backbone = Conv1dBase(
            input_channel=1,
            activation=conv_activation,
            **conv_kwargs,
        )
        feat_shape = self.backbone.output_shape(input_shape=input_shape)

        # Create netlist of all generated networks
        net_list = [self.backbone]

        # Possibly add pooling network
        if pool_class is not None:
            # Add an unsqueeze network so that the shape is correct to pass to pooling network
            self.unsqueeze = Unsqueeze(dim=-1)
            net_list.append(self.unsqueeze)
            # Get output shape
            feat_shape = self.unsqueeze.output_shape(feat_shape)
            # Create pooling network
            self.pool = eval(pool_class)(input_shape=feat_shape, **pool_kwargs)
            net_list.append(self.pool)
            feat_shape = self.pool.output_shape(feat_shape)
        else:
            self.unsqueeze, self.pool = None, None

        # flatten layer
        if self.flatten:
            net_list.append(torch.nn.Flatten(start_dim=1, end_dim=-1))

        # maybe linear layer
        if self.feature_dimension is not None:
            assert self.flatten
            linear = torch.nn.Linear(int(np.prod(feat_shape)), self.feature_dimension)
            net_list.append(linear)

        # Generate final network
        self.nets = nn.Sequential(*net_list)

    def output_shape(self, input_shape):
        """
        Function to compute output shape from inputs to this module.

        Args:
            input_shape (iterable of int): shape of input. Does not include batch dimension.
                Some modules may not need this argument, if their output does not depend
                on the size of the input, or if they assume fixed size input.

        Returns:
            out_shape ([int]): list of integers corresponding to output shape
        """
        if self.feature_dimension is not None:
            # linear output
            return [self.feature_dimension]
        feat_shape = self.backbone.output_shape(input_shape)
        if self.pool is not None:
            # pool output
            feat_shape = self.pool.output_shape(self.unsqueeze.output_shape(feat_shape))
        # backbone + flat output
        return [np.prod(feat_shape)] if self.flatten else feat_shape

    def forward(self, inputs):
        """
        Forward pass through visual core.
        """
        ndim = len(self.input_shape)
        assert tuple(inputs.shape)[-ndim:] == tuple(self.input_shape)
        return super(ScanCore, self).forward(inputs)

    def __repr__(self):
        """Pretty print network."""
        header = '{}'.format(str(self.__class__.__name__))
        msg = ''
        indent = ' ' * 2
        msg += textwrap.indent(
            "\ninput_shape={}\noutput_shape={}".format(self.input_shape, self.output_shape(self.input_shape)), indent)
        msg += textwrap.indent("\nbackbone_net={}".format(self.backbone), indent)
        msg += textwrap.indent("\npool_net={}".format(self.pool), indent)
        msg = header + '(' + msg + '\n)'
        return msg



"""
================================================
Observation Randomizer Networks
================================================
"""
class Randomizer(Module):
    """
    Base class for randomizer networks. Each randomizer should implement the @output_shape_in,
    @output_shape_out, @forward_in, and @forward_out methods. The randomizer's @forward_in
    method is invoked on raw inputs, and @forward_out is invoked on processed inputs
    (usually processed by a @VisualCore instance). Note that the self.training property
    can be used to change the randomizer's behavior at train vs. test time.
    """
    def __init__(self):
        super(Randomizer, self).__init__()

    def __init_subclass__(cls, **kwargs):
        """
        Hook method to automatically register all valid subclasses so we can keep track of valid observation randomizers
        in a global dict.

        This global dict stores mapping from observation randomizer network name to class.
        We keep track of these registries to enable automated class inference at runtime, allowing
        users to simply extend our base randomizer class and refer to that class in string form
        in their config, without having to manually register their class internally.
        This also future-proofs us for any additional randomizer classes we would
        like to add ourselves.
        """
        ObsUtils.register_randomizer(cls)

    def output_shape(self, input_shape=None):
        """
        This function is unused. See @output_shape_in and @output_shape_out.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def output_shape_in(self, input_shape=None):
        """
        Function to compute output shape from inputs to this module. Corresponds to
        the @forward_in operation, where raw inputs (usually observation modalities)
        are passed in.

        Args:
            input_shape (iterable of int): shape of input. Does not include batch dimension.
                Some modules may not need this argument, if their output does not depend 
                on the size of the input, or if they assume fixed size input.

        Returns:
            out_shape ([int]): list of integers corresponding to output shape
        """
        raise NotImplementedError

    @abc.abstractmethod
    def output_shape_out(self, input_shape=None):
        """
        Function to compute output shape from inputs to this module. Corresponds to
        the @forward_out operation, where processed inputs (usually encoded observation
        modalities) are passed in.

        Args:
            input_shape (iterable of int): shape of input. Does not include batch dimension.
                Some modules may not need this argument, if their output does not depend 
                on the size of the input, or if they assume fixed size input.

        Returns:
            out_shape ([int]): list of integers corresponding to output shape
        """
        raise NotImplementedError

    @abc.abstractmethod
    def forward_in(self, inputs):
        """
        Randomize raw inputs.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def forward_out(self, inputs):
        """
        Processing for network outputs.
        """
        return inputs


class CropRandomizer(Randomizer):
    """
    Randomly sample crops at input, and then average across crop features at output.
    """
    def __init__(
        self,
        input_shape,
        crop_height, 
        crop_width, 
        num_crops=1,
        pos_enc=False,
    ):
        """
        Args:
            input_shape (tuple, list): shape of input (not including batch dimension)
            crop_height (int): crop height
            crop_width (int): crop width
            num_crops (int): number of random crops to take
            pos_enc (bool): if True, add 2 channels to the output to encode the spatial
                location of the cropped pixels in the source image
        """
        super(CropRandomizer, self).__init__()

        assert len(input_shape) == 3 # (C, H, W)
        assert crop_height < input_shape[1]
        assert crop_width < input_shape[2]

        self.input_shape = input_shape
        self.crop_height = crop_height
        self.crop_width = crop_width
        self.num_crops = num_crops
        self.pos_enc = pos_enc

    def output_shape_in(self, input_shape=None):
        """
        Function to compute output shape from inputs to this module. Corresponds to
        the @forward_in operation, where raw inputs (usually observation modalities)
        are passed in.

        Args:
            input_shape (iterable of int): shape of input. Does not include batch dimension.
                Some modules may not need this argument, if their output does not depend 
                on the size of the input, or if they assume fixed size input.

        Returns:
            out_shape ([int]): list of integers corresponding to output shape
        """

        # outputs are shape (C, CH, CW), or maybe C + 2 if using position encoding, because
        # the number of crops are reshaped into the batch dimension, increasing the batch
        # size from B to B * N
        out_c = self.input_shape[0] + 2 if self.pos_enc else self.input_shape[0]
        return [out_c, self.crop_height, self.crop_width]

    def output_shape_out(self, input_shape=None):
        """
        Function to compute output shape from inputs to this module. Corresponds to
        the @forward_out operation, where processed inputs (usually encoded observation
        modalities) are passed in.

        Args:
            input_shape (iterable of int): shape of input. Does not include batch dimension.
                Some modules may not need this argument, if their output does not depend 
                on the size of the input, or if they assume fixed size input.

        Returns:
            out_shape ([int]): list of integers corresponding to output shape
        """
        
        # since the forward_out operation splits [B * N, ...] -> [B, N, ...]
        # and then pools to result in [B, ...], only the batch dimension changes,
        # and so the other dimensions retain their shape.
        return list(input_shape)

    def forward_in(self, inputs):
        """
        Samples N random crops for each input in the batch, and then reshapes
        inputs to [B * N, ...].
        """
        assert len(inputs.shape) >= 3 # must have at least (C, H, W) dimensions
        out, _ = ObsUtils.sample_random_image_crops(
            images=inputs,
            crop_height=self.crop_height, 
            crop_width=self.crop_width, 
            num_crops=self.num_crops,
            pos_enc=self.pos_enc,
        )
        # [B, N, ...] -> [B * N, ...]
        return TensorUtils.join_dimensions(out, 0, 1)

    def forward_out(self, inputs):
        """
        Splits the outputs from shape [B * N, ...] -> [B, N, ...] and then average across N
        to result in shape [B, ...] to make sure the network output is consistent with
        what would have happened if there were no randomization.
        """
        batch_size = (inputs.shape[0] // self.num_crops)
        out = TensorUtils.reshape_dimensions(inputs, begin_axis=0, end_axis=0, 
            target_dims=(batch_size, self.num_crops))
        return out.mean(dim=1)

    def __repr__(self):
        """Pretty print network."""
        header = '{}'.format(str(self.__class__.__name__))
        msg = header + "(input_shape={}, crop_size=[{}, {}], num_crops={})".format(
            self.input_shape, self.crop_height, self.crop_width, self.num_crops)
        return msg
