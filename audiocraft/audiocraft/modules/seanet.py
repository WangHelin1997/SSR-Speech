# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import typing as tp
import torch
import numpy as np
import torch.nn as nn

from .conv import StreamableConv1d, StreamableConvTranspose1d
from .lstm import StreamableLSTM


class SEANetResnetBlock(nn.Module):
    """Residual block from SEANet model.

    Args:
        dim (int): Dimension of the input/output.
        kernel_sizes (list): List of kernel sizes for the convolutions.
        dilations (list): List of dilations for the convolutions.
        activation (str): Activation function.
        activation_params (dict): Parameters to provide to the activation function.
        norm (str): Normalization method.
        norm_params (dict): Parameters to provide to the underlying normalization used along with the convolution.
        causal (bool): Whether to use fully causal convolution.
        pad_mode (str): Padding mode for the convolutions.
        compress (int): Reduced dimensionality in residual branches (from Demucs v3).
        true_skip (bool): Whether to use true skip connection or a simple
            (streamable) convolution as the skip connection.
    """
    def __init__(self, dim: int, kernel_sizes: tp.List[int] = [3, 1], dilations: tp.List[int] = [1, 1],
                 activation: str = 'ELU', activation_params: dict = {'alpha': 1.0},
                 norm: str = 'none', norm_params: tp.Dict[str, tp.Any] = {}, causal: bool = False,
                 pad_mode: str = 'reflect', compress: int = 2, true_skip: bool = True):
        super().__init__()
        assert len(kernel_sizes) == len(dilations), 'Number of kernel sizes should match number of dilations'
        act = getattr(nn, activation)
        hidden = dim // compress
        block = []
        for i, (kernel_size, dilation) in enumerate(zip(kernel_sizes, dilations)):
            in_chs = dim if i == 0 else hidden
            out_chs = dim if i == len(kernel_sizes) - 1 else hidden
            block += [
                act(**activation_params),
                StreamableConv1d(in_chs, out_chs, kernel_size=kernel_size, dilation=dilation,
                                 norm=norm, norm_kwargs=norm_params,
                                 causal=causal, pad_mode=pad_mode),
            ]
        self.block = nn.Sequential(*block)
        self.shortcut: nn.Module
        if true_skip:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = StreamableConv1d(dim, dim, kernel_size=1, norm=norm, norm_kwargs=norm_params,
                                             causal=causal, pad_mode=pad_mode)

    def forward(self, x):
        return self.shortcut(x) + self.block(x)


class SEANetEncoder(nn.Module):
    """SEANet encoder.

    Args:
        channels (int): Audio channels.
        dimension (int): Intermediate representation dimension.
        n_filters (int): Base width for the model.
        n_residual_layers (int): nb of residual layers.
        ratios (Sequence[int]): kernel size and stride ratios. The encoder uses downsampling ratios instead of
            upsampling ratios, hence it will use the ratios in the reverse order to the ones specified here
            that must match the decoder order. We use the decoder order as some models may only employ the decoder.
        activation (str): Activation function.
        activation_params (dict): Parameters to provide to the activation function.
        norm (str): Normalization method.
        norm_params (dict): Parameters to provide to the underlying normalization used along with the convolution.
        kernel_size (int): Kernel size for the initial convolution.
        last_kernel_size (int): Kernel size for the initial convolution.
        residual_kernel_size (int): Kernel size for the residual layers.
        dilation_base (int): How much to increase the dilation with each layer.
        causal (bool): Whether to use fully causal convolution.
        pad_mode (str): Padding mode for the convolutions.
        true_skip (bool): Whether to use true skip connection or a simple
            (streamable) convolution as the skip connection in the residual network blocks.
        compress (int): Reduced dimensionality in residual branches (from Demucs v3).
        lstm (int): Number of LSTM layers at the end of the encoder.
        disable_norm_outer_blocks (int): Number of blocks for which we don't apply norm.
            For the encoder, it corresponds to the N first blocks.
    """
    def __init__(self, channels: int = 1, dimension: int = 128, n_filters: int = 32, n_residual_layers: int = 3,
                 ratios: tp.List[int] = [8, 5, 4, 2], activation: str = 'ELU', activation_params: dict = {'alpha': 1.0},
                 norm: str = 'none', norm_params: tp.Dict[str, tp.Any] = {}, kernel_size: int = 7,
                 last_kernel_size: int = 7, residual_kernel_size: int = 3, dilation_base: int = 2, causal: bool = False,
                 pad_mode: str = 'reflect', true_skip: bool = True, compress: int = 2, lstm: int = 0,
                 disable_norm_outer_blocks: int = 0):
        super().__init__()
        self.channels = channels
        self.dimension = dimension
        self.n_filters = n_filters
        self.ratios = list(reversed(ratios))
        del ratios
        self.n_residual_layers = n_residual_layers
        self.hop_length = np.prod(self.ratios)
        self.n_blocks = len(self.ratios) + 2  # first and last conv + residual blocks
        self.disable_norm_outer_blocks = disable_norm_outer_blocks
        assert self.disable_norm_outer_blocks >= 0 and self.disable_norm_outer_blocks <= self.n_blocks, \
            "Number of blocks for which to disable norm is invalid." \
            "It should be lower or equal to the actual number of blocks in the network and greater or equal to 0."

        act = getattr(nn, activation)
        mult = 1
        model: tp.List[nn.Module] = [
            StreamableConv1d(channels, mult * n_filters, kernel_size,
                             norm='none' if self.disable_norm_outer_blocks >= 1 else norm,
                             norm_kwargs=norm_params, causal=causal, pad_mode=pad_mode)
        ]
        # Downsample to raw audio scale
        for i, ratio in enumerate(self.ratios):
            block_norm = 'none' if self.disable_norm_outer_blocks >= i + 2 else norm
            # Add residual layers
            for j in range(n_residual_layers):
                model += [
                    SEANetResnetBlock(mult * n_filters, kernel_sizes=[residual_kernel_size, 1],
                                      dilations=[dilation_base ** j, 1],
                                      norm=block_norm, norm_params=norm_params,
                                      activation=activation, activation_params=activation_params,
                                      causal=causal, pad_mode=pad_mode, compress=compress, true_skip=true_skip)]

            # Add downsampling layers
            model += [
                act(**activation_params),
                StreamableConv1d(mult * n_filters, mult * n_filters * 2,
                                 kernel_size=ratio * 2, stride=ratio,
                                 norm=block_norm, norm_kwargs=norm_params,
                                 causal=causal, pad_mode=pad_mode),
            ]
            mult *= 2

        if lstm:
            model += [StreamableLSTM(mult * n_filters, num_layers=lstm)]

        model += [
            act(**activation_params),
            StreamableConv1d(mult * n_filters, dimension, last_kernel_size,
                             norm='none' if self.disable_norm_outer_blocks == self.n_blocks else norm,
                             norm_kwargs=norm_params, causal=causal, pad_mode=pad_mode)
        ]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


class SEANetDecoder(nn.Module):
    """SEANet decoder.

    Args:
        channels (int): Audio channels.
        dimension (int): Intermediate representation dimension.
        n_filters (int): Base width for the model.
        n_residual_layers (int): nb of residual layers.
        ratios (Sequence[int]): kernel size and stride ratios.
        activation (str): Activation function.
        activation_params (dict): Parameters to provide to the activation function.
        final_activation (str): Final activation function after all convolutions.
        final_activation_params (dict): Parameters to provide to the activation function.
        norm (str): Normalization method.
        norm_params (dict): Parameters to provide to the underlying normalization used along with the convolution.
        kernel_size (int): Kernel size for the initial convolution.
        last_kernel_size (int): Kernel size for the initial convolution.
        residual_kernel_size (int): Kernel size for the residual layers.
        dilation_base (int): How much to increase the dilation with each layer.
        causal (bool): Whether to use fully causal convolution.
        pad_mode (str): Padding mode for the convolutions.
        true_skip (bool): Whether to use true skip connection or a simple.
            (streamable) convolution as the skip connection in the residual network blocks.
        compress (int): Reduced dimensionality in residual branches (from Demucs v3).
        lstm (int): Number of LSTM layers at the end of the encoder.
        disable_norm_outer_blocks (int): Number of blocks for which we don't apply norm.
            For the decoder, it corresponds to the N last blocks.
        trim_right_ratio (float): Ratio for trimming at the right of the transposed convolution under the causal setup.
            If equal to 1.0, it means that all the trimming is done at the right.
    """
    def __init__(self, channels: int = 1, dimension: int = 128, n_filters: int = 32, n_residual_layers: int = 3,
                 ratios: tp.List[int] = [8, 5, 4, 2], activation: str = 'ELU', activation_params: dict = {'alpha': 1.0},
                 final_activation: tp.Optional[str] = None, final_activation_params: tp.Optional[dict] = None,
                 norm: str = 'none', norm_params: tp.Dict[str, tp.Any] = {}, kernel_size: int = 7,
                 last_kernel_size: int = 7, residual_kernel_size: int = 3, dilation_base: int = 2, causal: bool = False,
                 pad_mode: str = 'reflect', true_skip: bool = True, compress: int = 2, lstm: int = 0,
                 disable_norm_outer_blocks: int = 0, trim_right_ratio: float = 1.0):
        super().__init__()
        self.dimension = dimension
        self.channels = channels
        self.n_filters = n_filters
        self.ratios = ratios
        del ratios
        self.n_residual_layers = n_residual_layers
        self.hop_length = np.prod(self.ratios)
        self.n_blocks = len(self.ratios) + 2  # first and last conv + residual blocks
        self.disable_norm_outer_blocks = disable_norm_outer_blocks
        assert self.disable_norm_outer_blocks >= 0 and self.disable_norm_outer_blocks <= self.n_blocks, \
            "Number of blocks for which to disable norm is invalid." \
            "It should be lower or equal to the actual number of blocks in the network and greater or equal to 0."

        act = getattr(nn, activation)
        mult = int(2 ** len(self.ratios))
        model: tp.List[nn.Module] = [
            StreamableConv1d(dimension, mult * n_filters, kernel_size,
                             norm='none' if self.disable_norm_outer_blocks == self.n_blocks else norm,
                             norm_kwargs=norm_params, causal=causal, pad_mode=pad_mode)
        ]

        if lstm:
            model += [StreamableLSTM(mult * n_filters, num_layers=lstm)]

        # Upsample to raw audio scale
        for i, ratio in enumerate(self.ratios):
            block_norm = 'none' if self.disable_norm_outer_blocks >= self.n_blocks - (i + 1) else norm
            # Add upsampling layers
            model += [
                act(**activation_params),
                StreamableConvTranspose1d(mult * n_filters, mult * n_filters // 2,
                                          kernel_size=ratio * 2, stride=ratio,
                                          norm=block_norm, norm_kwargs=norm_params,
                                          causal=causal, trim_right_ratio=trim_right_ratio),
            ]
            # Add residual layers
            for j in range(n_residual_layers):
                model += [
                    SEANetResnetBlock(mult * n_filters // 2, kernel_sizes=[residual_kernel_size, 1],
                                      dilations=[dilation_base ** j, 1],
                                      activation=activation, activation_params=activation_params,
                                      norm=block_norm, norm_params=norm_params, causal=causal,
                                      pad_mode=pad_mode, compress=compress, true_skip=true_skip)]

            mult //= 2

        # Add final layers
        model += [
            act(**activation_params),
            StreamableConv1d(n_filters, channels, last_kernel_size,
                             norm='none' if self.disable_norm_outer_blocks >= 1 else norm,
                             norm_kwargs=norm_params, causal=causal, pad_mode=pad_mode)
        ]
        # Add optional final activation to decoder (eg. tanh)
        if final_activation is not None:
            final_act = getattr(nn, final_activation)
            final_activation_params = final_activation_params or {}
            model += [
                final_act(**final_activation_params)
            ]
        self.model = nn.Sequential(*model)

    def forward(self, z):
        y = self.model(z)
        return y

    
# class WMSEANetDecoder(nn.Module):
#     """Watermark SEANet decoder.

#     Args:
#         channels (int): Audio channels.
#         dimension (int): Intermediate representation dimension.
#         n_filters (int): Base width for the model.
#         n_residual_layers (int): nb of residual layers.
#         ratios (Sequence[int]): kernel size and stride ratios.
#         activation (str): Activation function.
#         activation_params (dict): Parameters to provide to the activation function.
#         final_activation (str): Final activation function after all convolutions.
#         final_activation_params (dict): Parameters to provide to the activation function.
#         norm (str): Normalization method.
#         norm_params (dict): Parameters to provide to the underlying normalization used along with the convolution.
#         kernel_size (int): Kernel size for the initial convolution.
#         last_kernel_size (int): Kernel size for the initial convolution.
#         residual_kernel_size (int): Kernel size for the residual layers.
#         dilation_base (int): How much to increase the dilation with each layer.
#         causal (bool): Whether to use fully causal convolution.
#         pad_mode (str): Padding mode for the convolutions.
#         true_skip (bool): Whether to use true skip connection or a simple.
#             (streamable) convolution as the skip connection in the residual network blocks.
#         compress (int): Reduced dimensionality in residual branches (from Demucs v3).
#         lstm (int): Number of LSTM layers at the end of the encoder.
#         disable_norm_outer_blocks (int): Number of blocks for which we don't apply norm.
#             For the decoder, it corresponds to the N last blocks.
#         trim_right_ratio (float): Ratio for trimming at the right of the transposed convolution under the causal setup.
#             If equal to 1.0, it means that all the trimming is done at the right.
#     """
#     def __init__(self, channels: int = 1, dimension: int = 128, n_filters: int = 32, n_residual_layers: int = 3,
#                  ratios: tp.List[int] = [8, 5, 4, 2], activation: str = 'ELU', activation_params: dict = {'alpha': 1.0},
#                  final_activation: tp.Optional[str] = None, final_activation_params: tp.Optional[dict] = None,
#                  norm: str = 'none', norm_params: tp.Dict[str, tp.Any] = {}, kernel_size: int = 7,
#                  last_kernel_size: int = 7, residual_kernel_size: int = 3, dilation_base: int = 2, causal: bool = False,
#                  pad_mode: str = 'reflect', true_skip: bool = True, compress: int = 2, lstm: int = 0,
#                  disable_norm_outer_blocks: int = 0, trim_right_ratio: float = 1.0):
#         super().__init__()
#         self.dimension = dimension
#         self.channels = channels
#         self.n_filters = n_filters
#         self.ratios = ratios
#         del ratios
#         self.n_residual_layers = n_residual_layers
#         self.hop_length = np.prod(self.ratios)
#         self.n_blocks = len(self.ratios) + 2  # first and last conv + residual blocks
#         self.disable_norm_outer_blocks = disable_norm_outer_blocks
#         assert self.disable_norm_outer_blocks >= 0 and self.disable_norm_outer_blocks <= self.n_blocks, \
#             "Number of blocks for which to disable norm is invalid." \
#             "It should be lower or equal to the actual number of blocks in the network and greater or equal to 0."

#         act = getattr(nn, activation)
#         mult = int(2 ** len(self.ratios))
#         model: tp.List[nn.Module] = [
#             StreamableConv1d(dimension, mult * n_filters, kernel_size,
#                              norm='none' if self.disable_norm_outer_blocks == self.n_blocks else norm,
#                              norm_kwargs=norm_params, causal=causal, pad_mode=pad_mode)
#         ]

#         if lstm:
#             model += [StreamableLSTM(mult * n_filters, num_layers=lstm)]

#         # Upsample to raw audio scale
#         for i, ratio in enumerate(self.ratios):
#             block_norm = 'none' if self.disable_norm_outer_blocks >= self.n_blocks - (i + 1) else norm
#             # Add upsampling layers
#             model += [
#                 act(**activation_params),
#                 StreamableConvTranspose1d(mult * n_filters, mult * n_filters // 2,
#                                           kernel_size=ratio * 2, stride=ratio,
#                                           norm=block_norm, norm_kwargs=norm_params,
#                                           causal=causal, trim_right_ratio=trim_right_ratio),
#             ]
#             # Add residual layers
#             for j in range(n_residual_layers):
#                 model += [
#                     SEANetResnetBlock(mult * n_filters // 2, kernel_sizes=[residual_kernel_size, 1],
#                                       dilations=[dilation_base ** j, 1],
#                                       activation=activation, activation_params=activation_params,
#                                       norm=block_norm, norm_params=norm_params, causal=causal,
#                                       pad_mode=pad_mode, compress=compress, true_skip=true_skip)]

#             mult //= 2

#         # Add final layers
#         model += [
#             act(**activation_params),
#             StreamableConv1d(n_filters, channels, last_kernel_size,
#                              norm='none' if self.disable_norm_outer_blocks >= 1 else norm,
#                              norm_kwargs=norm_params, causal=causal, pad_mode=pad_mode)
#         ]
#         # Add optional final activation to decoder (eg. tanh)
#         if final_activation is not None:
#             final_act = getattr(nn, final_activation)
#             final_activation_params = final_activation_params or {}
#             model += [
#                 final_act(**final_activation_params)
#             ]
#         self.model = nn.Sequential(*model)

#         self.wm_embed = nn.Embedding(2, dimension // 16, max_norm=True)

#         self.wm_encoder = SEANetEncoder(channels=channels, dimension=dimension, n_filters=n_filters, n_residual_layers=n_residual_layers,
#                  ratios=self.ratios, activation=activation, activation_params=activation_params,
#                  norm=norm, norm_params=norm_params, kernel_size=kernel_size,
#                  last_kernel_size=last_kernel_size, residual_kernel_size=residual_kernel_size, dilation_base=dilation_base, causal=causal,
#                  pad_mode=pad_mode, true_skip=true_skip, compress=compress, lstm=lstm,
#                  disable_norm_outer_blocks=disable_norm_outer_blocks
#         )
        
#         wm_proj0: tp.List[nn.Module] = [
#                 act(**activation_params),
#                 StreamableConv1d(dimension + dimension + dimension//16, dimension, 1, norm='none', norm_kwargs=norm_params, causal=causal, pad_mode=pad_mode),
#             ]
#         self.wm_proj0 = nn.Sequential(*wm_proj0)
        
#         wm_predictor: tp.List[nn.Module] = [
#                 act(**activation_params),
#                 StreamableConv1d(dimension, 2, 1, norm='none', norm_kwargs=norm_params, causal=causal, pad_mode=pad_mode),
#             ]
#         self.wm_predictor = nn.Sequential(*wm_predictor)

#     def forward(self, x, y, z):
#         # decode
#         out = torch.cat([x, z, self.wm_embed(y).transpose(2, 1)], dim=1)
#         out = self.wm_proj0(out)
#         x = self.model(out)

#         m = self.wm_encoder(x)
#         m = self.wm_predictor(m)
        
#         return x, m.transpose(2,1)


class WMSEANetDecoder(nn.Module):
    """Watermark SEANet decoder.

    Args:
        channels (int): Audio channels.
        dimension (int): Intermediate representation dimension.
        n_filters (int): Base width for the model.
        n_residual_layers (int): nb of residual layers.
        ratios (Sequence[int]): kernel size and stride ratios.
        activation (str): Activation function.
        activation_params (dict): Parameters to provide to the activation function.
        final_activation (str): Final activation function after all convolutions.
        final_activation_params (dict): Parameters to provide to the activation function.
        norm (str): Normalization method.
        norm_params (dict): Parameters to provide to the underlying normalization used along with the convolution.
        kernel_size (int): Kernel size for the initial convolution.
        last_kernel_size (int): Kernel size for the initial convolution.
        residual_kernel_size (int): Kernel size for the residual layers.
        dilation_base (int): How much to increase the dilation with each layer.
        causal (bool): Whether to use fully causal convolution.
        pad_mode (str): Padding mode for the convolutions.
        true_skip (bool): Whether to use true skip connection or a simple.
            (streamable) convolution as the skip connection in the residual network blocks.
        compress (int): Reduced dimensionality in residual branches (from Demucs v3).
        lstm (int): Number of LSTM layers at the end of the encoder.
        disable_norm_outer_blocks (int): Number of blocks for which we don't apply norm.
            For the decoder, it corresponds to the N last blocks.
        trim_right_ratio (float): Ratio for trimming at the right of the transposed convolution under the causal setup.
            If equal to 1.0, it means that all the trimming is done at the right.
    """
    def __init__(self, channels: int = 1, dimension: int = 128, n_filters: int = 32, n_residual_layers: int = 3,
                 ratios: tp.List[int] = [8, 5, 4, 2], activation: str = 'ELU', activation_params: dict = {'alpha': 1.0},
                 final_activation: tp.Optional[str] = None, final_activation_params: tp.Optional[dict] = None,
                 norm: str = 'none', norm_params: tp.Dict[str, tp.Any] = {}, kernel_size: int = 7,
                 last_kernel_size: int = 7, residual_kernel_size: int = 3, dilation_base: int = 2, causal: bool = False,
                 pad_mode: str = 'reflect', true_skip: bool = True, compress: int = 2, lstm: int = 0,
                 disable_norm_outer_blocks: int = 0, trim_right_ratio: float = 1.0):
        super().__init__()
        self.dimension = dimension
        self.channels = channels
        self.n_filters = n_filters
        self.ratios = ratios
        del ratios
        self.n_residual_layers = n_residual_layers
        self.hop_length = np.prod(self.ratios)
        self.n_blocks = len(self.ratios) + 2  # first and last conv + residual blocks
        self.disable_norm_outer_blocks = disable_norm_outer_blocks
        assert self.disable_norm_outer_blocks >= 0 and self.disable_norm_outer_blocks <= self.n_blocks, \
            "Number of blocks for which to disable norm is invalid." \
            "It should be lower or equal to the actual number of blocks in the network and greater or equal to 0."

        act = getattr(nn, activation)
        mult = int(2 ** len(self.ratios))
        model: tp.List[nn.Module] = [
            StreamableConv1d(dimension, mult * n_filters, kernel_size,
                             norm='none' if self.disable_norm_outer_blocks == self.n_blocks else norm,
                             norm_kwargs=norm_params, causal=causal, pad_mode=pad_mode)
        ]

        if lstm:
            model += [StreamableLSTM(mult * n_filters, num_layers=lstm)]

        # Upsample to raw audio scale
        for i, ratio in enumerate(self.ratios):
            block_norm = 'none' if self.disable_norm_outer_blocks >= self.n_blocks - (i + 1) else norm
            # Add upsampling layers
            model += [
                act(**activation_params),
                StreamableConvTranspose1d(mult * n_filters, mult * n_filters // 2,
                                          kernel_size=ratio * 2, stride=ratio,
                                          norm=block_norm, norm_kwargs=norm_params,
                                          causal=causal, trim_right_ratio=trim_right_ratio),
            ]
            # Add residual layers
            for j in range(n_residual_layers):
                model += [
                    SEANetResnetBlock(mult * n_filters // 2, kernel_sizes=[residual_kernel_size, 1],
                                      dilations=[dilation_base ** j, 1],
                                      activation=activation, activation_params=activation_params,
                                      norm=block_norm, norm_params=norm_params, causal=causal,
                                      pad_mode=pad_mode, compress=compress, true_skip=true_skip)]

            mult //= 2

        # Add final layers
        model += [
            act(**activation_params),
            StreamableConv1d(n_filters, channels, last_kernel_size,
                             norm='none' if self.disable_norm_outer_blocks >= 1 else norm,
                             norm_kwargs=norm_params, causal=causal, pad_mode=pad_mode)
        ]
        # Add optional final activation to decoder (eg. tanh)
        if final_activation is not None:
            final_act = getattr(nn, final_activation)
            final_activation_params = final_activation_params or {}
            model += [
                final_act(**final_activation_params)
            ]
        self.model = nn.Sequential(*model)

        self.skip_encoder = SEANetEncoder(channels=channels, dimension=dimension, n_filters=n_filters, n_residual_layers=n_residual_layers,
                 ratios=self.ratios, activation=activation, activation_params=activation_params,
                 norm=norm, norm_params=norm_params, kernel_size=kernel_size,
                 last_kernel_size=last_kernel_size, residual_kernel_size=residual_kernel_size, dilation_base=dilation_base, causal=causal,
                 pad_mode=pad_mode, true_skip=true_skip, compress=compress, lstm=lstm,
                 disable_norm_outer_blocks=disable_norm_outer_blocks
        )

        self.wm_embed = nn.Embedding(2, dimension // 16, max_norm=True)

        self.wm_encoder = SEANetEncoder(channels=channels, dimension=dimension, n_filters=n_filters, n_residual_layers=n_residual_layers,
                 ratios=self.ratios, activation=activation, activation_params=activation_params,
                 norm=norm, norm_params=norm_params, kernel_size=kernel_size,
                 last_kernel_size=last_kernel_size, residual_kernel_size=residual_kernel_size, dilation_base=dilation_base, causal=causal,
                 pad_mode=pad_mode, true_skip=true_skip, compress=compress, lstm=lstm,
                 disable_norm_outer_blocks=disable_norm_outer_blocks
        )
        
        wm_proj0: tp.List[nn.Module] = [
                act(**activation_params),
                StreamableConv1d(dimension + dimension//16, dimension, 1, norm='none', norm_kwargs=norm_params, causal=causal, pad_mode=pad_mode),
            ]
        self.wm_proj0 = nn.Sequential(*wm_proj0)
        
        mult = int(2 ** len(self.ratios))
        mult //= 2
        wm_proj1: tp.List[nn.Module] = [
                act(**activation_params),
                StreamableConv1d(mult * n_filters + dimension//16, mult * n_filters, 1, norm='none', norm_kwargs=norm_params, causal=causal, pad_mode=pad_mode),
            ]
        self.wm_proj1 = nn.Sequential(*wm_proj1)
        mult //= 2

        wm_proj2: tp.List[nn.Module] = [
                act(**activation_params),
                StreamableConv1d(mult * n_filters + dimension//16, mult * n_filters, 1, norm='none', norm_kwargs=norm_params, causal=causal, pad_mode=pad_mode),
            ]
        self.wm_proj2 = nn.Sequential(*wm_proj2)
        mult //= 2
        
        wm_proj3: tp.List[nn.Module] = [
                act(**activation_params),
                StreamableConv1d(mult * n_filters + dimension//16, mult * n_filters, 1, norm='none', norm_kwargs=norm_params, causal=causal, pad_mode=pad_mode),
            ]
        self.wm_proj3 = nn.Sequential(*wm_proj3)
        # mult //= 2
        
        # wm_proj4: tp.List[nn.Module] = [
        #         act(**activation_params),
        #         StreamableConv1d(mult * n_filters + dimension//16, mult * n_filters, 1, norm='none', norm_kwargs=norm_params, causal=causal, pad_mode=pad_mode),
        #     ]
        # self.wm_proj4 = nn.Sequential(*wm_proj4)


        wm_predictor: tp.List[nn.Module] = [
                act(**activation_params),
                StreamableConv1d(dimension, 2, 1, norm='none', norm_kwargs=norm_params, causal=causal, pad_mode=pad_mode),
            ]
        self.wm_predictor = nn.Sequential(*wm_predictor)

    def forward(self, x, y, z):
        # x: latents, y: labels, z: waveform
        # get skip features and labels
        skips = []
        labels = []
        z = self.skip_encoder.model[0:2](z)
        # skips.append(z)
        # labels.append(torch.repeat_interleave(y, repeats=self.ratios[0]*self.ratios[1]*self.ratios[2]*self.ratios[3], dim=-1).to(y.device))
        z = self.skip_encoder.model[2:5](z)
        skips.append(z)
        labels.append(torch.repeat_interleave(y, repeats=self.ratios[0]*self.ratios[1]*self.ratios[2], dim=-1).to(y.device))
        z = self.skip_encoder.model[5:8](z)
        skips.append(z)
        labels.append(torch.repeat_interleave(y, repeats=self.ratios[0]*self.ratios[1], dim=-1).to(y.device))
        z = self.skip_encoder.model[8:11](z)
        skips.append(z)
        labels.append(torch.repeat_interleave(y, repeats=self.ratios[0], dim=-1).to(y.device))
        z = self.skip_encoder.model[11:](z)
        skips.append(z)
        labels.append(y)
        
        # decode
        out = torch.cat([skips.pop(), self.wm_embed(labels.pop()).transpose(2, 1)], dim=1)
        out = self.wm_proj0(out) + x
        x = self.model[:4](out)
        
        out = torch.cat([skips.pop(), self.wm_embed(labels.pop()).transpose(2, 1)], dim=1)
        out = self.wm_proj1(out) + x
        x = self.model[4:7](out)

        out = torch.cat([skips.pop(), self.wm_embed(labels.pop()).transpose(2, 1)], dim=1)
        out = self.wm_proj2(out) + x
        x = self.model[7:10](out)

        out = torch.cat([skips.pop(), self.wm_embed(labels.pop()).transpose(2, 1)], dim=1)
        out = self.wm_proj3(out) + x
        x = self.model[10:](out)

        # out = torch.cat([skips.pop(), self.wm_embed(labels.pop()).transpose(2, 1)], dim=1)
        # out = self.wm_proj4(out) + x
        # x = self.model[13:](out)

        m = self.wm_encoder(x)
        m = self.wm_predictor(m)
        
        return x, m.transpose(2,1)
