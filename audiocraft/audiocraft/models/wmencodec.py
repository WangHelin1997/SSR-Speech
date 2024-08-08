# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Compression models or wrapper around existing models.
Also defines the main interface that a model must follow to be usable as an audio tokenizer.
"""

from abc import ABC, abstractmethod
import logging
import math
from pathlib import Path
import typing as tp

import numpy as np
import torch
from torch import nn
import random
from .. import quantization as qt


logger = logging.getLogger()


class WMCompressionModel(ABC, nn.Module):
    """Base API for all compression model that aim at being used as audio tokenizers
    with a language model.
    """

    @abstractmethod
    def forward(self, x: torch.Tensor) -> qt.QuantizedResult:
        ...

    @abstractmethod
    def encode(self, x: torch.Tensor) -> tp.Tuple[torch.Tensor, tp.Optional[torch.Tensor]]:
        """See `EncodecModel.encode`."""
        ...

    @abstractmethod
    def decode(self, codes: torch.Tensor, scale: tp.Optional[torch.Tensor] = None):
        """See `EncodecModel.decode`."""
        ...

    @abstractmethod
    def decode_latent(self, codes: torch.Tensor):
        """Decode from the discrete codes to continuous latent space."""
        ...

    @property
    @abstractmethod
    def channels(self) -> int:
        ...

    @property
    @abstractmethod
    def frame_rate(self) -> float:
        ...

    @property
    @abstractmethod
    def sample_rate(self) -> int:
        ...

    @property
    @abstractmethod
    def cardinality(self) -> int:
        ...

    @property
    @abstractmethod
    def num_codebooks(self) -> int:
        ...

    @property
    @abstractmethod
    def total_codebooks(self) -> int:
        ...

    @abstractmethod
    def set_num_codebooks(self, n: int):
        """Set the active number of codebooks used by the quantizer."""
        ...

    @staticmethod
    def get_pretrained(
            name: str, device: tp.Union[torch.device, str] = 'cpu'
            ) -> 'CompressionModel':
        """Instantiate a CompressionModel from a given pretrained model.

        Args:
            name (Path or str): name of the pretrained model. See after.
            device (torch.device or str): Device on which the model is loaded.

        Pretrained models:
            - dac_44khz (https://github.com/descriptinc/descript-audio-codec)
            - dac_24khz (same)
            - facebook/encodec_24khz (https://huggingface.co/facebook/encodec_24khz)
            - facebook/encodec_32khz (https://huggingface.co/facebook/encodec_32khz)
            - your own model on HugginFace. Export instructions to come...
        """

        from . import builders, loaders
        model: WMCompressionModel
        if name in ['dac_44khz', 'dac_24khz']:
            model_type = name.split('_')[1]
            logger.info("Getting pretrained compression model from DAC %s", model_type)
            model = DAC(model_type)
        elif name in ['debug_compression_model']:
            logger.info("Getting pretrained compression model for debug")
            model = builders.get_debug_compression_model()
        elif Path(name).exists():
            # We assume here if the paths exist that it is in fact an AC checkpoint
            # that was exported using `audiocraft.utils.export` functions.
            model = loaders.load_wmcompression_model(name, device=device)
        else:
            logger.info("Getting pretrained compression model from HF %s", name)
            hf_model = HFEncodecModel.from_pretrained(name)
            model = HFEncodecCompressionModel(hf_model).to(device)
        return model.to(device).eval()


class WMEncodecModel(WMCompressionModel):
    """Encodec model operating on the raw waveform.

    Args:
        encoder (nn.Module): Encoder network.
        decoder (nn.Module): Decoder network.
        quantizer (qt.BaseQuantizer): Quantizer network.
        frame_rate (int): Frame rate for the latent representation.
        sample_rate (int): Audio sample rate.
        channels (int): Number of audio channels.
        causal (bool): Whether to use a causal version of the model.
        renormalize (bool): Whether to renormalize the audio before running the model.
    """
    # we need assignment to override the property in the abstract class,
    # I couldn't find a better way...
    frame_rate: float = 0
    sample_rate: int = 0
    channels: int = 0

    def __init__(self,
                 encoder: nn.Module,
                 decoder: nn.Module,
                 wmdecoder: nn.Module,
                 quantizer: qt.BaseQuantizer,
                 frame_rate: int,
                 sample_rate: int,
                 channels: int,
                 causal: bool = False,
                 renormalize: bool = False):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.wmdecoder = wmdecoder
        self.quantizer = quantizer
        self.frame_rate = frame_rate
        self.sample_rate = sample_rate
        self.channels = channels
        self.renormalize = renormalize
        self.causal = causal
        if self.causal:
            # we force disabling here to avoid handling linear overlap of segments
            # as supported in original EnCodec codebase.
            assert not self.renormalize, 'Causal model does not support renormalize'

    @property
    def total_codebooks(self):
        """Total number of quantizer codebooks available."""
        return self.quantizer.total_codebooks

    @property
    def num_codebooks(self):
        """Active number of codebooks used by the quantizer."""
        return self.quantizer.num_codebooks

    def set_num_codebooks(self, n: int):
        """Set the active number of codebooks used by the quantizer."""
        self.quantizer.set_num_codebooks(n)

    @property
    def cardinality(self):
        """Cardinality of each codebook."""
        return self.quantizer.bins

    def preprocess(self, x: torch.Tensor) -> tp.Tuple[torch.Tensor, tp.Optional[torch.Tensor]]:
        scale: tp.Optional[torch.Tensor]
        if self.renormalize:
            mono = x.mean(dim=1, keepdim=True)
            volume = mono.pow(2).mean(dim=2, keepdim=True).sqrt()
            scale = 1e-8 + volume
            x = x / scale
            scale = scale.view(-1, 1)
        else:
            scale = None
        return x, scale

    def postprocess(self,
                    x: torch.Tensor,
                    scale: tp.Optional[torch.Tensor] = None) -> torch.Tensor:
        if scale is not None:
            assert self.renormalize
            x = x * scale.view(-1, 1, 1)
        return x

    def apply_random_masking(self, 
                             batch_size, 
                             N, 
                             min_mask_regions=0, 
                             max_mask_regions=2, 
                             max_mask_fraction=0.7,
                            ):
        mask_tensor = torch.zeros(batch_size, N, dtype=torch.long)
        
        for i in range(batch_size):
            num_regions = random.randint(min_mask_regions, max_mask_regions)
            total_masked = 0
            
            for _ in range(num_regions):
                if total_masked >= int(max_mask_fraction * N):
                    break
                
                # Randomly choose the length of the mask
                mask_len = random.randint(1, int(N * max_mask_fraction))
                
                # Make sure we do not exceed the max_mask_fraction
                if total_masked + mask_len > max_mask_fraction * N:
                    mask_len = int(max_mask_fraction * N) - total_masked

                start_idx = random.randint(0, N - mask_len)
                mask_tensor[i, start_idx:start_idx + mask_len] = 1
                total_masked += mask_len
                
        return mask_tensor

    def forward(self, x: torch.Tensor) -> qt.QuantizedResult:
        assert x.dim() == 3
        length = x.shape[-1]
        x, scale = self.preprocess(x)
        y = x.clone()
        emb = self.encoder(x)
        mark_label = self.apply_random_masking(emb.shape[0], emb.shape[-1]).to(emb.device)
        q_res = self.quantizer(emb, self.frame_rate)
        
        out, mark = self.wmdecoder(q_res.x, mark_label.clone(), y)

        # remove extra padding added by the encoder and decoder
        assert out.shape[-1] >= length, (out.shape[-1], length)
        out = out[..., :length]

        q_res.x = self.postprocess(out, scale)

        return q_res, mark, mark_label

    def encode(self, x: torch.Tensor) -> tp.Tuple[torch.Tensor, tp.Optional[torch.Tensor]]:
        """Encode the given input tensor to quantized representation along with scale parameter.

        Args:
            x (torch.Tensor): Float tensor of shape [B, C, T]

        Returns:
            codes, scale (tuple of torch.Tensor, torch.Tensor): Tuple composed of:
                codes a float tensor of shape [B, K, T] with K the number of codebooks used and T the timestep.
                scale a float tensor containing the scale for audio renormalizealization.
        """
        assert x.dim() == 3
        x, scale = self.preprocess(x)
        emb = self.encoder(x)
        codes = self.quantizer.encode(emb)
        return codes, scale

    def decode(self, codes: torch.Tensor, scale: tp.Optional[torch.Tensor] = None):
        """Decode the given codes to a reconstructed representation, using the scale to perform
        audio denormalization if needed.

        Args:
            codes (torch.Tensor): Int tensor of shape [B, K, T]
            scale (torch.Tensor, optional): Float tensor containing the scale value.

        Returns:
            out (torch.Tensor): Float tensor of shape [B, C, T], the reconstructed audio.
        """
        emb = self.decode_latent(codes)
        out = self.decoder(emb)
        out = self.postprocess(out, scale)
        # out contains extra padding added by the encoder and decoder
        return out
    
    def wmdecode(self, codes: torch.Tensor, labels: torch.Tensor, wavform: torch.Tensor, scale: tp.Optional[torch.Tensor] = None):
        """Decode the given codes to a reconstructed representation, using the scale to perform
        audio denormalization if needed.

        Args:
            codes (torch.Tensor): Int tensor of shape [B, K, T]
            labels (torch.Tensor): Int tensor of shape [B, T]
            wavform (torch.Tensor): Float tensor of shape [B, C, T']
            scale (torch.Tensor, optional): Float tensor containing the scale value.

        Returns:
            out (torch.Tensor): Float tensor of shape [B, C, T], the reconstructed audio.
        """
        emb = self.decode_latent(codes)
        out, mark = self.wmdecoder(emb, labels, wavform)
        out = self.postprocess(out, scale)
        # out contains extra padding added by the encoder and decoder
        return out, mark

    def detect_watermark(self, x: torch.Tensor):
        assert x.dim() == 3
        m = self.wmdecoder.wm_encoder(x)
        m = self.wmdecoder.wm_predictor(m).squeeze(-1)
        m = torch.argmax(m, dim=-1)
        return m

    def decode_latent(self, codes: torch.Tensor):
        """Decode from the discrete codes to continuous latent space."""
        return self.quantizer.decode(codes)
