# @ hwang258@jh.edu

import random

import numpy as np
import logging
import argparse, copy
from typing import Dict, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.classification import MulticlassAccuracy

from .modules.utils import make_pad_mask

from .modules.embedding import SinePositionalEmbedding, TokenEmbedding
from .modules.transformer import (
    LayerNorm,
    TransformerEncoder,
    TransformerEncoderLayer,
)
from huggingface_hub import PyTorchModelHubMixin
from argparse import Namespace
import typing as tp

def top_k_top_p_filtering(
    logits, top_k=0, top_p=1.0, filter_value=-float("Inf"), min_tokens_to_keep=1
):
    """Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
    Args:
        logits: logits distribution shape (batch size, vocabulary size)
        if top_k > 0: keep only top k tokens with highest probability (top-k filtering).
        if top_p < 1.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
            Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        Make sure we keep at least min_tokens_to_keep per batch example in the output
    From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    if top_k > 0:
        top_k = min(
            max(top_k, min_tokens_to_keep), logits.size(-1)
        )  # Safety check
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(
            F.softmax(sorted_logits, dim=-1), dim=-1
        )

        # Remove tokens with cumulative probability above the threshold (token with 0 are kept)
        sorted_indices_to_remove = cumulative_probs > top_p
        if min_tokens_to_keep > 1:
            # Keep at least min_tokens_to_keep (set to min_tokens_to_keep-1 because we add the first one below)
            sorted_indices_to_remove[..., :min_tokens_to_keep] = 0
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
            ..., :-1
        ].clone()
        sorted_indices_to_remove[..., 0] = 0

        # scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(
            1, sorted_indices, sorted_indices_to_remove
        )
        logits[indices_to_remove] = filter_value
    return logits


def topk_sampling(logits, top_k=10, top_p=1.0, temperature=1.0):
    # temperature: (`optional`) float
    #     The value used to module the next token probabilities. Must be strictly positive. Default to 1.0.
    # top_k: (`optional`) int
    #     The number of highest probability vocabulary tokens to keep for top-k-filtering. Between 1 and infinity. Default to 50.
    # top_p: (`optional`) float
    #     The cumulative probability of parameter highest probability vocabulary tokens to keep for nucleus sampling. Must be between 0 and 1. Default to 1.

    # Temperature (higher temperature => more likely to sample low probability tokens)
    if temperature != 1.0:
        logits = logits / temperature
    # Top-p/top-k filtering
    logits = top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)
    # Sample
    token = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
    return token

class SSR_Speech(
        nn.Module,
        PyTorchModelHubMixin,
        library_name="ssr_speech",
        repo_url=None,
        tags=None,
    ):
    def __new__(cls, args: Optional[Namespace] = None, config: Optional[Dict] = None, **kwargs) -> "SSR_Speech":
        # If initialized from Namespace args => convert to dict config for 'PyTorchModelHubMixin' to serialize it as config.json
        # Won't affect instance initialization
        if args is not None:
            if config is not None:
                raise ValueError("Cannot provide both `args` and `config`.")
            config = vars(args)
        return super().__new__(cls, args=args, config=config, **kwargs)

    def __init__(self, args: Optional[Namespace] = None, config: Optional[Dict] = None):
        super().__init__()

        # If loaded from HF Hub => convert config.json to Namespace args before initializing
        if args is None:
            if config is None:
                raise ValueError("Either `args` or `config` must be provided.")
            args = Namespace(**config)

        self.args = copy.copy(args)
        if not getattr(self.args, "n_special", False):
            self.args.n_special = 3
        self.args.eos = getattr(self.args, "eos", -1)

        if isinstance(self.args.audio_vocab_size, str):
            self.args.audio_vocab_size = eval(self.args.audio_vocab_size)

        self.n_text_tokens = self.args.text_vocab_size + 1
        assert self.args.text_pad_token == self.args.text_vocab_size, f"self.args.text_vocab_size: {self.args.text_vocab_size}, self.args.text_pad_token: {self.args.text_pad_token}"

        self.n_audio_tokens = [int(self.args.audio_vocab_size) + self.args.n_special + self.args.max_n_spans] * self.args.n_codebooks # special tokens: empty token, EOG token, audio pad token, mask tokens
        assert self.args.audio_vocab_size == self.args.empty_token, self.args.empty_token
        assert self.args.eog == self.args.audio_vocab_size + 1, self.args.eog
        assert self.args.audio_pad_token == self.args.audio_vocab_size + 2, self.args.audio_pad_token
        assert self.args.eos == self.args.audio_vocab_size + 3, self.args.eos
        assert self.args.sos == self.args.audio_vocab_size + 4, self.args.sos
        assert self.args.mts == self.args.audio_vocab_size + 5, self.args.mts

        self.text_embedding = TokenEmbedding(
            dim_model=self.args.d_model,
            vocab_size=self.n_text_tokens, 
            dropout=self.args.text_embedding_dropout
        )

        self.audio_embedding = nn.ModuleList(
            [
                TokenEmbedding(
                dim_model=self.args.audio_embedding_dim, 
                vocab_size=self.n_audio_tokens[k], 
                dropout=self.args.audio_embedding_dropout
            ) for k in range(self.args.n_codebooks)
            ]
        )
        self.text_positional_embedding = SinePositionalEmbedding(
            self.args.d_model,
            dropout=self.args.text_positional_embedding_dropout,
            scale=False,
            alpha=True, # learnable scaler, scale the volume of positional embedding
        )
        self.audio_positional_embedding = SinePositionalEmbedding(
            self.args.d_model,
            dropout=self.args.audio_positional_embedding_dropout,
            scale=False,
            alpha=True, # learnable scaler, scale the volume of positional embedding
        )

        dec_layer = TransformerEncoderLayer(
            self.args.d_model,
            self.args.nhead,
            dim_feedforward=self.args.d_model * 4,
            dropout=self.args.trm_dropout,
            batch_first=True,
            norm_first=True,
            layer_norm_cls=LayerNorm
        )
        self.decoder = TransformerEncoder(
            dec_layer,
            num_layers=self.args.num_decoder_layers,
            norm=LayerNorm(self.args.d_model),
        )
        
        self.predict_layer = nn.ModuleList(
            [
                nn.Sequential(nn.Linear(self.args.d_model, self.args.audio_vocab_size//2), nn.GELU(), nn.Linear(self.args.audio_vocab_size//2, self.n_audio_tokens[k])) for k in range(self.args.n_codebooks)
            ]
        )
        
        self.accuracy_metrics = nn.ModuleList(
            [MulticlassAccuracy(
                self.n_audio_tokens[k],
                top_k=10,
                average="micro",
                multidim_average="global",
                ignore_index=None,
            ) for k in range(self.args.n_codebooks)]
        )

    def embed_y(self, cated_y):
        # [K,T,B]
        embedded_y = torch.stack([self.audio_embedding[k](cated_y[k]) for k in range(self.args.n_codebooks)], dim=0) # [K, T, B, D]
        assert embedded_y.shape[0] == self.args.n_codebooks, embedded_y.shape
        assert embedded_y.shape[-1] == self.args.d_model, embedded_y.shape
        embedded_y = embedded_y.sum(dim=0) # [K,T,B,D]->[T,B,D]
        embedded_y = embedded_y.transpose(1,0) # [T,B,D]->[B,T,D]
        return embedded_y

        
    def prepare_input_target(self, cated_y, y_lens):
        
        embedded_y = self.embed_y(cated_y) # [B,T,D]
        
        # positional embedding
        y_input = self.audio_positional_embedding(embedded_y)
        
        # make attention mask and padding mask
        y_padding_mask = make_pad_mask(y_lens).to(cated_y.device)
        y_attention_mask = torch.triu(torch.ones(y_input.shape[1], y_input.shape[1]), diagonal=1).bool().to(y_padding_mask.device)
        return y_input, y_padding_mask, y_attention_mask


    def dec_forward(
            self, 
            x_input, 
            x_lens,
            x_attention_mask,
            x_padding_mask,
            y_input,
            new_y_lens,
            y_attention_mask,
            y_padding_mask,
            past=None,
            last_3_tokens=False
        ):
            x_attn_mask = F.pad(
                x_attention_mask,
                (0, new_y_lens.max()),
                value=True,
            ) # x attn to all x, doesn't attn to any y, this follow figure 3 of the valle paper
            y_attn_mask = F.pad(
                y_attention_mask,
                (x_lens.max(), 0), # y is padded at the front
                value=False,
            ) # y attn to all x, for y itself use lower triangle mask to ensure autoregressive
            xy_attn_mask = torch.concat([x_attn_mask, y_attn_mask], dim=0)

            # merge key padding and attention masks
            bsz, src_len = x_input.shape[0], x_lens.max() + new_y_lens.max()
            xy_padding_mask = torch.concat([x_padding_mask, y_padding_mask], dim=1)
            _xy_padding_mask = (
                xy_padding_mask.view(bsz, 1, 1, src_len)
                .expand(-1, self.args.nhead, -1, -1)
                .reshape(bsz * self.args.nhead, 1, src_len)
            )
            # Check shapes and resize+broadcast as necessary
            if xy_attn_mask.shape != _xy_padding_mask.shape:
                assert xy_attn_mask.ndim + 1 == _xy_padding_mask.ndim, f"xy_attn_mask.shape: {xy_attn_mask.shape}, _xy_padding_mask: {_xy_padding_mask.shape}"
                xy_attn_mask = xy_attn_mask.unsqueeze(0).repeat(_xy_padding_mask.shape[0], 1, 1)  # Example approach
            xy_attn_mask = xy_attn_mask.logical_or(_xy_padding_mask)

            new_attn_mask = torch.zeros_like(xy_attn_mask)
            new_attn_mask.masked_fill_(xy_attn_mask, float("-inf"))
            xy_attn_mask = new_attn_mask

            xy_input = torch.cat([x_input, y_input], dim=1)

            if past == None: # do not use kvcache
                out, _ =  self.decoder((xy_input, None), mask=xy_attn_mask)
                return out[:, x_lens.max():], None
            else: # use kvcache
                if past.ndim > 3: # uses kvcache, only need to pass the last tokens, this doesn't work with multi-span speech editing yet
                    if last_3_tokens:
                        xy_input = xy_input[:, -3:]
                        xy_attn_mask = xy_attn_mask[:, -3:]
                    else:
                        xy_input = xy_input[:, -1:]
                        xy_attn_mask = xy_attn_mask[:, -1:]

                out, present = self.decoder((xy_input, None), mask=xy_attn_mask, past=past)
                if isinstance(out, tuple): # get rid of stage_embedding
                    out = out[0]

                if out.shape[1] > x_lens.max(): # the first pass, not kvcache yet
                    return out[:, x_lens.max():], present
                else: # used kvcache
                    return out, present

    def forward(self, batch):
        """
        Args:
          x:
            A 2-D tensor of shape (N, S).
          x_lens:
            A 1-D tensor of shape (N,). It contains the number of tokens in `x`
            before padding.
          y:
            A 3-D tensor of shape (N, K, T).
            where K is the number of codebooks
          y_lens:
            A 1-D tensor of shape (N,). It contains the number of tokens in `x`
            before padding.
        """
        x, x_lens, y, y_lens = batch["x"], batch["x_lens"], batch["y"], batch["y_lens"]
        if len(x) == 0:
            return None
        x = x[:, :x_lens.max()] # this deal with gradient accumulation, where x_lens.max() might not be longer than the length of the current slice of x
        y = y[:, :, :y_lens.max()]
        assert x.ndim == 2, x.shape
        assert x_lens.ndim == 1, x_lens.shape
        assert y.ndim == 3 and y.shape[1] == self.args.n_codebooks, y.shape
        assert y_lens.ndim == 1, y_lens.shape
        
        targets = y.clone()
        y = y.permute(1,2,0) # [B,K,T]->[K,T,B]
        # makes attention mask and padding mask for x
        x_padding_mask = make_pad_mask(x_lens).to(x.device)
        x_attention_mask = torch.triu(torch.ones(x.shape[1], x.shape[1]), diagonal=1).bool().to(x_padding_mask.device)
        x_input = self.text_embedding(x)
        x_input = self.text_positional_embedding(x_input)
        y_input, y_padding_mask, y_attention_mask = self.prepare_input_target(y, y_lens)
        y_out = self.dec_forward(
                    x_input, 
                    x_lens,
                    x_attention_mask,
                    x_padding_mask,
                    y_input,
                    y_lens,
                    y_attention_mask,
                    y_padding_mask
                )
        y_out = y_out[0] # no kv-caching during training
        assert y_out.shape == y_input.shape, f"y_out.shape: {y_out.shape}, y_input.shape: {y_input.shape}" # [B S D]
        
        logits = torch.stack([self.predict_layer[i](y_out) for i in range(self.args.n_codebooks)], dim=1) # [B K S card]
        assert logits.shape[1] == self.args.n_codebooks and logits.shape[3] == self.n_audio_tokens[0], logits.shape

        targets = targets.permute(1,0,2) # [K B T]
        logits = logits.permute(1,0,2,3) # [K B S card]

        logits = logits[:, :, :-1]
        targets = targets[:, :, 1:]
        
        if self.args.predict_mask_token:
            masks = (targets != self.args.audio_pad_token) & (targets != self.args.empty_token)
        else:
            masks = (targets != self.args.audio_pad_token) & (targets != self.args.empty_token) & (targets < self.args.mts)
            
        tmp_masks = masks.clone()
        
        if not self.args.predict_all:
            eos_pos = (targets == self.args.mts).nonzero(as_tuple=False)
            for k, b, t in eos_pos:
                tmp_masks[k, b, :t] = False
            
            
        assert masks.shape[0] == self.args.n_codebooks, masks.shape
        
        loss = []
        ntokens = []
        top10acc = []

        for k, (logit, target, mask, tmp_mask) in enumerate(zip(logits, targets, masks, tmp_masks)):
            logit = logit.reshape(-1, logit.size(-1)) # B*S card
            target = target.reshape(-1) # B*T
            mask = mask.reshape(-1).bool()
            tmp_mask = tmp_mask.reshape(-1).bool()

            loss.append(F.cross_entropy(logit[tmp_mask], target[tmp_mask], reduction='mean'))
            top10acc.append(self.accuracy_metrics[k](logit[tmp_mask].detach(), target[tmp_mask]))
            ntokens.append(len(target[mask]))
        
        all_ntokens = sum(ntokens)
        if self.args.codebook_weight != None:
            codebook_weight = eval(self.args.codebook_weight)
        else:
            codebook_weight = [1.] * self.args.n_codebooks
        loss = sum([l*nt*cw for l, nt, cw in zip(loss, ntokens, codebook_weight)])
        top10acc_by_codebook = [t10a*nt for t10a, nt in zip(top10acc, ntokens)]
        top10acc = sum(top10acc_by_codebook)
        ntokens = torch.tensor(all_ntokens).to(logits.device)

        return {
            "loss": loss,
            "top10acc": top10acc,
            "top10acc_by_codebook": top10acc_by_codebook,
            "effective_ntoken": ntokens,
        }

    def rearrange(self, y, non_mask_intervals, mask_intervals):
    
        assert self.args.eos > 0, f"eos={self.args.eos} should > 0"
        
        rearranged_y = []
        sos_tensor = torch.LongTensor([self.args.sos] * self.args.n_codebooks).unsqueeze(-1).to(y.device)
        eos_tensor = torch.LongTensor([self.args.eos] * self.args.n_codebooks).unsqueeze(-1).to(y.device)
        eog_tensor = torch.LongTensor([self.args.eog] * self.args.n_codebooks).unsqueeze(-1).to(y.device)
        for i, item in enumerate(non_mask_intervals):
            if i == 0:
                if item[0] == item[1]: # case: (0,0)
                    rearranged_y.append(sos_tensor)
                else:
                    rearranged_y.append(torch.cat([sos_tensor, y[:, item[0]: item[1]]], dim=-1))
            elif i == len(non_mask_intervals)-1:
                if item[0] == item[1]: # case: (N,N)
                    rearranged_y.append(eos_tensor)
                else:
                    rearranged_y.append(torch.cat([y[:, item[0]: item[1]], eos_tensor], dim=-1))
            else:
                rearranged_y.append(y[:, item[0]: item[1]])
                
        for i, item in enumerate(mask_intervals):
            rearranged_y.append(torch.cat([y[:, item[0]: item[1]], eog_tensor], dim=-1))

        return rearranged_y

    def get_pattern_sequence(self, tokens: torch.Tensor, n_q: int, special_token: int, delays: tp.Optional[tp.List[int]] = None, 
                         empty_initial: int = 0) -> torch.Tensor:
        """Generate a pattern sequence for delayed codebooks without batch dimension.
    
        Args:
            tokens (torch.Tensor): Input tensor of shape [K, T].
            n_q (int): Number of codebooks.
            delays (Optional[List[int]]): Delay for each codebook. Defaults to increasing delays.
            empty_initial (int): Number of initial empty steps. Defaults to 0.
            special_token (int): Special token used to fill non-pattern coordinates in the new sequence.
    
        Returns:
            torch.Tensor: Modified tokens based on the pattern.
        """
        K, T = tokens.shape
        assert K == n_q, "Number of codebooks (K) must match n_q"
        if delays is None:
            delays = list(range(n_q))
        max_delay = max(delays)
        pattern_length = T + max_delay + empty_initial
        pattern_tokens = torch.full((K, pattern_length), fill_value=special_token, dtype=tokens.dtype).to(tokens.device)
    
        for t in range(T):
            for q in range(n_q):
                delayed_t = t + delays[q] + empty_initial
                if delayed_t < pattern_length:
                    pattern_tokens[q, delayed_t] = tokens[q, t]
    
        return pattern_tokens

    def revert_pattern_sequence(self, pattern_tokens: torch.Tensor, n_q: int,
                            delays: tp.Optional[tp.List[int]] = None, special_token: int = -1) -> torch.Tensor:
        """Revert the pattern sequence back to the original multi-codebook sequence without batch dimension.
    
        Args:
            pattern_tokens (torch.Tensor): Pattern tensor of shape [K, S].
            n_q (int): Number of codebooks.
            delays (Optional[List[int]]): Delay for each codebook. Defaults to increasing delays.
            special_token (int): Special token used to fill non-pattern coordinates in the new sequence.
    
        Returns:
            torch.Tensor: Reverted tokens of shape [K, T].
        """
        K, S = pattern_tokens.shape
        assert K == n_q, "Number of codebooks (K) must match n_q"
        if delays is None:
            delays = list(range(n_q))
        T = S - max(delays)
        reverted_tokens = torch.full((K, T), fill_value=special_token, dtype=pattern_tokens.dtype).to(pattern_tokens.device)
    
        for t in range(T):
            for q in range(n_q):
                delayed_t = t + delays[q]
                if delayed_t < S:
                    reverted_tokens[q, t] = pattern_tokens[q, delayed_t]
    
        return reverted_tokens
        
    def shift(self, rearranged_y):
        
        shifted_y = [self.get_pattern_sequence(tokens=cur_y, n_q=self.args.n_codebooks, special_token=self.args.empty_token) for cur_y in rearranged_y] # the first item is values, later two are indexes and mask
        
        return shifted_y
    
    def insert_mask(self, shifted_y):

        num_masks = (len(shifted_y) - 1) // 2
        assert num_masks == (len(shifted_y) - 1) / 2, len(shifted_y)
        emb_inds = list(range(self.args.mts, self.args.mts+ self.args.max_n_spans))
        if self.args.shuffle_mask_embedding:
            random.shuffle(emb_inds)
        emb_inds_use = emb_inds[:num_masks]
        mask_value = emb_inds_use + emb_inds_use
        assert len(shifted_y) == len(mask_value) + 1, len(mask_value)
        
        inserted_y = []
        mask_position = [-1] * (self.args.max_n_spans*2)
        for j in range(len(shifted_y)-1):
            inserted_y.append(shifted_y[j])
            mask_position[j] = sum([item.shape[1] for item in inserted_y]) # each item is of shape [K S], so take shape[1]
            tmp = torch.LongTensor([mask_value[j]] * self.args.n_codebooks).unsqueeze(-1).to(shifted_y[0].device)
            inserted_y.append(tmp)

        inserted_y.append(shifted_y[-1])
        mask_position = [item for item in mask_position if item != -1]
        
        return inserted_y, mask_position
    
    def cat_y(self, inserted_y):

        cated_y = torch.cat(inserted_y, dim=1)
        assert cated_y.shape[0] == self.args.n_codebooks, cated_y.shape
        new_y_lens = cated_y.shape[1]
        
        return cated_y, new_y_lens
    
    def inference(
        self,
        x: torch.Tensor,
        x_lens: torch.Tensor,
        prompt_x: torch.Tensor,
        prompt_x_lens: torch.Tensor,
        y: torch.Tensor,
        prompt: torch.Tensor,
        mask_interval: list[torch.Tensor],
        top_k: int=-100,
        top_p: float=1.0,
        temperature: float=1.0,
        stop_repetition: int=-1,
        kvcache: int=1,
        silence_tokens: list[int]=[1388,1898,131],
        cfg_coef: float=1.5,
        cfg_stride: int=1,
        aug_text: bool=False,
        aug_context: bool=False,
        cfg_pretrained: bool=False,
    ) -> torch.Tensor:
        """
        Args:
          x:
            A 2-D tensor of shape (1, L).
          x_lens:
            A 1-D tensor of shape (1,). It contains the number of tokens in `x`
            before padding.
          y:
            A 3-D tensor of shape (1, T, K).
          mask_interval:
            a list of tensors of shape (M, 2). contains M mask_start and mask_end. list length is actually 1, because we only support single sample inference for now
          top_k: (`optional`) int
            The number of highest probability tokens to keep for top-k-filtering. Default to -100.
          top_p: (`optional`) float
            For Neucleus sampling
          temperature: (`optional`) float
            The value used to module the next token probabilities. Must be strictly positive. Default to 1.0.
          stop_repetition (`optional`) int
            if not -1, will set the logits of a token that repeated this many times to be -100000, to avoid generating it again. This only apply to tokens from the first codebook
          kvcache (`optional`) int
            if 1, use kvcache to speed up sampling
          cfg_coef: float (>= 1.0)
          aug_text: whether use cfg to improve the text input
          aug_context: whether improve the context by combining original audio and text
          cfg_pretrained: whether use cfg in training
        """
        
        assert cfg_coef >= 1.0, cfg_coef
        assert x.ndim == 2, x.shape
        assert x_lens.ndim == 1, x_lens.shape
        assert y.ndim == 3, y.shape
        y = y.transpose(2,1) # [1,T,K] -> [1,K,T]
        assert prompt.ndim == 3, prompt.shape
        prompt = prompt.transpose(2,1)
        assert y.shape[0] == 1 and y.shape[1] == self.args.n_codebooks, y.shape # there is no padding
        assert prompt.shape[0] == 1 and prompt.shape[1] == self.args.n_codebooks, prompt.shape # there is no padding
        assert mask_interval.shape == torch.Size((1, mask_interval.shape[1], 2)), mask_interval

        # whether to use context
        context_len = sum([item[1] - item[0] for item in mask_interval[0]])
        if aug_context and context_len < 2 * 50:
            aug_context = True
        else:
            aug_context = False
        
        # augment
        if aug_text and not aug_context: # [t, ab, m] [t', ab, m]
            y = y.repeat(2, 1, 1)
            if not cfg_pretrained:
                uncond_x = torch.randint(0, self.n_text_tokens, (1, x.shape[1])).to(x.device)
            else:
                uncond_x = torch.tensor([self.args.text_vocab_size-1], dtype=torch.long).unsqueeze(0).repeat(1, x.shape[1]).to(x.device)
            x = torch.cat([x, uncond_x], dim=0)

        if aug_text and aug_context: # [tc, t, c, ab, m] [tc, t', c, ab, m]
            out_len = prompt.shape[2]
            gt_y = torch.cat([prompt, y], dim=-1)
            y = gt_y.repeat(2, 1, 1)
            gt_x = torch.cat([prompt_x, x], dim=1)
            if not cfg_pretrained:
                uncond_x = torch.randint(0, self.n_text_tokens, (1, gt_x.shape[1])).to(gt_x.device)
            else:
                uncond_x = torch.tensor([self.args.text_vocab_size-1], dtype=torch.long).unsqueeze(0).repeat(1, gt_x.shape[1]).to(gt_x.device)
            x = torch.cat([gt_x, uncond_x], dim=0)
            
        if not aug_text and aug_context: # [tc, t, c, ab, m]
            out_len = prompt.shape[2]
            y = torch.cat([prompt, y], dim=-1)
            x = torch.cat([prompt_x, x], dim=1)


        # make x attention mask and x_input
        x_lens = torch.LongTensor([x.shape[-1]]).to(x_lens.device)
        x_attention_mask = torch.triu(torch.ones(x.shape[1], x.shape[1]), diagonal=1).bool().to(x.device)
        x_input = self.text_embedding(x)
        x_input = self.text_positional_embedding(x_input)

        # make initial y_input
        # make mask_interval and non_mask_interval
        y_len = y.shape[2]
        y_lens = torch.LongTensor([y_len]).to(y.device)
        mask_interval = mask_interval[0]
        if aug_context:
            mask_interval = [[item[0]+out_len, item[1]+out_len] for item in mask_interval]
        starts =  [item[0].item() for item in mask_interval] + [y_len]
        ends = [0] + [item[1].item() for item in mask_interval]
        mask_intervals = [
            (item[0].item(), item[1].item()) for item in mask_interval
        ] # a werid name change, mask_interval is input, now is mask_intervals, with one more dimension
        non_mask_intervals = [
            (ns, ne) for ns, ne in zip(ends, starts)
        ]

        # prepare input sequences
        rearranged_y = self.rearrange(y[0], non_mask_intervals, mask_intervals)
        shifted_y = self.shift(rearranged_y) # each element [K S], patterns is not used, as we directly use the original input y
        inserted_y, mask_position = self.insert_mask(shifted_y)
        cated_y, new_y_lens = self.cat_y(inserted_y) # KT
        num_task = len(mask_position)//2
        cated_y = cated_y[:, :mask_position[num_task]] # of shape [K,T] input of the network
        new_y_lens = torch.LongTensor([mask_position[num_task]]).to(cated_y.device)
        cated_y = cated_y.unsqueeze(0).permute(1,2,0) # B,K,T -> K,T,B
        if aug_text:
            cated_y = cated_y.repeat(1, 1, 2)
        embedded_y = self.embed_y(cated_y) #BTD
        
        if aug_text:
            x_padding_mask = torch.full((2, x_lens[0]), False).to(x.device)
            if cfg_pretrained:
                x_padding_mask[1:, 1:] = True
            past = torch.ones([self.args.num_decoder_layers, 2, 2], device=x.device, dtype=torch.float32) if kvcache else None
        else:
            x_padding_mask = torch.full((1, x_lens[0]), False).to(x.device)
            past = torch.ones([self.args.num_decoder_layers, 2, 1], device=x.device, dtype=torch.float32) if kvcache else None
            
        emb_inds = list(range(self.args.mts, self.args.mts+ self.args.max_n_spans))
        
        
        generated = []
        logging.info(f"silence tokens: {silence_tokens}, note that if you are not using the pretrained encodec 6f79c6a8, make sure you specified it yourself, rather than using the default")
            
        for idx in range(num_task):
            cur_generated = []
            prev_token = None
            consec_silence_count = 0
            num_gen = 0
            num_eog = 0
            num_cfg_tag = 1

            # add mask token
            mts = torch.LongTensor([emb_inds[idx]] * self.args.n_codebooks).unsqueeze(-1).to(embedded_y.device) # K, 1
            mts_emb = torch.stack([self.audio_embedding[k](mts[k]) for k in range(self.args.n_codebooks)], dim=0) # [K,1,D]
            mts_emb = mts_emb.sum(dim=0,keepdim=True) # [1,1,D]
            if aug_text:
                mts_emb = mts_emb.repeat(2,1,1)
            embedded_y = torch.cat([embedded_y, mts_emb], dim=1)
            # positional embedding
            y_input = self.audio_positional_embedding(embedded_y) # [B T D]
            # make attention mask and padding mask
            y_attention_mask = torch.triu(torch.ones(y_input.shape[1], y_input.shape[1]), diagonal=1).bool().to(y.device)
            new_y_lens = torch.LongTensor([y_input.shape[1]]).to(y.device)
            if aug_text:
                y_padding_mask = torch.full((2,new_y_lens[0]), False).to(y.device)
            else:
                y_padding_mask = torch.full((1,new_y_lens[0]), False).to(y.device)
            
            while True:
                # get model output
                y_out, present = self.dec_forward(
                                        x_input, 
                                        x_lens,
                                        x_attention_mask,
                                        x_padding_mask,
                                        y_input,
                                        new_y_lens,
                                        y_attention_mask,
                                        y_padding_mask,
                                        past=past,
                                        last_3_tokens=False
                                        )
                if past != None:
                    past = torch.cat([past, present.to(past.dtype)], dim=-2) if past.ndim > 3 else present.to(past.dtype)
                y_out = y_out[:, -1:] # only take the last one
                logits = torch.stack([self.predict_layer[i](y_out) for i in range(self.args.n_codebooks)], dim=1) # [B K S card], B==S==1, so [1 K 1 card]
                logits = logits.squeeze() # [K card]
                if aug_text:
                    if num_cfg_tag == cfg_stride:
                        logits = cfg_coef * logits[0] + (1 - cfg_coef) * logits[1]
                        num_cfg_tag = 1
                    else:
                        num_cfg_tag += 1
                        logits = logits[0]
                assert logits.shape == torch.Size((self.args.n_codebooks, self.n_audio_tokens[0])), f"{logits.shape}"
                # filter out mts, sos and eos
                for jj in range(self.args.n_codebooks):
                    logits[jj][self.args.eos] = -10000.
                    logits[jj][self.args.sos] = -10000.
                    for mts in range(self.args.mts, self.args.mts+ self.args.max_n_spans):
                        logits[jj][mts] = -10000.
                # add first empty tokens
                if num_gen < self.args.n_codebooks - 1:
                    for jj in range(num_gen + 1, self.args.n_codebooks):
                        logits[jj][self.args.empty_token] = 10000.
                # deal with eog token
                if num_eog > 0: # codebook 1 has produced eog token
                    for jj in range(num_eog+1,self.args.n_codebooks):
                        logits[jj][self.args.eog] = -10000
                        logits[jj][self.args.empty_token] = -10000
                    samples = topk_sampling(
                                logits, top_k=top_k, top_p=top_p, temperature=temperature
                            ) # [K, 1]
                    for jj in range(num_eog):
                        samples[jj, 0] = self.args.empty_token
                    samples[num_eog, 0] = self.args.eog
                    num_eog += 1
                else: # codebook 1 did not produce eog token
                    # filter out eog for codebook 2-4
                    for jj in range(1,self.args.n_codebooks):
                        logits[jj][self.args.eog] = -10000
                        
                    # silence repetition handling
                    if stop_repetition > 0 and prev_token in silence_tokens and consec_silence_count > stop_repetition:
                        if logits[0, prev_token] < 0:
                            logits[0, prev_token] = logits[0, prev_token] * (consec_silence_count - (stop_repetition-1))
                        else:
                            logits[0, prev_token] = logits[0, prev_token] / (consec_silence_count - (stop_repetition-1))
                    
                    samples = topk_sampling(
                                logits, top_k=top_k, top_p=top_p, temperature=temperature
                            ) # [K, 1]
                    
                    assert samples.shape == torch.Size((self.args.n_codebooks, 1)), f"samples.shape: {samples.shape}"
                    
                    if (
                        samples[0,0] == self.args.eog or torch.argmax(logits[0], dim=-1) == self.args.eog or y_input.shape[1] > x_lens[0] * 10
                    ): # last one means y is already too long, shouldn't happen, but put it here
                        samples[0,0] = self.args.eog
                        num_eog += 1
                    
                    if samples[0,0] in silence_tokens and samples[0,0] == prev_token:
                        consec_silence_count += 1
                    else:
                        consec_silence_count = 0
                    prev_token = samples[0,0]

                num_gen += 1
                cur_generated.append(samples.squeeze(-1))
                
                if num_eog == self.args.n_codebooks: # current span is done
                    break
                    
                # prepare input for next token prediction
                samples_emb = torch.stack([self.audio_embedding[k](samples[k]) for k in range(self.args.n_codebooks)], dim=0) # [K,1,D]
                samples_emb = samples_emb.sum(dim=0,keepdim=True) # [1,1,D]
                if aug_text:
                    samples_emb = samples_emb.repeat(2, 1, 1)
                embedded_y = torch.cat([embedded_y, samples_emb], dim=1)
                # positional embedding
                y_input = self.audio_positional_embedding(embedded_y) # [B T D]
                # make attention mask and padding mask
                y_attention_mask = torch.triu(torch.ones(y_input.shape[1], y_input.shape[1]), diagonal=1).bool().to(y.device)
                new_y_lens = torch.LongTensor([y_input.shape[1]]).to(y.device)
                if aug_text:
                    y_padding_mask = torch.full((2,new_y_lens[0]), False).to(y.device)
                else:
                    y_padding_mask = torch.full((1,new_y_lens[0]), False).to(y.device)
            generated.append(cur_generated)
        assert len(generated) == num_task, f"len(generated): {len(generated)}, num_task: {num_task}"

        # # combine non_masked_span with generated spans
        # first need to shift the generated part back
        flatten_gen = []
        for i, orig_span in enumerate(generated):
            span = torch.stack(orig_span, dim=0) # [T K]
            span = span.transpose(1,0) # [K, T]
            assert span.shape[0] == self.args.n_codebooks, span.shape
            unshifted_span = self.revert_pattern_sequence(pattern_tokens=span, n_q=self.args.n_codebooks, special_token=self.args.empty_token)
            assert unshifted_span.shape[1] == span.shape[1]-self.args.n_codebooks+1, f"unshifted_span:{unshifted_span.shape}, orig_span:{span.shape}"
            unshifted_span = unshifted_span[:,:-1] # remove eog token
            flatten_gen.append(unshifted_span)
                
        res = []
        marks = []
        masks = []
        tmp = 0
        for orig_interval, gen in zip(non_mask_intervals, flatten_gen):
            res.append(y[0, :, orig_interval[0]:orig_interval[1]])
            masks.append((tmp, tmp+orig_interval[1]-orig_interval[0]))
            tmp_mark = [0] * (orig_interval[1] - orig_interval[0])
            marks = [*marks, *tmp_mark]
            res.append(gen)
            tmp += orig_interval[1]-orig_interval[0] + gen.shape[-1]
            tmp_mark = [1] * gen.shape[-1]
            marks = [*marks, *tmp_mark]
        if y.shape[-1] != non_mask_intervals[-1][1] + 1: # edit last tokens or tts
            res.append(y[0, :, non_mask_intervals[-1][0]:non_mask_intervals[-1][1]])
            masks.append((tmp, tmp+non_mask_intervals[-1][1]-non_mask_intervals[-1][0]))
            tmp_mark = [0] * (non_mask_intervals[-1][1] - non_mask_intervals[-1][0])
            marks = [*marks, *tmp_mark]
        res = torch.cat(res, dim=1).unsqueeze(0) # [K,new_T] -> [1, K, new_T]
        marks = torch.LongTensor(marks).unsqueeze(0)
        if aug_context:
            res = res[:, :, out_len:]
            marks = marks[:, out_len:]
            masks = [(item[0]-out_len, item[1]-out_len) for item in masks]
            non_mask_intervals = [(item[0]-out_len, item[1]-out_len) for item in non_mask_intervals]

        return res, marks, masks, non_mask_intervals


if __name__ == "__main__":
    # debug
    pass
