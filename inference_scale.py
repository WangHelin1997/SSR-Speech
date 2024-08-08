# @ hwang258@jh.edu

import argparse, pickle
import logging
import os, random
import numpy as np
import torch
import torchaudio

from data.tokenizer import (
    AudioTokenizer,
    TextTokenizer,
    tokenize_audio,
    tokenize_text
)
import time

@torch.no_grad()
def inference_one_sample(model, model_args, phn2num, text_tokenizer, audio_tokenizer, audio_fn, prompt_text, target_text, mask_interval, cfg_coef, aug_text, aug_context, cfg_pretrained, device, decode_config):
    # phonemize
    text_tokens = [phn2num[phn] for phn in
            tokenize_text(
                text_tokenizer, text=target_text.strip()
            ) if phn in phn2num
        ]
    text_tokens = torch.LongTensor(text_tokens).unsqueeze(0)
    text_tokens_lens = torch.LongTensor([text_tokens.shape[-1]])

    prompt_text_tokens = [phn2num[phn] for phn in
        tokenize_text(
            text_tokenizer, text=prompt_text.strip()
        ) if phn in phn2num
    ]
    prompt_text_tokens = torch.LongTensor(prompt_text_tokens).unsqueeze(0)
    prompt_text_tokens_lens = torch.LongTensor([prompt_text_tokens.shape[-1]])

    encoded_frames = tokenize_audio(audio_tokenizer, audio_fn)
    original_audio = encoded_frames[0][0].transpose(2,1) # [1,T,K]
    assert original_audio.ndim==3 and original_audio.shape[0] == 1 and original_audio.shape[2] == model_args.n_codebooks, original_audio.shape
    logging.info(f"with direct encodec encoding before input, original audio length: {original_audio.shape[1]} codec frames, which is {original_audio.shape[1]/decode_config['codec_sr']:.2f} sec.")

    # forward
    stime = time.time()
    encoded_frames = model.inference(
        text_tokens.to(device),
        text_tokens_lens.to(device),
        prompt_text_tokens.to(device),
        prompt_text_tokens_lens.to(device),
        original_audio[...,:model_args.n_codebooks].to(device), # [1,T,8]
        original_audio[...,:model_args.n_codebooks].to(device), # [1,T,8]
        mask_interval=mask_interval.unsqueeze(0).to(device),
        top_k=decode_config['top_k'],
        top_p=decode_config['top_p'],
        temperature=decode_config['temperature'],
        stop_repetition=decode_config['stop_repetition'],
        kvcache=decode_config['kvcache'],
        silence_tokens=eval(decode_config['silence_tokens']) if type(decode_config['silence_tokens']) == str else decode_config['silence_tokens'],
        cfg_coef=cfg_coef,
        aug_text=aug_text,
        aug_context=aug_context,
        cfg_pretrained=cfg_pretrained,
    ) # output is [1,K,T]
    logging.info(f"inference on one sample take: {time.time() - stime:.4f} sec.")
    if type(encoded_frames) == tuple:
        encoded_frames = encoded_frames[0]
    logging.info(f"generated encoded_frames.shape: {encoded_frames.shape}, which is {encoded_frames.shape[-1]/decode_config['codec_sr']} sec.")

    # decode (both original and generated)
    original_sample = audio_tokenizer.decode(
        [(original_audio.transpose(2,1), None)] # [1,T,8] -> [1,8,T]
    )
    generated_sample = audio_tokenizer.decode(
        [(encoded_frames, None)]
    )

    return original_sample, generated_sample


def get_mask_interval(ali_fn, word_span):
    with open(ali_fn, "r") as rf:
        data = [l.strip().split(",") for l in rf.readlines()]
        data = data[1:]
    data = [item for item in data if item[3] == 'words']
    # print(data)
    s, e = word_span[0], word_span[1]
    assert s <= e, f"s:{s}, e:{e}"
    assert s >= 0, f"s:{s}"
    assert e <= len(data), f"e:{e}"
    if e == 0: # start
        start = 0.
        end = float(data[0][0])
    elif s == len(data): # end
        start = float(data[-1][1])
        end = float(data[-1][1]) # don't know the end yet
    elif s == e: # insert
        start = float(data[s-1][1])
        end = float(data[s][0])
    else:
        start = float(data[s-1][1]) if s > 0 else float(data[s][0])
        end = float(data[e][0]) if e < len(data) else float(data[-1][1])

    return (start, end)

if __name__ == "__main__":
    pass