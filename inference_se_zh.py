# @ hwang258@jh.edu

import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
os.environ["USER"] = "root" # TODO change this to your username

import shutil
import torch
import torchaudio
import numpy as np
import random
from argparse import Namespace
from data.tokenizer import (
    AudioTokenizer,
    TextTokenizer,
)
import torchaudio
import torchaudio.transforms as transforms
from edit_utils_zh import parse_edit
from inference_scale import get_mask_interval
from inference_scale import inference_one_sample_se
import time
from tqdm import tqdm

# hyperparameters for inference
sub_amount = 0.16
codec_audio_sr = 16000
codec_sr = 50
top_k = 0
top_p = 0.8
temperature = 1
kvcache = 1
seed = 1
silence_tokens = [1388,1898,131] # if there are long silence in the generated audio, reduce the stop_repetition to 3, 2 or even 1
stop_repetition = 2 # -1 means do not adjust prob of silence tokens. if there are long silence or unnaturally strecthed words, increase sample_batch_size to 2, 3 or even 4
sample_batch_size = 1 # what this will do to the model is that the model will run sample_batch_size examples of the same audio
cfg_coef = 1.5
aug_text = True
aug_context = False
cfg_pretrained = False
use_watermark = True

def seed_everything(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
seed_everything(seed)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"using {device}")

from models import ssr
filepath = os.path.join('./pretrained_models/Chinese_25k/e830M/', "best_bundle.pth")
ckpt = torch.load(filepath, map_location="cpu")
model = ssr.SSR_Speech(ckpt["config"])
model.load_state_dict(ckpt["model"])
config = vars(model.args)
phn2num = ckpt["phn2num"]
model.to(device)
model.eval()
encodec_fn = "./pretrained_models/WMEncodec/checkpoint.th"
audio_tokenizer = AudioTokenizer(signature=encodec_fn) # will also put the neural codec model on gpu
text_tokenizer = TextTokenizer(backend="espeak", language='cmn')


def main(orig_audio, orig_transcript, target_transcript, temp_folder, output_dir, savename, savetag=1, mfa=False, use_downloaded_mfa=True, mfa_dict_path=None, mfa_path=None):
    
    start_time = time.time()

    # move the audio and transcript to temp folder
    os.makedirs(temp_folder, exist_ok=True)
    os.system(f"cp {orig_audio} {temp_folder}")
    filename = os.path.splitext(orig_audio.split("/")[-1])[0]
    with open(f"{temp_folder}/{filename}.txt", "w") as f:
        f.write(orig_transcript)

    # resampling audio to 16k Hz
    import librosa
    import soundfile as sf
    audio, sr = librosa.load(os.path.join(temp_folder, filename+'.wav'), sr=16000)
    sf.write(os.path.join(temp_folder, filename+'.wav'), audio, 16000)
        
    # run MFA to get the alignment
    align_temp = f"{temp_folder}/mfa_alignments"
    os.makedirs(align_temp, exist_ok=True)
    if mfa:
        if use_downloaded_mfa:
            os.system(f"mfa align --overwrite -j 1 --output_format csv {temp_folder} {mfa_dict_path} {mfa_path} {align_temp} --clean")
        else:
            os.system(f"mfa align --overwrite -j 1 --output_format csv {temp_folder} mandarin_china_mfa mandarin_mfa {align_temp} --clean")

    audio_fn = f"{temp_folder}/{filename}.wav"
    transcript_fn = f"{temp_folder}/{filename}.txt"
    align_fn = f"{align_temp}/{filename}.csv"

    # run the script to turn user input to the format that the model can take
    operations, orig_spans = parse_edit(orig_transcript, target_transcript)
    print(operations)
    print("orig_spans: ", orig_spans)
    
    if len(orig_spans) > 3:
        raise RuntimeError("Current model only supports maximum 3 editings")
        
    starting_intervals = []
    ending_intervals = []
    for orig_span in orig_spans:
        start, end = get_mask_interval(align_fn, orig_span)
        starting_intervals.append(start)
        ending_intervals.append(end)

    print("intervals: ", starting_intervals, ending_intervals)

    info = torchaudio.info(audio_fn)
    audio_dur = info.num_frames / info.sample_rate
    
    def combine_spans(spans, threshold=0.2):
        spans.sort(key=lambda x: x[0])
        combined_spans = []
        current_span = spans[0]

        for i in range(1, len(spans)):
            next_span = spans[i]
            if current_span[1] >= next_span[0] - threshold:
                current_span[1] = max(current_span[1], next_span[1])
            else:
                combined_spans.append(current_span)
                current_span = next_span
        combined_spans.append(current_span)
        return combined_spans
    
    morphed_span = [(max(start - sub_amount, 0), min(end + sub_amount, audio_dur))
                    for start, end in zip(starting_intervals, ending_intervals)] # in seconds
    morphed_span = combine_spans(morphed_span, threshold=0.2)
    print("morphed_spans: ", morphed_span)
    os.makedirs(output_dir, exist_ok=True)
    save_morphed_span = f"{output_dir}/{savename}_mask.pt"
    torch.save(morphed_span, save_morphed_span)
    mask_interval = [[round(span[0]*codec_sr), round(span[1]*codec_sr)] for span in morphed_span]
    mask_interval = torch.LongTensor(mask_interval) # [M,2], M==1 for now

    decode_config = {'top_k': top_k, 'top_p': top_p, 'temperature': temperature, 'stop_repetition': stop_repetition, 'kvcache': kvcache, "codec_audio_sr": codec_audio_sr, "codec_sr": codec_sr, "silence_tokens": silence_tokens}
    
    for num in tqdm(range(sample_batch_size)):
        seed_everything(seed+num)
        orig_audio, new_audio = inference_one_sample_se(model, Namespace(**config), phn2num, text_tokenizer, audio_tokenizer, audio_fn, orig_transcript, target_transcript, mask_interval, cfg_coef, aug_text, aug_context, cfg_pretrained, use_watermark, device, decode_config)
        # save segments for comparison
        orig_audio, new_audio = orig_audio[0].cpu(), new_audio[0].cpu()
        save_fn_new = f"{output_dir}/{savename}_new_seed{seed+num}.wav"
        torchaudio.save(save_fn_new, new_audio, codec_audio_sr)
    
    save_fn_orig = f"{output_dir}/{savename}_orig.wav"
    torchaudio.save(save_fn_orig, orig_audio, codec_audio_sr)
        
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    print(f"Running time: {elapsed_time:.4f} s")


if __name__ == "__main__":
    
    orig_audio = "./demo/pony.wav"
    orig_transcript =    "能够更有效率地结合给用户提升更多的这种体验也包括他的这个他的后台的效率提升等等我相信这些额额业界的解决方案应该说是"
    target_transcript =  "能够更有效率地结合给用户提升更多的体验也包括他的这个他的后台的效率提升等等在这个基础上我相信这些业界的解决方案应该说是"
    temp_folder = "./demo/temp_test3"
    output_dir = f"./demo/generated_se"
    savename = 'pony'
    savetag = 1
    mfa=False
    use_downloaded_mfa=True
    mfa_dict_path = "./pretrained_models/mandarin_china_mfa.dict"
    mfa_path = "./pretrained_models/mandarin_mfa.zip"
    
    main(
        orig_audio=orig_audio, 
        orig_transcript=orig_transcript, 
        target_transcript=target_transcript, 
        temp_folder=temp_folder, 
        output_dir=output_dir, 
        savename=savename, 
        savetag=savetag,
        mfa=mfa, 
        use_downloaded_mfa=use_downloaded_mfa, 
        mfa_dict_path=mfa_dict_path, 
        mfa_path=mfa_path
    )



