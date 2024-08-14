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
from edit_utils_en import parse_tts
from inference_scale import get_mask_interval
from inference_scale import inference_one_sample_tts
import time
from tqdm import tqdm
import shutil

# hyperparameters for inference
sub_amount = 0.01
codec_audio_sr = 16000
codec_sr = 50
top_k = 0
top_p = 0.8
temperature = 1
kvcache = 1
seed = 1 # random seed magic
silence_tokens = [1388,1898,131] # if there are long silence in the generated audio, reduce the stop_repetition to 3, 2 or even 1
stop_repetition = 2 # -1 means do not adjust prob of silence tokens. if there are long silence or unnaturally strecthed words, increase sample_batch_size to 2, 3 or even 4
sample_batch_size = 1 # what this will do to the model is that the model will run sample_batch_size examples of the same audio
cfg_coef = 1.5
aug_text = True


# what this will do to the model is that the model will run sample_batch_size examples of the same audio, and pick the one that's the shortest
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
filepath = os.path.join('./pretrained_models/English_10k/e830M/', "best_bundle.pth")
ckpt = torch.load(filepath, map_location="cpu")
model = ssr.SSR_Speech(ckpt["config"])
model.load_state_dict(ckpt["model"])
config = vars(model.args)
phn2num = ckpt["phn2num"]
model.to(device)
model.eval()
encodec_fn = "./pretrained_models/VoiceCraft/encodec_4cb2048_giga.th"
audio_tokenizer = AudioTokenizer(signature=encodec_fn) # will also put the neural codec model on gpu
text_tokenizer = TextTokenizer(backend="espeak")


def main(orig_audio, orig_transcript, target_transcript, temp_folder, output_dir, savename, savetag=1, mfa=False, use_downloaded_mfa=True, mfa_dict_path=None, mfa_path=None):
    
    start_time = time.time()
    # move the audio and transcript to temp folder
    os.makedirs(temp_folder, exist_ok=True)
    os.system(f"cp {orig_audio} {temp_folder}")
    
    filename = os.path.splitext(orig_audio.split("/")[-1])[0]
    with open(f"{temp_folder}/{filename}.txt", "w") as f:
        f.write(orig_transcript)

    # resampling
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
            os.system(f"mfa align --overwrite -j 1 --output_format csv {temp_folder} english_us_arpa english_us_arpa {align_temp} --clean")

    audio_fn = f"{temp_folder}/{filename}.wav"
    transcript_fn = f"{temp_folder}/{filename}.txt"
    align_fn = f"{align_temp}/{filename}.csv"

    # run the script to turn user input to the format that the model can take
    orig_spans = parse_tts(orig_transcript, target_transcript)
    print("orig_spans: ", orig_spans)
        
    starting_intervals = []
    ending_intervals = []
    for orig_span in orig_spans:
        start, end = get_mask_interval(align_fn, orig_span)
        starting_intervals.append(start)
        ending_intervals.append(end)

    print("intervals: ", starting_intervals, ending_intervals)

    info = torchaudio.info(audio_fn)
    audio_dur = info.num_frames / info.sample_rate
    
    morphed_span = [(max(start, 1/codec_sr), min(end, audio_dur))
                    for start, end in zip(starting_intervals, ending_intervals)] # in seconds
    mask_interval = [[round(span[0]*codec_sr), round(span[1]*codec_sr)] for span in morphed_span]
    mask_interval = torch.LongTensor(mask_interval) # [M,2], M==1 for now
    print("mask_interval: ", mask_interval)
    
    decode_config = {'top_k': top_k, 'top_p': top_p, 'temperature': temperature, 'stop_repetition': stop_repetition, 'kvcache': kvcache, "codec_audio_sr": codec_audio_sr, "codec_sr": codec_sr, "silence_tokens": silence_tokens}

    os.makedirs(output_dir, exist_ok=True)
    for num in tqdm(range(sample_batch_size)):
        seed_everything(seed+num)
        new_audio = inference_one_sample_tts(model, Namespace(**config), phn2num, text_tokenizer, audio_tokenizer, audio_fn, orig_transcript, target_transcript, mask_interval, cfg_coef, aug_text, device, decode_config)
        # save segments for comparison
        new_audio = new_audio[0].cpu()
        save_fn_new = f"{output_dir}/{savename}_new_seed{seed+num}.wav"
        torchaudio.save(save_fn_new, new_audio, codec_audio_sr)
    
    save_fn_orig = f"{output_dir}/{savename}_orig.wav"
    shutil.copyfile(audio_fn, save_fn_orig)
        
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    print(f"Running time: {elapsed_time:.4f} s")


if __name__ == "__main__":
    
    orig_audio = "./demo/5895_34622_000026_000002.wav"
    orig_transcript =    "Gwynplaine had, besides, for his work and for his feats of strength, round his neck and over his shoulders, an esclavine of leather."
    target_transcript =  "Gwynplaine had, besides, for his work and for his feats of strength, I cannot believe that the same model can also do text to speech synthesis too!"
    temp_folder = "./demo/temp_test2"
    output_dir = f"./demo/generated_tts"
    savename = '5895_34622_000026_000002'
    savetag = 1
    mfa=False
    use_downloaded_mfa=True
    mfa_dict_path = "./pretrained_models/english_us_arpa.dict"
    mfa_path = "./pretrained_models/english_us_arpa.zip"
    
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


