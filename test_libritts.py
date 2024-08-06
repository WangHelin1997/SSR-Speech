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
from edit_utils_en import parse_edit
from inference_scale import get_mask_interval
from inference_scale import inference_one_sample_voicecraft
import time
from tqdm import tqdm

import pandas as pd

file_path = './RealEdit.txt'
audiopath = '/apdcephfs_cq10/share_1603164/user/helinhwang/libritts/LibriTTS/'
df = pd.read_csv(file_path, delimiter='\t')
data_dict = []

for index, row in df.iterrows():
    wav_file = row['wav_fn']
    if not wav_file.startswith('YOU') and not wav_file.startswith('show_'):
        wav_path = os.path.join(audiopath, 'dev-clean', wav_file.split('_')[0], wav_file.split('_')[1], wav_file)
        if os.path.exists(wav_path):
            orig_transcript = row['orig_transcript']
            if '|' in orig_transcript:
                orig_transcript = orig_transcript.split('|')[0]
                
            new_transcript = row['new_transcript']
            if '|' in new_transcript:
                new_transcript = new_transcript.split('|')[-1]
            data_dict.append([wav_file, wav_path, orig_transcript, new_transcript])
            
        wav_path = os.path.join(audiopath, 'dev-other', wav_file.split('_')[0], wav_file.split('_')[1], wav_file)
        if os.path.exists(wav_path):
            orig_transcript = row['orig_transcript']
            if '|' in orig_transcript:
                orig_transcript = orig_transcript.split('|')[0]
                
            new_transcript = row['new_transcript']
            if '|' in new_transcript:
                new_transcript = new_transcript.split('|')[-1]
            data_dict.append([wav_file, wav_path, orig_transcript, new_transcript])
print(f"Processing: {len(data_dict)} files...")


# hyperparameters for inference
sub_amount = 0.08
codec_audio_sr = 16000
codec_sr = 50
top_k = 0
top_p = 0.8
temperature = 1
kvcache = 1
seed = 1
silence_tokens = [1388,1898,131] # if there are long silence in the generated audio, reduce the stop_repetition to 3, 2 or even 1
stop_repetition = 2 # -1 means do not adjust prob of silence tokens. if there are long silence or unnaturally strecthed words, increase sample_batch_size to 2, 3 or even 4
sample_batch_size = 5 # what this will do to the model is that the model will run sample_batch_size examples of the same audio, and pick the one that's the shortest

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
voicecraft_name="best_bundle.pth"

# # the old way of loading the model
from models import voicecraft
filepath = os.path.join('/apdcephfs_cq10/share_1603164/user/helinhwang/VoiceCraft/pretrained_models/English_10k/e830M/', voicecraft_name)
ckpt = torch.load(filepath, map_location="cpu")
model = voicecraft.VoiceCraft(ckpt["config"])
model.load_state_dict(ckpt["model"])
config = vars(model.args)
phn2num = ckpt["phn2num"]
model.to(device)
model.eval()
encodec_fn = "/apdcephfs_cq10/share_1603164/user/helinhwang/VoiceCraft/pretrained_models/VoiceCraft/encodec_4cb2048_giga.th"
audio_tokenizer = AudioTokenizer(signature=encodec_fn) # will also put the neural codec model on gpu
text_tokenizer = TextTokenizer(backend="espeak")

def preprocess(temp_folder, mfa=False, use_downloaded_mfa=True, mfa_dict_path=None, mfa_path=None):
    
    os.makedirs(temp_folder, exist_ok=True)
    for item in tqdm(data_dict):
        orig_audio = item[1]
        orig_transcript = item[2]
        
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
            os.system(f"mfa align --overwrite -j 1 --output_format csv {temp_folder} english_us_arpa english_us_arpa {align_temp} --clean")

        

def main(filename, orig_transcript, target_transcript, temp_folder, output_dir, savename):
    
    start_time = time.time()
    align_temp = f"{temp_folder}/mfa_alignments"
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
    morphed_span = combine_spans(morphed_span, threshold=1/codec_sr)
    print("morphed_spans: ", morphed_span)
    save_morphed_span = f"{output_dir}/{savename}_mask.pt"
    os.makedirs(output_dir, exist_ok=True)
    torch.save(morphed_span, save_morphed_span)

    mask_interval = [[round(span[0]*codec_sr), round(span[1]*codec_sr)] for span in morphed_span]
    mask_interval = torch.LongTensor(mask_interval) # [M,2], M==1 for now

    decode_config = {'top_k': top_k, 'top_p': top_p, 'temperature': temperature, 'stop_repetition': stop_repetition, 'kvcache': kvcache, "codec_audio_sr": codec_audio_sr, "codec_sr": codec_sr, "silence_tokens": silence_tokens}
    new_audios = []
    for num in tqdm(range(sample_batch_size)):
        seed_everything(seed+num)
        orig_audio, new_audio = inference_one_sample_voicecraft(model, Namespace(**config), phn2num, text_tokenizer, audio_tokenizer, audio_fn, target_transcript, mask_interval, device, decode_config)
        # save segments for comparison
        orig_audio, new_audio = orig_audio[0].cpu(), new_audio[0].cpu()
        new_audios.append(new_audio)

    for num in range(sample_batch_size):
        # print(new_audios[num].shape)
        if new_audios[num].shape[0] < new_audio.shape[0]:
            new_audio = new_audios[num]
    
    os.makedirs(output_dir, exist_ok=True)
    for num in range(sample_batch_size):
        save_fn_new = f"{output_dir}/{savename}_new_seed{seed+num}.wav"
        torchaudio.save(save_fn_new, new_audios[num], codec_audio_sr)
        
    save_fn_new = f"{output_dir}/{savename}_new_seed{seed}_final.wav"
    torchaudio.save(save_fn_new, new_audio, codec_audio_sr)
    
    save_fn_orig = f"{output_dir}/{savename}_orig.wav"
    if not os.path.isfile(save_fn_orig):
        orig_audio, orig_sr = torchaudio.load(audio_fn)
        if orig_sr != codec_audio_sr:
            orig_audio = torchaudio.transforms.Resample(orig_sr, codec_audio_sr)(orig_audio)
        torchaudio.save(save_fn_orig, orig_audio, codec_audio_sr)
        
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    print(f"Running time: {elapsed_time:.4f} s")


if __name__ == "__main__":
    
    temp_folder = "./demo/temp_voicecraft_libritts"
    output_dir = "./demo/voicecraft_libritts_se"
    mfa=True
    use_downloaded_mfa=True
    mfa_dict_path = "/apdcephfs_cq10/share_1603164/user/helinhwang/VoiceCraft/pretrained_models/english_us_arpa.dict"
    mfa_path = "/apdcephfs_cq10/share_1603164/user/helinhwang/VoiceCraft/pretrained_models/english_us_arpa.zip"

    # preprocess(
    #     temp_folder=temp_folder, 
    #     mfa=mfa,
    #     use_downloaded_mfa=use_downloaded_mfa, 
    #     mfa_dict_path=mfa_dict_path, 
    #     mfa_path=mfa_path
    # )
    
    for item in data_dict:
        filename = item[0].split(".wav")[0]
        orig_transcript = item[2]
        target_transcript = item[3]
        savename = filename
        main(
            filename=filename, 
            orig_transcript=orig_transcript, 
            target_transcript=target_transcript, 
            temp_folder=temp_folder, 
            output_dir=output_dir, 
            savename=savename
        )



