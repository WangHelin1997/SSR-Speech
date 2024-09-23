# @ hwang258@jh.edu

import json
import argparse
def parse_args():
    parser = argparse.ArgumentParser(description="encode the dataset using encodec model")
    parser.add_argument("--json_path", type=str, default=None)
    parser.add_argument('--save_dir', type=str, default=None)
    parser.add_argument('--save_tag', type=str, default='wmencodec')
    parser.add_argument('--dataset_name', type=str, default=None)
    parser.add_argument('--encodec_model_path', type=str, default=None)
    parser.add_argument('--n_workers', type=int, default=8, help="Number of parallel worker processes")
    parser.add_argument('--batch_size', type=int, default=64, help="batch size for encodec encoding, decrease it if OOM. This is the sum of batch size *over each gpu*, so increase it if you are using more gpus")
    parser.add_argument('--model_sr', type=int, default=16000, help='encodec input audio sample rate')
    parser.add_argument('--downsample_rate', type=int, default=320, help='encodec downsample rate')
    parser.add_argument('--model_code_sr', type=int, default=50, help='encodec model code sample rate')
    parser.add_argument('--start', type=int, default=0, help='start index for parallel processing')
    parser.add_argument('--end', type=int, default=500000, help='end index for parallel processing')
    return parser.parse_args()
    
if __name__ == "__main__":
    import logging
    formatter = (
        "%(asctime)s [%(levelname)s] %(filename)s:%(lineno)d || %(message)s"
    )
    logging.basicConfig(format=formatter, level=logging.INFO)
    args = parse_args()

    import os
    os.environ["USER"] = "root"
    import numpy as np
    import torch
    from tqdm import tqdm
    import time
    import torchaudio
    from datasets import load_dataset, DownloadConfig
    import pandas as pd
    from tokenizer import TextTokenizer, tokenize_text
    import torchaudio.transforms as transforms
    
    codes_save_root = os.path.join(args.save_dir, args.dataset_name, args.save_tag)
    os.makedirs(codes_save_root, exist_ok=True)


    def sort_by_audio_len(lens):
        inds = np.argsort(lens).tolist()
        logging.info(f"longest: {lens[inds[-1]]*args.model_code_sr} encodec codes, {lens[inds[-1]]:.2f} sec.")
        logging.info(f"shortest: {lens[inds[0]]*args.model_code_sr} encodec codes, {lens[inds[0]]:.2f} sec.")
        logging.info(f"median: {lens[inds[len(inds)//2]]*args.model_code_sr} encodec codes, {lens[inds[len(inds)//2]]:.2f} sec.")
        logging.info(f"95 percentile longest: {lens[inds[int(len(inds)*0.95)]]*args.model_code_sr} encodec codes, {lens[inds[int(len(inds)*0.95)]]:.2f} sec.")
        return inds[::-1]
    
    def write_array_to_txt_file(array, filename):
        with open(filename, 'w') as f:
            for a in array[:-1]:
                f.write(' '.join(map(str, a))+'\n')
            f.write(' '.join(map(str, array[-1])))
    

    ### phonemization
    # load tokenizer
    # load the encodec model
    from audiocraft.solvers import WMCompressionSolver
    model = WMCompressionSolver.model_from_checkpoint(args.encodec_model_path)
    model = model.cuda()
    model = model.eval()

        
    class mydataset(torch.utils.data.Dataset):
        def __init__(self, args):
            super().__init__()
            with open(args.json_path, 'r') as json_file:
                self.data = json.load(json_file)
            self.data = self.data[args.start:args.end]
            self.args = args
            
        def __len__(self):
            return len(self.data)
        
        def __getitem__(self, ind):
            segment_id = self.data[ind]["segment_id"]
            audio, sr = torchaudio.load(self.data[ind]["wav"])
            if sr != self.args.model_sr:
                resampler = transforms.Resample(orig_freq=sr, new_freq=self.args.model_sr)
                audio = resampler(audio)
            duration = audio.shape[1] / sr
            return segment_id, audio.squeeze(), duration
        
        def collate(self, batch):
            res = {'segment_id': [], "audio": [], "duration":[]}
            for item in batch:
                if item[0] != None:
                    res['segment_id'].append(item[0])
                    res['audio'].append(item[1])
                    res['duration'].append(item[2])
            return res


    ## encodec codes extraction
    logging.info("encodec encoding...")
    train_dataset = mydataset(args)
    print(len(train_dataset))
    loader = torch.torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=args.n_workers, collate_fn=train_dataset.collate)
    for batch in tqdm(loader):
        lengths = np.array(batch['duration'])
        audio_batch = batch['audio']
        segment_id_batch = batch['segment_id']
        padded_wav = torch.nn.utils.rnn.pad_sequence(audio_batch, batch_first=True).unsqueeze(1) # [B, T] -> [B, 1, T]
        with torch.no_grad():
            encoded_frames = model.encode(padded_wav.cuda())
            codes = encoded_frames[0].cpu()
        for i, length in enumerate(lengths):
            save_fn = os.path.join(codes_save_root, segment_id_batch[i]+".txt")
            if not os.path.exists(save_fn):
                actual_len = round(length * args.model_code_sr)
                cur_code = codes[i].tolist() if type(codes) == list else codes[i, :, :actual_len].tolist()
                write_array_to_txt_file(cur_code, save_fn)
