# @ hwang258@jh.edu

import argparse
def parse_args():
    parser = argparse.ArgumentParser(description="encode the librilight dataset using encodec model")
    parser.add_argument("--audiopath", type=str, default=None)
    parser.add_argument('--save_dir', type=str, default=None)
    parser.add_argument('--save_tag', type=str, default='encodec_16khz_4codebooks')
    parser.add_argument('--dataset_name', type=str, default=None)
    parser.add_argument('--encodec_model_path', type=str, default=None)
    parser.add_argument('--n_workers', type=int, default=4, help="Number of parallel worker processes")
    parser.add_argument('--mega_batch_size', type=int, default=120, help="Number of samples in each mega batch for multiprocess dataloading")
    parser.add_argument('--batch_size', type=int, default=32, help="batch size for encodec encoding, decrease it if OOM. This is the sum of batch size *over each gpu*, so increase it if you are using more gpus")
    parser.add_argument('--model_sr', type=int, default=16000, help='encodec input audio sample rate')
    parser.add_argument('--downsample_rate', type=int, default=320, help='encodec downsample rate')
    parser.add_argument('--model_code_sr', type=int, default=50, help='encodec model code sample rate')
    parser.add_argument('--len_cap', type=float, default=20.0, help='will drop audios that are longer than this number')
    parser.add_argument('--start', type=int, default=0, help='start index for parallel processing')
    parser.add_argument('--end', type=int, default=500000, help='end index for parallel processing')
    parser.add_argument('--max_len', type=int, default=30000, help='max length of audio in samples, if exceed, will cut a batch into half to process, decrease this number if OOM on your machine')
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
    import tqdm
    import time
    import torchaudio
    from datasets import load_dataset, DownloadConfig
    import pandas as pd
    from tokenizer import TextTokenizer, tokenize_text
    import torchaudio.transforms as transforms
    
    # get the path encodec_16khz_4codebooks
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
        def __init__(self, args, split):
            super().__init__()
            import glob
            self.data = glob.glob(os.path.join(args.audiopath, "*.wav"))
            self.data = self.data[args.start:args.end]

        def checkout(self, data):
            out = []
            for ind in range(len(data)):
                segment_id = data[ind].split('/')[-1].split(".wav")[0]
                save_fn = os.path.join(codes_save_root, segment_id+".txt")
                if not os.path.exists(save_fn):
                    out.append(data[ind])
            return out
            
        def __len__(self):
            return len(self.data)
        def __getitem__(self, ind):
            segment_id = self.data[ind].split('/')[-1].split(".wav")[0]
            if os.path.exists(self.data[ind]):
                audio, sr = torchaudio.load(self.data[ind])
            else:
                audio, sr = torchaudio.load(self.data[ind].replace('/apdcephfs_cq2', '/apdcephfs_cq2_1297902'))
            if sr != 16000:
                resampler = transforms.Resample(orig_freq=sr, new_freq=16000)
                audio = resampler(audio)
            duration = audio.shape[1] / sr
            return segment_id, audio.squeeze(), sr, duration
        def collate(self, batch):
            res = {'segment_id': [], "audio": [], "sr": [], "duration":[]}
            for item in batch:
                if item[0] != None:
                    res['segment_id'].append(item[0])
                    res['audio'].append(item[1])
                    res['sr'].append(item[2])
                    res['duration'].append(item[3])
            return res


    ## encodec codes extraction
    logging.info("encodec encoding...")
    train_dataset = mydataset(args, 'train')
    print(len(train_dataset))
    train_loader = torch.torch.utils.data.DataLoader(train_dataset, batch_size=args.mega_batch_size, shuffle=False, drop_last=False, num_workers=args.n_workers, collate_fn=train_dataset.collate)
    splits = ['train']
    loaders = [train_loader]

    for split, loader in zip(splits, loaders):
        skip = 0
        logging.info(f"now processing split {split}...")
        for m, mega_batch in enumerate(loader):
            logging.info(f"====================================")
            logging.info(f"====================================")
            lengths = np.array(mega_batch['duration'])
            sorted_inds = sort_by_audio_len(lengths)
            for j in range(len(sorted_inds))[::-1]:
                if lengths[sorted_inds[j]] < 0.2 or lengths[sorted_inds[j]] > args.len_cap: # skip samples that are too short (shorter than 0.2s), or too big (bigger than 80s)
                    skip += 1
                    del sorted_inds[j]
            
            n_steps = int(np.ceil(len(sorted_inds) / args.batch_size))
            for n in tqdm.tqdm(range(n_steps), disable=True):
                inds_used = sorted_inds[n*args.batch_size:(n+1)*args.batch_size]
                audio_batch = [mega_batch['audio'][id] for id in inds_used]
                sr_batch = [mega_batch['sr'][id] for id in inds_used]
                segment_id_batch = [mega_batch['segment_id'][id] for id in inds_used]
                padded_wav = torch.nn.utils.rnn.pad_sequence(audio_batch, batch_first=True).unsqueeze(1) # [B, T] -> [B, 1, T]
                all_lens = [lengths[id] for id in inds_used]
                with torch.no_grad():
                    if max(all_lens) > args.max_len and len(all_lens) > 1: # NOTE decrease args.max_len if OOM, or chunk it into more than 2 forward passes
                        codes = []
                        inwav = padded_wav.cuda()
                        codes.append(model.encode(inwav[:len(inwav)//2])[0].cpu())
                        codes.append(model.encode(inwav[len(inwav)//2:])[0].cpu())
                        codes = torch.cat(codes, dim=0)
                    else:
                        encoded_frames = model.encode(padded_wav.cuda())
                        # logging.info(f"encoded_frames: {encoded_frames[0].shape}")
                        codes = encoded_frames[0].cpu()

                for i, length in enumerate(all_lens):
                    save_fn = os.path.join(codes_save_root, segment_id_batch[i]+".txt")
                    if not os.path.exists(save_fn):
                        actual_len = round(length * args.model_code_sr) # 320 is downsample rate for this model
                        cur_code = codes[i].tolist() if type(codes) == list else codes[i, :, :actual_len].tolist()
                        write_array_to_txt_file(cur_code, save_fn)
