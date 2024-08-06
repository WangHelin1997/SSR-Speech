# @ hwang258@jh.edu

import argparse
def parse_args():
    parser = argparse.ArgumentParser(description="encode the librilight dataset using encodec model")
    parser.add_argument("--dataset_name", type=str, default='English', help='name of dataset')
    parser.add_argument('--dataset_dir', type=str, default=None, help="dataset path")
    parser.add_argument('--save_dir', type=str, default=None, help="path to the manifest, phonemes, and encodec codes dirs")
    return parser.parse_args()

if __name__ == "__main__":
    import logging
    formatter = (
        "%(asctime)s [%(levelname)s] %(filename)s:%(lineno)d || %(message)s"
    )
    logging.basicConfig(format=formatter, level=logging.INFO)
    args = parse_args()

    import os
    import numpy as np
    import torch
    import tqdm
    import time
    import pandas as pd
    import multiprocessing
    from tokenizer import TextTokenizer, tokenize_text
    
    # get the path
    phn_save_root = os.path.join(args.save_dir, args.dataset_name, "phonemes")
    os.makedirs(phn_save_root, exist_ok=True)

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
    text_tokenizer = TextTokenizer(backend="espeak") # add language='cmn' when you process mandarin

    punc2sym = {" <COMMA>": ",", " <PERIOD>": ".", " <QUESTIONMARK>": "?", " <EXCLAMATIONPOINT>": "!"} # note the space in front of each punc name
    gar2sym = {"<SIL>": "#%#", "<MUSIC>": "##%", "<NOISE>": "%%#", "<OTHER>":"%#%"} # so that they are savely keep as the original sym when using tokenize_text
    punc2sym.update(gar2sym)

    word2sym = { "h æ ʃ h ɐ ʃ p ɚ s ɛ n t": "<MUSIC>", "h æ ʃ p ɚ s ɛ n t h æ ʃ": "<SIL>", "p ɚ s ɛ n t h ɐ ʃ p ɚ s ɛ n t": "<OTHER>", "p ɚ s ɛ n t p ɚ s ɛ n t h æ ʃ": "<NOISE>"}
    forbidden_words = set(['#%#', '##%', '%%#', '%#%'])

    stime = time.time()
    logging.info("loading the dataset...")

    splits = ['validation', 'test', 'train']
    
    logging.info(f"phonemizing...")
    # you will see a ton of [WARNING] words_mismatch.py:88......, it's not a issue
    for split in tqdm.tqdm(splits):
        skip = 0
        logging.info(f"now processing split {split}...")
        jsondata = pd.read_json(path_or_buf=os.path.join(args.dataset_dir, 'trans', split+'.json'), lines=True)
        N=88
        df_split = np.array_split(jsondata, N)
        print(len(jsondata))
        # Optional: Save each part to a separate JSON file
        cmds = []
        for idx, part in enumerate(df_split):
            # if idx >80 and idx <=100:
            part.reset_index(drop=True, inplace=True)
            cmds.append((idx, part))

        def process_one(indx, splitdata):
            vocab_fn = os.path.join(args.save_dir, args.dataset_name, f"vocab_{split}_{str(indx)}.txt")
            phn_vocab = set()
            all_lens = []
            for key in tqdm.tqdm(range(len(splitdata))):
                save_fn = os.path.join(phn_save_root, splitdata['segment_id'][key]+".txt")
                if not os.path.exists(save_fn):
                    text = splitdata['trans'][key]
                    if sum(word in forbidden_words for word in text.split(" ")):
                        logging.info(f"skip {splitdata['segment_id'][key]}, because it contains forbiden words. It's transcript: {text}")
                        skip += 1
                        continue
                    for k, v in punc2sym.items():
                        text = text.replace(k, v)
                    phn = tokenize_text(text_tokenizer, text)
                    phn_seq = " ".join(phn)
                    for k, v in word2sym.items():
                        phn_seq = phn_seq.replace(k, v)
                    phn_vocab.update(phn_seq.split(" "))
                    all_lens.append(len(phn_seq.split(" ")))
                    with open(save_fn, "w") as f:
                        f.write(phn_seq)
                else:
                    print('exists')
                    with open(save_fn, "r") as f:
                        phn_seq = f.read()
                    phn_vocab.update(phn_seq.split(" "))
                    all_lens.append(len(phn_seq.split(" ")))

        with multiprocessing.Pool(processes=88) as pool:
            pool.starmap(process_one, cmds)
