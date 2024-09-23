# @ hwang258@jh.edu

import argparse
def parse_args():
    parser = argparse.ArgumentParser(description="phonemize the dataset using espeak-ng")
    parser.add_argument("--dataset_name", type=str, default='English', help='name of dataset')
    parser.add_argument("--json_path", type=str, default=None)
    parser.add_argument('--save_dir', type=str, default=None, help="path to the manifest, phonemes, and encodec codes dirs")
    return parser.parse_args()

if __name__ == "__main__":
    import logging
    formatter = (
        "%(asctime)s [%(levelname)s] %(filename)s:%(lineno)d || %(message)s"
    )
    logging.basicConfig(format=formatter, level=logging.INFO)
    args = parse_args()
    import json
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
    

    ### phonemization
    # load tokenizer
    text_tokenizer = TextTokenizer(backend="espeak") # add language='cmn' when you process mandarin
    punc2sym = {" <COMMA>": ",", " <PERIOD>": ".", " <QUESTIONMARK>": "?", " <EXCLAMATIONPOINT>": "!"} # note the space in front of each punc name
    gar2sym = {"<SIL>": "#%#", "<MUSIC>": "##%", "<NOISE>": "%%#", "<OTHER>":"%#%"} # so that they are savely keep as the original sym when using tokenize_text
    punc2sym.update(gar2sym)

    word2sym = { "h æ ʃ h ɐ ʃ p ɚ s ɛ n t": "<MUSIC>", "h æ ʃ p ɚ s ɛ n t h æ ʃ": "<SIL>", "p ɚ s ɛ n t h ɐ ʃ p ɚ s ɛ n t": "<OTHER>", "p ɚ s ɛ n t p ɚ s ɛ n t h æ ʃ": "<NOISE>"}
    forbidden_words = set(['#%#', '##%', '%%#', '%#%'])

    stime = time.time()
    
    logging.info(f"phonemizing...")
    # you will see a ton of [WARNING] words_mismatch.py:88......, it's not a issue

    with open(args.json_path, 'r') as json_file:
        jsondata = json.load(json_file)
        
    N=88 # Set to your CPU cores
    df_split = np.array_split(jsondata, N)
    print(len(jsondata))
    
    cmds = []
    for idx, part in enumerate(df_split):
        cmds.append((idx, part))

    def process_one(indx, splitdata):
        for key in tqdm.tqdm(range(len(splitdata))):
            save_fn = os.path.join(phn_save_root, splitdata[key]['segment_id']+".txt")
            if not os.path.exists(save_fn):
                text = splitdata[key]['trans']
                if sum(word in forbidden_words for word in text.split(" ")):
                    logging.info(f"skip {splitdata[key]['segment_id']}, because it contains forbiden words. It's transcript: {text}")
                    continue
                for k, v in punc2sym.items():
                    text = text.replace(k, v)
                phn = tokenize_text(text_tokenizer, text)
                phn_seq = " ".join(phn)
                for k, v in word2sym.items():
                    phn_seq = phn_seq.replace(k, v)
                with open(save_fn, "w") as f:
                    f.write(phn_seq)

    with multiprocessing.Pool(processes=88) as pool:
        pool.starmap(process_one, cmds)
