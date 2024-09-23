# @ hwang258@jh.edu

import os
import glob
from tqdm import tqdm
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="generate the vocabulary set")
    parser.add_argument('--save_dir', type=str, default=None, help="path to the manifest, phonemes, and encodec codes dirs")
    parser.add_argument("--dataset_name", type=str, default='English', help='name tag of dataset')
    return parser.parse_args()
    
if __name__ == "__main__":
    args = parse_args()
    files = glob.glob(os.path.join(args.save_dir, args.dataset_name, 'phonemes', '*.txt'))
    savepath = os.path.join(args.save_dir, args.dataset_name, 'vocab.txt')
    phn_vocab = []
    
    for f in tqdm(files):
        with open(f, 'r') as fi:
            data = fi.readlines()
        for x in data:
            x = x.split("\n")[0] if "\n" in x else x
            phn_vocab.append(x.split(" "))
    phn_vocab = set(phn_vocab)
    print(len(phn_vocab))
    with open(savepath, "w") as f:
        for i, phn in enumerate(list(phn_vocab)):
            if i < len(list(phn_vocab)) - 1:
                f.write(f"{str(i)} {phn}\n")
            else:
                f.write(f"{str(i)} {phn}")
