# @ hwang258@jh.edu

import os
import argparse
from tqdm import tqdm
import pandas as pd
import json

def parse_args():
    parser = argparse.ArgumentParser(description="encode the librilight dataset using encodec model")
    parser.add_argument("--json_path", type=str, default=None)
    parser.add_argument("--dataset_name", type=str, default=None, help='name tag of dataset')
    parser.add_argument('--save_dir', type=str, default=None, help="path to the manifest, phonemes, and encodec codes dirs")
    parser.add_argument("--audio_min_length", type=float, default=1, help="in second, drop the audio if length is shorter than this")
    parser.add_argument("--encodec_sr", type=int, default=50, help="for my encodec that takes 16kHz audio with a downsample rate of 320, the codec sample rate is 50Hz, i.e. 50 codes (x n_codebooks) per second")
    parser.add_argument("--savename", type=str, default=None)

    return parser.parse_args()
    
if __name__ == "__main__":
    args = parse_args()
    phn_save_root = os.path.join(args.save_dir, args.dataset_name, "phonemes")
    codes_save_root = os.path.join(args.save_dir, args.dataset_name, "wmencodec")
    manifest_root = os.path.join(args.save_dir, args.dataset_name, "manifest")
    os.makedirs(manifest_root, exist_ok=True)

    savelines = []
    with open(args.json_path, 'r') as json_file:
        jsondata = json.load(json_file)
    for key in tqdm(range(len(jsondata))):
        if os.path.exists(os.path.join(phn_save_root, jsondata[key]['segment_id']+".txt")) and os.path.exists(os.path.join(codes_save_root, jsondata[key]['segment_id']+".txt")):
            with open(os.path.join(codes_save_root, jsondata[key]['segment_id']+".txt"), 'r') as fi:
                x = fi.readlines()
            if len(x[0].split(' ')) > int(args.audio_min_length * args.encodec_sr):
                savelines.append([jsondata[key]['segment_id'], len(x[0].split(' '))])

    outputlines = ''
    for i in range(len(savelines)):
        outputlines+='0\t'+ savelines[i][0]+'\t'+str(savelines[i][1])+'\n'
    with open(os.path.join(manifest_root, args.savename+'.txt'), "w") as f:
        f.write(outputlines)
