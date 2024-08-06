# @ hwang258@jh.edu

from pathlib import Path
import torch
import pickle
import argparse
import logging
import torch.distributed as dist
from config import MyParser
from steps import trainer
import time
import os

if __name__ == "__main__":
    formatter = (
        "%(asctime)s [%(levelname)s] %(filename)s:%(lineno)d || %(message)s"
    )
    logging.basicConfig(format=formatter, level=logging.INFO)
    
    torch.cuda.empty_cache()
    args = MyParser().parse_args()
    logging.info(args)
    exp_dir = Path(args.exp_dir)
    exp_dir.mkdir(exist_ok=True, parents=True)
    logging.info(f"exp_dir: {str(exp_dir)}")

    if args.resume:
        try:
            resume = args.resume
            assert(bool(args.exp_dir))
            with open("%s/args.pkl" % args.exp_dir, "rb") as f:
                old_args = pickle.load(f)
            new_args = vars(args)
            old_args = vars(old_args)
            for key in new_args:
                if key not in old_args or old_args[key] != new_args[key]:
                    old_args[key] = new_args[key]
            args = argparse.Namespace(**old_args)
            args.resume = resume
        except:
            with open("%s/args.pkl" % args.exp_dir, "wb") as f:
                pickle.dump(args, f)
    else:
        with open("%s/args.pkl" % args.exp_dir, "wb") as f:
            pickle.dump(args, f)

    dist.init_process_group(backend='nccl', init_method='env://')
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    print(rank, world_size, torch.cuda.device_count())
    local_rank = rank % torch.cuda.device_count()  # Ensure local rank is within the number of GPUs available on the node
    # local_rank = rank
    torch.cuda.set_device(local_rank)
    
    my_trainer = trainer.Trainer(args, world_size, rank)
    my_trainer.train()