# @ hwang258@jh.edu
import os
import torch
import random
import copy
import logging
import shutil
import typing as tp

class dataset(torch.utils.data.Dataset):
    def __init__(self, args, split):
        super().__init__()
        self.args = args
        self.split = split
        assert self.split in ['train', 'validation', 'test']
        manifest_fn = os.path.join(self.args.dataset_dir, self.args.manifest_name, self.split+".txt")

        with open(manifest_fn, "r") as rf:
            data = [l.strip().split("\t") for l in rf.readlines()]
        lengths_list = [int(item[-1]) for item in data]
        self.data = []
        self.lengths_list = []
        for d, l in zip(data, lengths_list):
            if l >= self.args.encodec_sr*self.args.audio_min_length:
                if self.args.drop_long and l > self.args.encodec_sr*self.args.audio_max_length:
                    continue
                self.data.append(d)
                self.lengths_list.append(l)
        logging.info(f"number of data points for {self.split} split: {len(self.lengths_list)}")

        # phoneme vocabulary
        vocab_fn = os.path.join(self.args.dataset_dir,"vocab.txt")
        shutil.copy(vocab_fn, os.path.join(self.args.exp_dir, "vocab.txt"))
        with open(vocab_fn, "r") as f:
            temp = [l.strip().split(" ") for l in f.readlines() if len(l) != 0]
            self.phn2num = {item[1]:int(item[0]) for item in temp}
        
        self.symbol_set = set(["<SIL>", "<MUSIC>", "<NOISE>", "<OTHER>"])

    def __len__(self):
        return len(self.lengths_list)
    
    def _load_phn_enc(self, index):
        item = self.data[index]
        pf = os.path.join(self.args.dataset_dir, self.args.phn_folder_name, item[1]+".txt")
        ef = os.path.join(self.args.dataset_dir, self.args.encodec_folder_name, item[1]+".txt")
        try:
            with open(pf, "r") as p, open(ef, "r") as e:
                phns = [l.strip() for l in p.readlines()]
                assert len(phns) == 1, phns
                x = [self.phn2num[item] for item in phns[0].split(" ") if item not in self.symbol_set] # drop ["<SIL>", "<MUSIC>", "<NOISE>", "<OTHER>"], as they are not in training set annotation
                encos = [l.strip().split() for k, l in enumerate(e.readlines()) if k < self.args.n_codebooks]
                
                assert len(encos) == self.args.n_codebooks, ef
                if self.args.special_first:
                    y = [[int(n)+self.args.n_special for n in l] for l in encos]
                else:
                    y = [[int(n) for n in l] for l in encos]
        except Exception as e:
            logging.info(f"loading failed for {pf} and {ef}, maybe files don't exist or are corrupted")
            logging.info(f"error message: {e}")
            return [], [[]]

        return x, y

    def prepare_mask_intervals(self, y_len): 
        # random generate mask intervals
        # Mask Intervals: [(5, 9), (19, 29)]
        # Non-Mask Intervals: [[(0, 5), (9, 19), (29, 30)]]
    
        if self.args.mask_sample_dist == "uniform":
            n_spans = random.choice(range(1, self.args.max_n_spans + 1))
        elif "poisson" in self.args.mask_sample_dist.lower():
            param = float(self.args.mask_sample_dist[len("poisson"):])
            poisson_sample = torch.poisson(torch.tensor([param]))
            n_spans = int(poisson_sample.clamp(1, self.args.max_n_spans).item())
        
        starts = random.sample(range(0, y_len - self.args.mask_len_min), n_spans)
        starts = sorted(starts)
    
        for j in range(len(starts) - 1, 0, -1):
            if starts[j] - starts[j - 1] < self.args.min_gap:
                del starts[j]
        assert len(starts) > 0, f"there is no masked span left, y_len: {y_len}, sampled n_spans: {n_spans}"

        tmp_mask_len_max = int(self.args.max_mask_portion * y_len / len(starts))
        
        ends = []
        for j, start in enumerate(starts):
            if j < len(starts) - 1:
                mask_len = random.randint(self.args.mask_len_min, min(tmp_mask_len_max, starts[j+1]-starts[j]-self.args.min_gap+1))
            else:
                mask_len = random.randint(self.args.mask_len_min, min(tmp_mask_len_max, y_len-starts[j]))
            ends.append(start + mask_len)
    
        if self.args.tts_enhanced > 0 and random.random() < 0.5:
            starts[-1] = max(starts[-1], y_len - tmp_mask_len_max)
            ends[-1] = y_len
            
        mask_intervals = [(s, e) for s, e in zip(starts, ends)]
        non_mask_intervals = [(ns, ne) for ns, ne in zip([0] + ends, starts + [y_len])]
        
        return mask_intervals, non_mask_intervals

    
    def rearrange(self, y, non_mask_intervals, mask_intervals):
    
        assert self.args.eos > 0, f"eos={self.args.eos} should > 0"
        
        rearranged_y = []
        sos_tensor = torch.tensor([self.args.sos] * self.args.n_codebooks).unsqueeze(-1)
        eos_tensor = torch.tensor([self.args.eos] * self.args.n_codebooks).unsqueeze(-1)
        eog_tensor = torch.tensor([self.args.eog] * self.args.n_codebooks).unsqueeze(-1)
        for i, item in enumerate(non_mask_intervals):
            if i == 0:
                if item[0] == item[1]: # case: (0,0)
                    rearranged_y.append(sos_tensor)
                else:
                    rearranged_y.append(torch.cat([sos_tensor, y[:, item[0]: item[1]]], dim=-1))
            elif i == len(non_mask_intervals)-1:
                if item[0] == item[1]: # case: (N,N)
                    rearranged_y.append(eos_tensor)
                else:
                    rearranged_y.append(torch.cat([y[:, item[0]: item[1]], eos_tensor], dim=-1))
            else:
                rearranged_y.append(y[:, item[0]: item[1]])
                
        for i, item in enumerate(mask_intervals):
            rearranged_y.append(torch.cat([y[:, item[0]: item[1]], eog_tensor], dim=-1))

        return rearranged_y

    def get_pattern_sequence(self, tokens: torch.Tensor, n_q: int, special_token: int, delays: tp.Optional[tp.List[int]] = None, 
                         empty_initial: int = 0) -> torch.Tensor:
        """Generate a pattern sequence for delayed codebooks without batch dimension.
    
        Args:
            tokens (torch.Tensor): Input tensor of shape [K, T].
            n_q (int): Number of codebooks.
            delays (Optional[List[int]]): Delay for each codebook. Defaults to increasing delays.
            empty_initial (int): Number of initial empty steps. Defaults to 0.
            special_token (int): Special token used to fill non-pattern coordinates in the new sequence.
    
        Returns:
            torch.Tensor: Modified tokens based on the pattern.
        """
        K, T = tokens.shape
        assert K == n_q, "Number of codebooks (K) must match n_q"
        if delays is None:
            delays = list(range(n_q))
        max_delay = max(delays)
        pattern_length = T + max_delay + empty_initial
        pattern_tokens = torch.full((K, pattern_length), fill_value=special_token, dtype=tokens.dtype, device=tokens.device)
    
        for t in range(T):
            for q in range(n_q):
                delayed_t = t + delays[q] + empty_initial
                if delayed_t < pattern_length:
                    pattern_tokens[q, delayed_t] = tokens[q, t]
    
        return pattern_tokens

    def revert_pattern_sequence(self, pattern_tokens: torch.Tensor, n_q: int,
                            delays: tp.Optional[tp.List[int]] = None, special_token: int = -1) -> torch.Tensor:
        """Revert the pattern sequence back to the original multi-codebook sequence without batch dimension.
    
        Args:
            pattern_tokens (torch.Tensor): Pattern tensor of shape [K, S].
            n_q (int): Number of codebooks.
            delays (Optional[List[int]]): Delay for each codebook. Defaults to increasing delays.
            special_token (int): Special token used to fill non-pattern coordinates in the new sequence.
    
        Returns:
            torch.Tensor: Reverted tokens of shape [K, T].
        """
        K, S = pattern_tokens.shape
        assert K == n_q, "Number of codebooks (K) must match n_q"
        if delays is None:
            delays = list(range(n_q))
        T = S - max(delays)
        reverted_tokens = torch.full((K, T), fill_value=special_token, dtype=pattern_tokens.dtype, device=pattern_tokens.device)
    
        for t in range(T):
            for q in range(n_q):
                delayed_t = t + delays[q]
                if delayed_t < S:
                    reverted_tokens[q, t] = pattern_tokens[q, delayed_t]
    
        return reverted_tokens
        
    def shift(self, rearranged_y):
        
        shifted_y = [self.get_pattern_sequence(tokens=cur_y, n_q=self.args.n_codebooks, special_token=self.args.empty_token) for cur_y in rearranged_y] # the first item is values, later two are indexes and mask
        
        return shifted_y
    
    def insert_mask(self, shifted_y):

        num_masks = (len(shifted_y) - 1) // 2
        assert num_masks == (len(shifted_y) - 1) / 2, len(shifted_y)
        emb_inds = list(range(self.args.mts, self.args.mts+ self.args.max_n_spans))
        if self.args.shuffle_mask_embedding:
            random.shuffle(emb_inds)
        emb_inds_use = emb_inds[:num_masks]
        mask_value = emb_inds_use + emb_inds_use
        assert len(shifted_y) == len(mask_value) + 1, len(mask_value)
        
        inserted_y = []
        mask_position = [-1] * (self.args.max_n_spans*2)
        for j in range(len(shifted_y)-1):
            inserted_y.append(shifted_y[j])
            mask_position[j] = sum([item.shape[1] for item in inserted_y]) # each item is of shape [K S], so take shape[1]
            tmp = torch.tensor([mask_value[j]] * self.args.n_codebooks).unsqueeze(-1)
            inserted_y.append(tmp)

        inserted_y.append(shifted_y[-1])
        mask_position = [item for item in mask_position if item != -1]

        return inserted_y, mask_position
    
    def cat_y(self, inserted_y):

        cated_y = torch.cat(inserted_y, dim=1)
        assert cated_y.shape[0] == self.args.n_codebooks, cated_y.shape
        new_y_lens = cated_y.shape[1]
        
        return cated_y, new_y_lens


    def __getitem__(self, index):
        x, y = self._load_phn_enc(index)
        x_len, y_len = len(x), len(y[0])

        if x_len == 0 or y_len == 0: # load failure
            item = self.data[index]
            pf = os.path.join(self.args.dataset_dir, self.args.phn_folder_name, item[1]+".txt")
            logging.info(f"loading failed for {pf}, length is 0")
            return {
                "x": None,
                "x_len": None,
                "y": None,
                "y_len": None,
            }
        
        if y_len < self.args.encodec_sr * self.args.audio_min_length or x_len < self.args.text_min_length: # too short
            item = self.data[index]
            pf = os.path.join(self.args.dataset_dir, self.args.phn_folder_name, item[1]+".txt")
            logging.info(f"loading failed for {pf}, too short")
            return {
                "x": None,
                "x_len": None,
                "y": None,
                "y_len": None,
            }
        
        if self.args.drop_long:
            if x_len > self.args.text_max_length or y_len > self.args.encodec_sr * self.args.audio_max_length: # too long
                item = self.data[index]
                pf = os.path.join(self.args.dataset_dir, self.args.phn_folder_name, item[1]+".txt")
                logging.info(f"loading failed for {pf}, too long")
                return {
                    "x": None,
                    "x_len": None,
                    "y": None,
                    "y_len": None,
                }
                
        if self.args.cfg_enhanced and random.random() < 0.1: # We use the last unused token for cfg training
            x = torch.tensor([self.args.text_vocab_size-1], dtype=torch.long)
            x_len = len(x)

        mask_intervals, non_mask_intervals = self.prepare_mask_intervals(y_len)
        rearranged_y = self.rearrange(torch.LongTensor(y), non_mask_intervals, mask_intervals)
        shifted_y = self.shift(rearranged_y)
        inserted_y, mask_position = self.insert_mask(shifted_y)
        y, y_len = self.cat_y(inserted_y)
        x = torch.LongTensor(x)
        y = torch.LongTensor(y)

        if not (y < int(self.args.audio_vocab_size) + self.args.n_special + self.args.max_n_spans).all():
            item = self.data[index]
            pf = os.path.join(self.args.dataset_dir, self.args.phn_folder_name, item[1]+".txt")
            logging.info(f"loading failed for {pf}, index out of range")
            return {
                "x": None,
                "x_len": None,
                "y": None,
                "y_len": None,
            }
            
        return {
            "x": x,
            "x_len": x_len,
            "y": y,
            "y_len": y_len
        }

    def collate(self, batch):
        out = {key:[] for key in batch[0]}
        for item in batch:
            if item['x'] == None: # deal with load failure
                continue
            for key, val in item.items():
                out[key].append(val)
                
        res = {}
        if self.args.pad_x:
            res["x"] = torch.stack(out["x"], dim=0)
        else:
            res["x"] = torch.nn.utils.rnn.pad_sequence(out["x"], batch_first=True, padding_value=self.args.text_pad_token)
        res["x_lens"] = torch.LongTensor(out["x_len"])
        if self.args.dynamic_batching:
            if out['y'][0].ndim==2:
                res['y'] = torch.nn.utils.rnn.pad_sequence([item.transpose(1,0) for item in out['y']],padding_value=self.args.audio_pad_token)
                res['y'] = res['y'].permute(1,2,0) # T B K -> B K T
            else:
                assert out['y'][0].ndim==1, out['y'][0].shape
                res['y'] = torch.nn.utils.rnn.pad_sequence(out['y'], batch_first=True, padding_value=self.args.audio_pad_token)
        else:
            res['y'] = torch.stack(out['y'], dim=0)
        res["y_lens"] = torch.LongTensor(out["y_len"])
        return res



if __name__ == "__main__":
    # debug
    pass
