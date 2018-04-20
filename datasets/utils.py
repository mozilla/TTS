import torch
import numpy as np


use_cuda = torch.cuda.is_available()

class TBPTT():
    
    def __init__(self, text, mel_spec, linear_spec, lengths, tbp_len):
        self.tbp_len = tbp_len
        self.lengths = lengths
        self.text = text
        self.mel_spec = mel_spec
        self.linear_spec = linear_spec
        self.start = True
        self._setup_splits()
        self.num_iters = len(self.mel_spec)
        self._setup_lengths()
        self.mel_spec.reverse()
        self.linear_spec.reverse()
        self.lengths.reverse()
    
    def _setup_splits(self):
        self.mel_spec = list(torch.split(self.mel_spec, self.tbp_len, 1))
        self.linear_spec = list(torch.split(self.linear_spec, self.tbp_len, 1))
        assert len(self.mel_spec) == len(self.linear_spec)

    def _setup_lengths(self):
        bucket = []
        for i, l in enumerate(self.lengths):
            l = int(l.data[0])
            tbp_lens = [self.tbp_len] * int(np.floor(l / self.tbp_len))
            if l % self.tbp_len != 0:
                tbp_lens.append(l % self.tbp_len)
            diff = int(self.num_iters - len(tbp_lens))
            assert diff >= 0
            tbp_lens += [0] * diff
            if use_cuda:
                bucket.append(torch.LongTensor(tbp_lens).cuda())
            else:
                bucket.append(torch.LongTensor(tbp_lens))
        self.lengths = torch.autograd.Variable(torch.stack(bucket))
        self.lengths = list(torch.split(self.lengths, 1, 1))
        self.lengths = [l.squeeze() for l in self.lengths]
        
    def __next__(self):
        if len(self.mel_spec) == 0:
            raise StopIteration()
        if self.num_iters > len(self.mel_spec):
            self.start = False
        return self.text, self.mel_spec.pop(), self.linear_spec.pop(), self.lengths.pop()
    
    next = __next__

    def __iter__(self):
        return self

    def __len__(self):
        return self.num_iters
            
                