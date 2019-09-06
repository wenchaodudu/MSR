import nltk
import json
import torch
import pickle
import torch.utils.data as data
from torch.utils.data.sampler import Sampler
import pdb
import numpy as np
from string import punctuation
from random import shuffle
from copy import copy


class Dataset(data.Dataset):
    """Custom data.Dataset compatible with data.DataLoader."""
    def __init__(self, src_path, dictionary, lang, seen=None, order=None):
        """Reads source and target sequences from txt files."""
        data = pickle.load(open(src_path, 'rb'))
        self.src_seqs = []
        self.trg_seqs = []
        self.pos_seqs = []
        self.form_seqs = []
        self.char2id = dictionary['char']
        self.pos12id = dictionary['pos1']
        self.pos22id = dictionary['pos2']
        self.form2id = dictionary['form']
        self.lang = lang
        self.key_seqs = []
        unique = {}
        self.seen = {}
        self.bert_seqs = np.load(src_path + '.npy')
        for x in range(len(data)):
            if x % 10000 == 0:
                print(x)
            for word in data[x]:
                form = sorted([f for f in word[4].split('|') if f[:4] != 'lin=' and f[:8] != 'original'])
                q_key = ' '.join([word[0], word[2], word[3]] + form)
                #if seen and q_key in seen:
                #    continue
                #self.seen[q_key] = word[1]
                src, trg, pos, form = self.preprocess(word)
                key = ' '.join([str(i) for i in src + trg + pos + form])
                if 'dev' in src_path or key not in unique:
                    self.src_seqs.append(src)
                    self.trg_seqs.append(trg)
                    self.pos_seqs.append(pos)
                    self.form_seqs.append(form)
                    self.key_seqs.append(q_key)
                    #unique[key] = True
        if order is not None:
            offset = 0
            src_seqs = []
            trg_seqs = []
            pos_seqs = []
            form_seqs = []
            key_seqs = []
            for line in order:
                _order = [int(o) for o in line.split()]
                for o in _order:
                    src_seqs.append(self.src_seqs[offset+o])
                    trg_seqs.append(self.trg_seqs[offset+o])
                    pos_seqs.append(self.pos_seqs[offset+o])
                    form_seqs.append(self.form_seqs[offset+o])
                    key_seqs.append(self.key_seqs[offset+o])
                offset += len(_order)
            self.src_seqs = src_seqs
            self.trg_seqs = trg_seqs
            self.pos_seqs = pos_seqs
            self.form_seqs = form_seqs
            self.key_seqs = key_seqs
        self.num_total_seqs = len(self.src_seqs)
        print(self.num_total_seqs, len(self.bert_seqs))
        #assert self.num_total_seqs == len(self.bert_seqs)

    def __getitem__(self, index):
        """Returns one data pair (source and target)."""
        src_seq = self.src_seqs[index]
        trg_seq = self.trg_seqs[index]
        pos = self.pos_seqs[index]
        form = self.form_seqs[index]
        key = self.key_seqs[index]
        bert = self.bert_seqs[index]
        
        return src_seq, trg_seq, pos, form, key, bert

    def __len__(self):
        return self.num_total_seqs

    def preprocess(self, data):
        src = [self.char2id[c] for c in data[0]]
        trg = [1] + [self.char2id[c] for c in data[1]] + [2]
        pos = [self.pos12id[data[2]], self.pos22id[data[3]]]
        form = data[4].split('|')
        form = [self.form2id[f] for f in form if f in self.form2id]
        return src, trg, pos, form


def collate_fn(data):
    """Creates mini-batch tensors from the list of tuples (src_seq, trg_seq).

    We should build a custom collate_fn rather than using default collate_fn,
    because merging sequences (including padding) is not supported in default.
    Seqeuences are padded to the maximum length of mini-batch sequences (dynamic padding).

    Args:
        data: list of tuple (src_seq, trg_seq).
            - src_seq: torch tensor of shape (?); variable length.
            - trg_seq: torch tensor of shape (?); variable length.

    Returns:
        src_seqs: torch tensor of shape (batch_size, padded_length).
        src_lengths: list of length (batch_size); valid length for each padded source sequence.
        trg_seqs: torch tensor of shape (batch_size, padded_length).
        trg_lengths: list of length (batch_size); valid length for each padded target sequence.
    """
    def merge(sequences):
        lengths = [len(seq) for seq in sequences]
        padded_seqs = torch.zeros(len(sequences), max(lengths)).long()
        for i, seq in enumerate(sequences):
            end = lengths[i]
            if end:
                padded_seqs[i, :end] = torch.LongTensor(seq[:end])
        return padded_seqs, lengths
        
    data.sort(key=lambda x: len(x[0]), reverse=True)
    src_seqs, trg_seqs, pos_seqs, form_seqs, key_seqs, bert_seqs = zip(*data)

    # merge sequences (from tuple of 1D tensor to 2D tensor)
    src_seqs, src_lengths = merge(src_seqs)
    trg_seqs, trg_lengths = merge(trg_seqs)
    form_seqs, form_lengths = merge(form_seqs)

    return src_seqs, src_lengths, trg_seqs, trg_lengths, pos_seqs, (form_seqs, form_lengths), key_seqs, bert_seqs


def get_loader(src_path, word2id, lang, batch_size, seen=None, shuffle=True, order=None):
    """Returns data loader for custom dataset.

    Args:
        src_path: txt file path for source domain.
        trg_path: txt file path for target domain.
        src_word2id: word-to-id dictionary (source domain).
        trg_word2id: word-to-id dictionary (target domain).
        batch_size: mini-batch size.

    Returns:
        data_loader: data loader for custom dataset.
    """
    # build a custom dataset
    dataset = Dataset(src_path, word2id, lang, seen, order)

    # data loader for custome dataset
    # this will return (src_seqs, src_lengths, trg_seqs, trg_lengths) for each iteration
    # please see collate_fn for details
    #if label_path is None:
    data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              collate_fn=collate_fn)

    return data_loader


