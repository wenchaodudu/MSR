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
from anytree import Node, RenderTree, PreOrderIter, LevelOrderIter
from copy import copy


class Dataset(data.Dataset):
    """Custom data.Dataset compatible with data.DataLoader."""
    def __init__(self, src_path, trg_path, dictionary, lang, order, track):
        """Reads source and target sequences from txt files."""
        self.src_seqs = pickle.load(open(src_path, 'rb'))
        self.trg_seqs = pickle.load(open(trg_path, 'rb'))
        self.adj_seqs = []
        self.num_total_seqs = len(self.src_seqs)
        self.max_len = 30
        self.word2id = dictionary['word']
        self.pos2id = dictionary['pos']
        self.dep2id = dictionary['dep']
        if 'trg_word' in dictionary:
            self.trgword2id = dictionary['trg_word']
            self.trgpos2id = dictionary['trg_pos']
        self.pos2id[0] = 0
        self.dep2id[0] = 0
        self.lang = lang
        for x in range(self.num_total_seqs):
            if x % 10000 == 0:
                print(x)
            if order == 'seq':
                if track != 'T5':
                    src, trg, adj_lst = self.preprocess(self.src_seqs[x], order)
                    self.src_seqs[x] = src
                    self.trg_seqs[x] = trg
                    self.adj_seqs.append(adj_lst)
                else:
                    src, trg, adj_lst = self.t5_preprocess(self.src_seqs[x], self.trg_seqs[x], order)
                    self.src_seqs[x] = src
                    self.trg_seqs[x] = trg
                    self.adj_seqs.append(adj_lst)
            else:
                if track in ['T1', 'T3']:
                    if track == 'T3':
                    #if False:
                        ind = {}
                        i = 0
                        for _, data in enumerate(self.src_seqs[x]):
                            if data:
                                ind[_] = i
                                i += 1
                        src = [copy(s) for s in self.src_seqs[x] if s]
                        for _, data in enumerate(src):
                            data[3] = _
                            if data[4] != -1:
                                data[4] = ind[data[4]]
                        self.src_seqs[x] = src
                    src, trg, adj_lst = self.preprocess(self.src_seqs[x], order)
                    adj_lst += (self.trg_seqs[x],)
                    self.src_seqs[x] = src
                    self.trg_seqs[x] = trg
                    self.adj_seqs.append(adj_lst)
                elif track == 'T2':
                    src, trg, adj_lst = self.t2_preprocess(self.src_seqs[x], self.trg_seqs[x], order)
                    self.src_seqs[x] = src
                    self.trg_seqs[x] = trg
                    self.adj_seqs.append(adj_lst)
                elif track == 'T5':
                    src, trg, adj_lst = self.t5_preprocess(self.src_seqs[x], self.trg_seqs[x], order)
                    self.src_seqs[x] = src
                    self.trg_seqs[x] = trg
                    self.adj_seqs.append(adj_lst)

    def __getitem__(self, index):
        """Returns one data pair (source and target)."""
        src_seq = self.src_seqs[index]
        trg_seq = self.trg_seqs[index]
        adj_lst = self.adj_seqs[index]
        
        return src_seq, trg_seq, adj_lst

    def __len__(self):
        return self.num_total_seqs

    def t5_preprocess(self, sequence, trg, order):
        result = []
        for _, seq in enumerate(sequence):
            if seq[2] != 0:
                res = [self.word2id[seq[0]], self.pos2id[seq[2]], self.dep2id[seq[5]], seq[4], 0]
            else:
                res = [0, 0, 0, 0, 0]
            result.append(res)
        if order != 'seq':
            trg = []
            adj_lst = [[y for y in range(len(sequence)) if sequence[y][4] == x and sequence[y][2] != 0] for x in range(len(sequence))]
            root_ind = [x for x in range(len(sequence)) if sequence[x][4] == -1][0]
            graph = [Node(x) for x in range(len(sequence))]
            for x, seq in enumerate(sequence):
                if seq[4] != -1:
                    graph[x].parent = graph[seq[4]] 
            root = graph[root_ind]
            neigh = []
            depths = [node.depth + 1 for node in graph]

            func_mask = []
            par = []
            if order in ['depth', 'hybrid']:
                iteration = PreOrderIter(root)
            elif order == 'breadth':
                iteration = LevelOrderIter(root)
            if len(graph) == 1:
                #trg = [self.trgword2id['<eof>'], root_ind + len(self.trgword2id) + 1]
                trg = [root_ind + len(self.trgword2id) + 1, self.trgword2id['<eof>']]
                neigh = [[root_ind], []]
                #func_mask = [0, 1]
                func_mask = [1, 0]
            else:
                if order == 'hybrid':
                    for node in iteration:
                        children = [ch.name for ch in node.children]
                        if children:
                            lst = sorted(children + [node.name], key=lambda x: sequence[x][3])
                            last_cont = 0
                            neigh_lst = []
                            for x, i in enumerate(lst):
                                if sequence[i][2] == 0:
                                    #last_func = x + 1
                                    lst[x] = self.trgword2id[sequence[i][1].lower()]
                                else:
                                    lst[x] = i + len(self.trgword2id) + 1
                                    neigh_lst.append(i)
                                    last_cont = x
                            #lst.insert(last_func, self.trgword2id['<eof>'])
                            lst.append(self.trgword2id['<eof>'])
                            trg.extend(lst)
                            mask = [1] * (last_cont + 1) + [0] * (len(lst) - last_cont - 1)
                            par.extend([node.name] * len(lst))
                            #mask = [0] * len(lst)
                            func_mask.extend(mask)
                            for x, i in enumerate(lst):
                                neigh.append(neigh_lst)
                                if i > len(self.trgword2id):
                                    neigh_lst = neigh_lst[1:]
                elif order == 'depth':
                    def depth_first(node):
                        trg.append(node.name)
                        children = list(node.children)
                        prev.append(prev_ind)
                        if children:
                            lst = sorted(children + [Node(node.name)], key=lambda x: sequence[x.name][3])
                            for x, n in enumerate(lst):
                                neigh.append([m.name for m in lst[x:]])
                                depth_first(n)
                    neigh.append([root_ind])
                    depth_first(root)
            return result, trg, (adj_lst, root_ind, neigh, par, depths, func_mask)
        else:
            src = [[self.word2id[seq[0]], self.pos2id[seq[2]], self.dep2id[seq[5]]] for seq in sequence if seq[2]]
            trg = [self.trgword2id[seq[1].lower()] if seq[2] == 0 else self.word2id[seq[0]] for seq in sequence] + [1]
            label = [1 if seq[2] == 0 else 0 for seq in sequence] + [1]
            src_ind = [-1]
            for seq in sequence:
                if seq[2] == 0:
                    src_ind.append(src_ind[-1])
                else:
                    src_ind.append(src_ind[-1] + 1)
            return src, trg, (label, src_ind[1:])


    def t2_preprocess(self, sequence, trg, order):
        result = []
        trg_ind = []
        trg_seqs = []
        for _, seq in enumerate(sequence):
            if seq is None:
                res = [0, 0, 0, 0, 0]
            else:
                res = [self.word2id[seq[0]], self.pos2id[seq[2]], self.dep2id[seq[5]], seq[4], 0]
                if seq[4] > -1:
                    res[-1] = self.word2id[sequence[seq[4]][0]]
            result.append(res)
        for _, seq in enumerate(trg):
            res = [self.trgword2id[seq[0].lower()], self.trgpos2id[seq[2]], seq[4]]
            trg_ind.append(res)
            trg_seqs.append(0)

        adj_lst = [[y for y in range(len(sequence)) if sequence[y] and sequence[y][4] == x] for x in range(len(sequence))]
        root_ind = [x for x in range(len(sequence)) if sequence[x] and sequence[x][4] == -1][0]
        graph = [Node(x) for x in range(len(sequence))]
        for x, seq in enumerate(sequence):
            if seq and seq[4] != -1:
                graph[x].parent = graph[seq[4]] 
        depths = [node.depth + 1 if node else 0 for node in graph]

        return result, trg_seqs, (adj_lst, root_ind, trg_ind, depths)

    def preprocess(self, sequence, order):
        result = []
        for _, seq in enumerate(sequence):
            if seq is None:
                res = [0, 0, 0, 0, 0]
            elif seq[2] == '_' and seq[5] == '_':
                res = [self.word2id[seq[0]], 0, 0, seq[4], seq[6]]
            else:
                res = [self.word2id[seq[0]], self.pos2id[seq[2]], self.dep2id[seq[5]], seq[4], seq[6]]
            result.append(res)

        trg = []
        adj_lst = [[y for y in range(len(sequence)) if sequence[y][4] == x] for x in range(len(sequence))]
        root_ind = [x for x in range(len(sequence)) if sequence[x][4] == -1][0]
        graph = [Node(x) for x in range(len(sequence))]
        for x, seq in enumerate(sequence):
            if seq[4] != -1:
                graph[x].parent = graph[seq[4]] 
        root = graph[root_ind]
        neigh = []
        depths = [node.depth + 1 for node in graph]
        if order != 'seq':
            prev = []
            prev_ind = 0
            if order in ['depth', 'hybrid']:
                iteration = PreOrderIter(root)
            elif order == 'breadth':
                iteration = LevelOrderIter(root)
            if len(graph) == 1:
                trg = [[root_ind]]
                neigh = [[root_ind]]
            else:
                if order == 'hybrid':
                    for node in iteration:
                        children = [ch.name for ch in node.children]
                        prev.append(prev_ind)
                        if children:
                            lst = sorted(children + [node.name], key=lambda x: sequence[x][3])
                            trg.extend(lst)
                            for x in range(len(lst)):
                                neigh.append(lst[x:])
                        else:
                            prev_ind += 1
                elif order == 'depth':
                    def depth_first(node):
                        trg.append(node.name)
                        children = list(node.children)
                        prev.append(prev_ind)
                        if children:
                            lst = sorted(children + [Node(node.name)], key=lambda x: sequence[x.name][3])
                            for x, n in enumerate(lst):
                                neigh.append([m.name for m in lst[x:]])
                                depth_first(n)
                    neigh.append([root_ind])
                    depth_first(root)
            return result, trg, (adj_lst, root_ind, neigh, prev, depths)
        else:
            trg = [None for x in sequence]
            for x, seq in enumerate(sequence):
                trg[seq[3]] = x
            neigh = [trg]
            for x in range(len(trg)-1):
                neigh.append(neigh[-1][1:])
            return result, trg, (adj_lst, root_ind, neigh, [0] * len(sequence), depths)


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
    def merge(sequences, dim):
        lengths = [len(seq) for seq in sequences]
        if dim == 0:
            padded_seqs = torch.zeros(len(sequences), max(lengths)).long()
        else:
            padded_seqs = torch.zeros(len(sequences), max(lengths), dim).long()
        for i, seq in enumerate(sequences):
            end = lengths[i]
            if end:
                padded_seqs[i, :end] = torch.LongTensor(seq[:end])
        return padded_seqs, lengths


    src_seqs, trg_seqs, adj_seqs = zip(*data)
    max_len = max(len(src) for src in src_seqs)

    # merge sequences (from tuple of 1D tensor to 2D tensor)
    src_seqs, src_lengths = merge(src_seqs, len(src_seqs[0][0]))
    trg_seqs, trg_lengths = merge(trg_seqs, 0)

    return src_seqs, src_lengths, trg_seqs, trg_lengths, adj_seqs


def get_loader(src_path, trg_path, word2id, lang, order, track, batch_size, shuffle=True):
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
    dataset = Dataset(src_path, trg_path, word2id, lang, order, track)

    # data loader for custome dataset
    # this will return (src_seqs, src_lengths, trg_seqs, trg_lengths) for each iteration
    # please see collate_fn for details
    #if label_path is None:
    data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              collate_fn=collate_fn)

    return data_loader


