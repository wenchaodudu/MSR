import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import sys
import json
import pdb
import time
import numpy as np
import argparse
from data_loader import get_loader
from copy import copy

import sys
import pickle


def main(config):
    dictionary = json.load(open('./{}.{}.dictionary.json'.format(config.track, config.lang)))
    T1_dictionary = json.load(open('./T1.{}.dictionary.json'.format(config.lang)))
    vocab_size = len(dictionary) + 1
    word_embedding_dim = 300
    print("Vocabulary size:", vocab_size)

    split = 'test' if config.test else 'dev'
    if config.track == 'T3':
        test_loader = get_loader('./data/{}.{}.dev.st2.src'.format(config.track, config.lang), 
                                 './data/{}.{}.dev.trg'.format(config.track, config.lang), 
                                 dictionary, config.lang, config.order, config.track, 1, shuffle=False)
        #truth_test_loader = get_loader('./data/{}.{}.dev.src'.format(config.track, config.lang), 
        #                               './data/{}.{}.dev.trg'.format(config.track, config.lang), 
        #                               dictionary, config.lang, config.order, config.track, 1, shuffle=False)
    else:
        if config.test:
            test_loader = get_loader('./data/{}.{}.test.src'.format(config.track, config.lang), 
                                     './data/{}.{}.test.trg'.format(config.track, config.lang), 
                                     dictionary, config.lang, config.order, config.track, 1, shuffle=False)
        else:
            test_loader = get_loader('./data/{}.{}.dev.src'.format(config.track, config.lang), 
                                     './data/{}.{}.dev.trg'.format(config.track, config.lang), 
                                     dictionary, config.lang, config.order, config.track, 1, shuffle=False)

    if config.track == 'T3':
        attn_model = torch.load('transformer.{}.{}.T1.pt'.format(config.lang, config.order))
    else:
        attn_model = torch.load('transformer.{}.{}.{}.pt'.format(config.lang, config.order, config.track))
    attn_model.flatten_parameters()
    attn_model.eval()
    
    id2word = dict()
    for k, v in dictionary['word'].items():
        id2word[v] = k
    trgid2word = dict()
    if config.track != 'T1':
        for k, v in dictionary['trg_word'].items():
            trgid2word[v] = k

    def print_response(src, responses, output, dict_list):
        for x in range(src.size(0)):
            for y in range(src.size(1)):
                #output.write(id2word[src[x, y].item()].encode('utf8'))
                output.write(id2word[src[x, y].item()])
                output.write(' ')
            output.write('\n')
        for _, res in enumerate(responses):
            for x in range(res.size(0)):
                linebreak = False
                for y in range(res.size(1)):
                    word = dict_list[_][res[x, y].item()]
                    if word == '__eou__':
                        output.write('\n')
                        linebreak = True
                        break
                    if _ == len(responses) - 1:
                        if (word != 'REDUCE' and word[:3] != 'NT-' and word[:4] != 'RULE') or word.find('SLOT') > -1:
                            #output.write(word.encode('utf8'))
                            output.write(word)
                            output.write(' ')
                    else:
                        if word != 'REDUCE' and word[:3] != 'NT-' and word[:4] != 'RULE':
                            #output.write(word.encode('utf8'))
                            output.write(word)
                            output.write(' ')
                if not linebreak:
                    output.write('\n')

    def print_t5_response(src, responses, output, dict_list):
        for x in range(src.size(0)):
            for y in range(src.size(1)):
                if src[x, y].item() != 0:
                    #output.write(id2word[src[x, y].item()].encode('utf8'))
                    output.write(id2word[src[x, y].item()])
                    output.write(' ')
            output.write('\n')
        for _, res in enumerate(responses):
            for x, y in res:
                if y:
                    output.write(id2word[x])
                else:
                    output.write(trgid2word[x])
                output.write(' ')
            output.write('\n')
                
    dist = [0, 0, 0]
    lengths = [0, 0, 0]
    scores = [0, 0, 0]
    beam_size = 10
    top_k = 5
    if config.track == 'T5':
        beam_size = 15
        top_k = 10
    if config.test:
        output = open('{}.{}.test.txt'.format(config.track, config.lang), 'w')
        output_order = open('{}.{}.test.ord'.format(config.track, config.lang), 'w')
    else:
        output = open('{}.{}.txt'.format(config.track, config.lang), 'w')
        output_order = open('{}.{}.ord'.format(config.track, config.lang), 'w')
    if config.morph:
        morphed = open('T1.{}.{}.morph.txt'.format(config.lang, split)).readlines()

    if config.track == 'T2':
        orig_src = pickle.load(open('./data/T2.{}.dev.src'.format(config.lang), 'rb'))
        new_src = [None for x in orig_src]
    #if config.track == 'T3':
    #    truth_test_loader_iter = iter(truth_test_loader)
    total = 0
    for _, batch in enumerate(test_loader):
        print(_)
        src_seqs, src_lengths, trg_seqs, trg_lengths, adj_seqs = batch
        #if config.track == 'T3':
        #    true_src_seqs, __, __, __, __ = next(truth_test_loader_iter)
        
        with torch.no_grad():
            if config.track == 'T5':
                res = []
                _res = []
                for w in trg_seqs[0].tolist():
                    if w >= 400:
                        _res.append((src_seqs[0, w-400, 0].item(), False))
                    else:
                        _res.append((w, True))
                res.append(_res)
            else:
                res = [src_seqs[0, :, 0][trg_seqs[0]].unsqueeze(0)]
            if config.order == 'seq':
                if config.track == 'T5':
                    attn_responses, nll = attn_model.generate(src_seqs, src_lengths, beam_size, top_k, config.order, adj_seqs)
                else:
                    attn_responses, nll = attn_model.generate(src_seqs, src_lengths, beam_size, top_k, adj_seqs)
            else:
                if config.track == 'T2':
                    words = attn_model.generate(src_seqs, src_lengths, beam_size, top_k, config.order, adj_seqs[0])
                else:
                    attn_responses, nll, order = attn_model.tree_generate(src_seqs, src_lengths, beam_size, top_k, config.order, adj_seqs[0])
                    if config.track != 'T5':
                        output_order.write(' '.join([str(o) for o in order]))
                        output_order.write('\n')
            if config.track == 'T2':
                src = copy(orig_src[_])
                for word in words:
                    src.append([trgid2word[word[1]], '_', '_', len(src), word[0], '_', 0])
                new_src[_] = src
                print(len(orig_src[_]), len(src))
            else:
                if config.morph:
                    tokens = [token.split() for token in morphed[total:total+src_lengths[0]]]
                    src = [token.split()[0] for token in morphed[total:total+src_lengths[0]]]
                    truth = [token[2] if len(token) >= 3 else token[0] for token in tokens]
                    gen = [token.split()[1] for token in morphed[total:total+src_lengths[0]]]
                    for x in range(len(gen)):
                        if src[x].isupper():
                            gen[x] = gen[x].upper()
                        elif src[x][0].isupper():
                            gen[x] = gen[x].title()
                        else:
                            gen[x] = gen[x].lower()
                    gen = [gen[x] for x in order]
                    gen[0] = gen[0].title()
                    for x in range(len(gen)):
                        if x and gen[x-1] in ['.', '!', '?']:
                            gen[x] = gen[x].title()
                    output.write(' '.join(truth))
                    output.write('\n')
                    output.write(' '.join(src))
                    output.write('\n')
                    output.write(' '.join(gen))
                    output.write('\n')
                    output.write('\n')
                    total += src_lengths[0]
                else:
                    if config.track == 'T5':
                        res.append(attn_responses)
                    else:
                        res.append(attn_responses.unsqueeze(0))
        
        if not config.morph:
            if config.track == 'T2':
                pickle.dump(new_src, open('./data/T3.{}.dev.st2.src'.format(config.lang), 'wb'))
            else:
                if config.track == 'T5':
                    #print_response(true_src_seqs[:, :, 0], res, output, [id2word] * len(res))
                    print_t5_response(src_seqs[:, :, 0], res, output, [id2word] * len(res))
                else:
                    print_response(src_seqs[:, :, 0], res, output, [id2word] * len(res))
                output.write('\n')
                print()
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--vhred', type=bool, default=False)
    parser.add_argument('--use_saved', type=bool, default=False)
    parser.add_argument('--sample', type=bool, default=False)
    parser.add_argument('--print_every_n_batches', type=int, default=10)
    parser.add_argument('--path', type=str, default='')
    parser.add_argument('--output', type=str, default='result.txt')
    parser.add_argument('--lang', type=str, default='en')
    parser.add_argument('--order', type=str, default='seq')
    parser.add_argument('--track', type=str, default='T1')
    parser.add_argument('--iter', type=int, default=1)
    parser.add_argument('--num_mixture', type=int, default=3)
    parser.add_argument('--morph', default=False, action='store_true')
    parser.add_argument('--test', default=False, action='store_true')
    config = parser.parse_args()
    main(config)
