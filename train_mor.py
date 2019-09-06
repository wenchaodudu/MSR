import torch
import torch.nn as nn

import sys
import json
import pdb
import time
import pickle
import numpy as np
import argparse
from morph_data_loader import get_loader
from model import *


def main(config):
    print(config)

    dictionary = json.load(open('./{}.{}.mor.dictionary.json'.format(config.track, config.lang)))
    vocab_size = len(dictionary['char']) + 1
    print("Vocabulary size:", vocab_size)

    batch_size = config.batch_size
    if config.lang == 'es':
        batch_size = 256
    train_loader = get_loader('./data/{}.{}.train.mor'.format(config.track, config.lang), 
                              dictionary, config.lang, batch_size)
    split = 'dev'
    if config.test:
        split = 'test'
    if config.gen:
        if config.test:
            orders = open('T1.{}.{}.ord'.format(config.lang, split)).readlines()
        else:
            orders = None
        dev_loader = get_loader('./data/{}.{}.{}.mor'.format(config.track, config.lang, split), 
                                dictionary, config.lang, 1, shuffle=False, order=orders)
    else:
        dev_loader = get_loader('./data/{}.{}.{}.mor'.format(config.track, config.lang, split), 
                                dictionary, config.lang, batch_size, seen=train_loader.dataset.seen)
    hidden_size = 128
    word_embedding_dim = 128
    
    limit = np.sqrt(6 / word_embedding_dim)
    word_vectors = np.random.uniform(low=-limit, high=limit, size=(vocab_size, word_embedding_dim))

    model_name = 'char_mor.{}.pt'.format(config.lang)
    if not config.use_saved and not config.gen:
        hred = AttnDecoder(word_embedding_dim, hidden_size, dictionary, 0.5).cuda()
        for p in hred.parameters():
            torch.nn.init.uniform_(p.data, a=-0.1, b=0.1)
            #torch.nn.init.normal_(p.data, mean=0, std=0.005)
            '''
            if len(p.size()) > 1:
                torch.nn.init.xavier_uniform_(p.data)
            '''
    else:
        hred = torch.load(model_name).cuda()
        hred.flatten_parameters()
    
    params = list(filter(lambda x: x.requires_grad, list(hred.parameters())))
    optimizer = torch.optim.Adam(params, lr=0.001, betas=(0.9, 0.98), eps=1e-09)
    #optimizer = torch.optim.SGD(params, lr=0.001, momentum=0.99)

    id2word = dict()
    for k, v in dictionary['char'].items():
        id2word[v] = k 

    best_loss = np.inf
    power = 0
    for it in range(0, 15):
        ave_loss = 0
        ave_class_loss = 0
        kl_loss = 0
        last_time = time.time()
        #params = filter(lambda x: x.requires_grad, hred.parameters())
        #optimizer = torch.optim.SGD(params, lr=.01 * 0.85 ** it, momentum=0.99)
        hred.train()
        if not config.gen:
            for _, (src_seqs, src_lengths, trg_seqs, trg_lengths, pos_seqs, form_seqs, key_seqs, bert_seqs) in enumerate(train_loader):
                if _ % config.print_every_n_batches == 1:
                    print(ave_loss / min(_, config.print_every_n_batches), 
                          time.time() - last_time)
                    ave_loss = 0
                    ave_class_loss = 0
                    kl_loss = 0
                loss = hred.loss(src_seqs, src_lengths, trg_seqs, trg_lengths, pos_seqs, form_seqs, bert_seqs)
                ave_loss += loss.item()
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, 1)
                optimizer.step()

        # eval on dev
        dev_loss = 0
        dev_nn_loss = 0
        count = 0.0
        nn_total_count = 0
        total_f1 = 0
        correct = 0
        if config.gen:
            output = open('T1.{}.{}.morph.txt'.format(config.lang, split), 'w')
            #new_output = open('T1.{}.morph.txt.new'.format(config.lang), 'w')
            error = open('T1.{}.{}.morph.err'.format(config.lang, split), 'w')
        hred.eval()
        for i, (src_seqs, src_lengths, trg_seqs, trg_lengths, pos_seqs, form_seqs, key_seqs, bert_seqs) in enumerate(dev_loader):
            with torch.no_grad():
                if config.baseline or config.mos:
                    loss = hred.loss(src_seqs, src_lengths, trg_seqs, trg_lengths, pos_seqs, form_seqs, bert_seqs)
                    if config.gen:
                        if i % 100 == 0:
                            print(i, correct)
                        key = key_seqs[0]
                        src = src_seqs[0].tolist()
                        src = ''.join([id2word[c] for c in src])
                        truth = trg_seqs[0].tolist()[1:-1]
                        truth = ''.join([id2word[c] for c in truth])
                        if key in train_loader.dataset.seen:
                            gen = train_loader.dataset.seen[key]
                        else:
                            gen, _ = hred.generate(src_seqs, src_lengths, pos_seqs, form_seqs, bert_seqs, 50, 5, 5)
                            gen = gen[0].tolist()
                            if 2 in gen:
                                end = gen.index(2)
                                gen = gen[:end]
                            gen = ''.join([id2word[c] for c in gen])
                        output.write('{} {} {}\n'.format(src, gen, truth))
                        if gen.lower() == truth.lower():
                            correct += 1
                        else:
                            error.write('{} {} {} {} {}\n'.format(i, src, key_seqs, gen, truth))

                    dev_loss += loss.item()

            count += 1
        if dev_loss < best_loss:
            best_loss = dev_loss
            torch.save(hred, model_name)
        print('dev loss: {}\n'.format(dev_loss / count))
        print('corret: {}\n'.format(correct / count))
        if config.gen:
            break



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--vhred', type=bool, default=False)
    parser.add_argument('--use_saved', action='store_true', default=False)
    parser.add_argument('--print_every_n_batches', type=int, default=100)
    parser.add_argument('--kl_weight', type=bool, default=True)
    parser.add_argument('--lr', type=float, default=0.25)
    parser.add_argument('--lang', type=str, default='en')
    parser.add_argument('--order', type=str, default='seq')
    parser.add_argument('--track', type=str, default='T1')
    parser.add_argument('--start_kl_weight', type=float, default=0)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--num_mixture', type=int, default=3)
    parser.add_argument('--glove', action='store_true', default=False)
    parser.add_argument('--bert', action='store_true', default=False)
    parser.add_argument('--baseline', action='store_true', default=True)
    parser.add_argument('--gen', action='store_true', default=False)
    parser.add_argument('--test', action='store_true', default=False)
    config = parser.parse_args()
    main(config)
