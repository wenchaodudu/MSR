import torch
import torch.nn as nn

import sys
import json
import pdb
import time
import pickle
import numpy as np
import argparse
from data_loader import get_loader
from model import *
#from bert_embedding import BertEmbedding
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM


def main(config):
    print(config)

    dictionary = json.load(open('./{}.{}.dictionary.json'.format(config.track, config.lang)))
    vocab_size = len(dictionary['word']) + 1
    print("Vocabulary size:", vocab_size)

    batch_size = config.batch_size
    train_loader = get_loader('./data/{}.{}.train.src'.format(config.track, config.lang), 
                              './data/{}.{}.train.trg'.format(config.track, config.lang), 
                              dictionary, config.lang, config.order, config.track, batch_size)
    dev_loader = get_loader('./data/{}.{}.dev.src'.format(config.track, config.lang), 
                            './data/{}.{}.dev.trg'.format(config.track, config.lang), 
                            dictionary, config.lang, config.order, config.track, batch_size)
    hidden_size = 512
    if config.bert:
        word_embedding_dim = 768
    else:
        word_embedding_dim = 300
    
    limit = np.sqrt(6 / word_embedding_dim)
    word_vectors = np.random.uniform(low=-limit, high=limit, size=(vocab_size, word_embedding_dim))

    if config.track == 'T2':
        config.print_every_n_batches = 100
    if config.baseline:
        model_name = 'transformer.{}.{}.{}.pt'.format(config.lang, config.order, config.track)
    elif config.mos:
        model_name = 'mos.{}.{}.pt'.format(config.data, config.num_mixture)
    elif config.vae:
        model_name = 'vae.{}.pt'.format(config.data)
    else:
        model_name = 'mixvae.vad.semi.{}.{}.pt'.format(config.data, config.num_mixture)
    if not config.use_saved:
        if config.baseline:
            if config.track == 'T2':
                hred = T2_Transformer(word_embedding_dim, 512, 1024, word_vectors, dictionary, 0.5, 5, 8).cuda()
            elif config.track == 'T5':
                hred = T6_Transformer(word_embedding_dim, 512, 1024, word_vectors, dictionary, 0.5, 5, 8).cuda()
            else:
                hred = Transformer(word_embedding_dim, 512, 1024, word_vectors, dictionary, 0.5, 5, 32).cuda()
        for p in hred.parameters():
            torch.nn.init.uniform_(p.data, a=-0.1, b=0.1)
            #torch.nn.init.normal_(p.data, mean=0, std=0.005)
            '''
            if len(p.size()) > 1:
                torch.nn.init.xavier_uniform_(p.data)
            '''
        if config.glove:
            print("Loading word vectors.")
            '''
            word2vec_file = open('/projects/dataset_processed/wenchaod/glove.840B.300d.txt')
            next(word2vec_file)
            found = 0
            for _, line in enumerate(word2vec_file):
                if _ % 100000 == 0:
                    print(_)
                word, vec = line.split(' ', 1)
                if word in dictionary['word']:
                    word_vectors[dictionary['word'][word]] = np.fromstring(vec, dtype=np.float32, sep=' ')
                    found += 1
            print(found)
            np.save('glove.840B.300d', word_vectors)
            '''
            word_vectors = np.load('glove.840B.300d.npy')
            hred.word_embed.from_pretrained(torch.from_numpy(word_vectors.astype(np.float32)), freeze=False)
        elif config.bert:
            '''
            if config.lang == 'en':
                tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
                bert_model = BertModel.from_pretrained('bert-base-cased')
            elif config.lang == 'zh':
                tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
                bert_model = BertModel.from_pretrained('bert-base-chinese')
            else:
                tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
                bert_model = BertModel.from_pretrained('bert-base-multilingual-cased')
            bert_model.eval()
            count = 0
            for word, ind in dictionary['word'].items():
                if count % 1000 == 0:
                    print(count)
                tokens = tokenizer.tokenize('[CLS] ' + word + ' [SEP]')
                seg_ids = [1] * len(tokens)
                word_ids = tokenizer.convert_tokens_to_ids(tokens)
                word_tensor = torch.tensor([word_ids])
                seg_tensor = torch.tensor([seg_ids])
                encoded_layers, _ = bert_model(word_tensor, seg_tensor)
                word_vectors[ind] = encoded_layers[-1][0][1:-1].mean(0).data.numpy()
                count += 1
            np.save('bert.{}'.format(config.lang), word_vectors)
            '''
            word_vectors = np.load('bert.{}.npy'.format(config.lang))
            hred.word_embed.from_pretrained(torch.from_numpy(word_vectors.astype(np.float32)), freeze=False)
    else:
        hred = torch.load(model_name).cuda()
        hred.flatten_parameters()
    
    params = list(filter(lambda x: x.requires_grad, list(hred.parameters())))
    optimizer = torch.optim.Adam(params, lr=0.001, betas=(0.9, 0.98), eps=1e-09)
    #optimizer = torch.optim.SGD(params, lr=0.001, momentum=0.99)

    if config.track == 'T2':
        best_loss = 0
    else:
        best_loss = np.inf
    power = 0
    for it in range(0, 10):
        ave_loss = 0
        ave_class_loss = 0
        kl_loss = 0
        last_time = time.time()
        #params = filter(lambda x: x.requires_grad, hred.parameters())
        if config.track == 'T1':
            if it > 3:
                optimizer = torch.optim.Adam(params, lr=0.0001, betas=(0.9, 0.98), eps=1e-09)
        hred.train()
        for _, (src_seqs, src_lengths, trg_seqs, trg_lengths, graph) in enumerate(train_loader):
            if _ % config.print_every_n_batches == 1:
                if config.track == 'T5':
                    print(ave_loss / min(_, config.print_every_n_batches), 
                          ave_class_loss / min(_, config.print_every_n_batches),
                          time.time() - last_time)
                else:
                    print(ave_loss / min(_, config.print_every_n_batches), time.time() - last_time)
                ave_loss = 0
                ave_class_loss = 0
                kl_loss = 0
            if config.baseline or config.mos:
                if config.track == 'T2':
                    loss, f1 = hred.loss(src_seqs, src_lengths, trg_seqs, trg_lengths, graph)
                elif config.track == 'T5':
                    loss, class_loss = hred.loss(src_seqs, src_lengths, trg_seqs, trg_lengths, graph)
                    ave_class_loss += class_loss.item()
                else:
                    loss = hred.loss(src_seqs, src_lengths, trg_seqs, trg_lengths, graph)
            ave_loss += loss.item()
            optimizer.zero_grad()
            if config.track == 'T5':
                (loss + class_loss).backward()
            else:
                loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 5)
            optimizer.step()
            if not (config.baseline or config.mos) and config.num_mixture == 3:
                for itt in range(0):
                    #loss, KL = hred.loss(src_seqs, src_lengths, indices, trg_seqs, trg_lengths, psn_seqs, psn_lengths, label_seqs)
                    loss, KL, prior, mix_probs, posterior, class_loss = hred.loss(src_seqs, src_lengths, indices, trg_seqs, trg_lengths, psn_seqs, psn_lengths, label_seqs, it)
                    post_optimizer.zero_grad()
                    (loss + KL).backward()
                    post_optimizer.step()
                

        # eval on dev
        dev_loss = 0
        dev_nn_loss = 0
        count = 0
        nn_total_count = 0
        total_f1 = 0
        hred.eval()
        for i, (src_seqs, src_lengths, trg_seqs, trg_lengths, graph) in enumerate(dev_loader):
            with torch.no_grad():
                if config.baseline or config.mos:
                    if config.track == 'T2':
                        loss, f1 = hred.loss(src_seqs, src_lengths, trg_seqs, trg_lengths, graph)
                        total_f1 += f1
                    elif config.track == 'T5':
                        loss, class_loss = hred.loss(src_seqs, src_lengths, trg_seqs, trg_lengths, graph)
                        loss = loss + class_loss
                    else:
                        loss = hred.loss(src_seqs, src_lengths, trg_seqs, trg_lengths, graph)
                    dev_loss += loss.item()
                
            count += 1
        if config.track == 'T2':
            if total_f1 > best_loss:
                best_loss = total_f1
                torch.save(hred, model_name)
            print('dev loss: {} {}\n'.format(dev_loss / count, total_f1 / count))
        else:
            if dev_loss < best_loss:
                best_loss = dev_loss
                torch.save(hred, model_name)
            print('dev loss: {}\n'.format(dev_loss / count))


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
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_mixture', type=int, default=3)
    parser.add_argument('--glove', action='store_true', default=False)
    parser.add_argument('--bert', action='store_true', default=False)
    parser.add_argument('--baseline', action='store_true', default=True)
    config = parser.parse_args()
    main(config)
