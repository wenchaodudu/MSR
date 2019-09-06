import pdb
import sys
import os
import torch
import numpy as np
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM



lang = sys.argv[1]
if lang == 'en':
    tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
    model = BertModel.from_pretrained('bert-base-cased')
else:
    tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
    model = BertModel.from_pretrained('bert-base-multilingual-cased')
model.eval()
TRAIN = int(sys.argv[2])

data = []
if TRAIN == 1:
    src_dir = 'T1-train'
    for src in os.listdir(src_dir):
        if src[:2] == lang:
            datapoint = []
            for l, line in enumerate(open(os.path.join(src_dir, src)).readlines()):
                if l % 1000 == 0:
                    print(l)
                line = line.split('\t')
                if len(line) <= 1:
                    datapoint = sorted(datapoint, key=lambda x: x[1])
                    words = ['[CLS]'] +  [t[0] for t in datapoint] + ['[SEP]']
                    try:
                        tokens = tokenizer.tokenize(' '.join(words))
                        seg_ids = [1] * len(tokens)
                        word_ids = tokenizer.convert_tokens_to_ids(tokens)
                        word_tensor = torch.tensor([word_ids])
                        seg_tensor = torch.tensor([seg_ids])
                        encoded_layers, _ = model(word_tensor, seg_tensor)
                        if len(tokens) != len(words):
                            _data = []
                            pointer = 1
                            for x in range(1, len(words) - 1):
                                start = pointer
                                t = len(words[x])
                                c = 0
                                #while tokens[pointer+1][:2] == '##':
                                while c < t:
                                    if tokens[pointer+1][:2] == '##':
                                        c += len(tokens[pointer]) - 1
                                    else:
                                        c += len(tokens[pointer])
                                    pointer += 1
                                vec = encoded_layers[-1][0][start:pointer+1].sum(0) / (pointer - start + 1)
                                _data.append(vec.unsqueeze(0))
                            data.append(torch.cat(_data, dim=0).data.numpy())
                            assert len(words) - 2 == data[-1].shape[0]
                        else:
                            data.append(encoded_layers[-1][0][1:-1].data.numpy())
                    except:
                        data.append(np.zeros((len(words) - 2, 768)))
                    datapoint = []
                else:
                    orig_id = int(line[5].split('=')[-1])
                    datapoint.append((line[1], orig_id))
    data = np.vstack(data)
    np.save('data/T1.{}.train.mor'.format(lang), data)
elif TRAIN == 2:
    src_dir = 'UD-dev'
    for src in os.listdir(src_dir):
        if src[:2] == lang:
            datapoint = []
            for l, line in enumerate(open(os.path.join(src_dir, src)).readlines()):
                if l % 1000 == 0:
                    print(l)
                if line[0] == '#':
                    continue
                line = line.split('\t')
                if '-' in line[0] or '.' in line[0]:
                    print(line[0])
                    continue
                if len(line) <= 1:
                    words = ['[CLS]'] + datapoint + ['[SEP]']
                    tokens = tokenizer.tokenize(' '.join(words))
                    seg_ids = [1] * len(tokens)
                    word_ids = tokenizer.convert_tokens_to_ids(tokens)
                    word_tensor = torch.tensor([word_ids])
                    seg_tensor = torch.tensor([seg_ids])
                    encoded_layers, _ = model(word_tensor, seg_tensor)
                    if len(tokens) != len(words):
                        pointer = 1
                        _data = []
                        try:
                            for x in range(1, len(words) - 1):
                                start = pointer
                                t = len(words[x])
                                c = 0
                                #while tokens[pointer+1][:2] == '##':
                                while c < t:
                                    if tokens[pointer+1][:2] == '##':
                                        c += len(tokens[pointer]) - 1
                                    else:
                                        c += len(tokens[pointer])
                                    pointer += 1
                                vec = encoded_layers[-1][0][start:pointer+1].sum(0) / (pointer - start + 1)
                                _data.append(vec.unsqueeze(0))
                            data.append(torch.cat(_data, dim=0).data.numpy())
                        except:
                            data.append(np.zeros((len(words) - 2, 768)))
                        assert len(words) - 2 == data[-1].shape[0]
                    else:
                        data.append(encoded_layers[-1][0][1:-1].data.numpy())
                    datapoint = []
                else:
                    datapoint.append(line[1])
    data = np.vstack(data)
    np.save('data/T1.{}.dev.mor'.format(lang), data)
else:
    src = open('T1.{}.test.txt'.format(lang)).readlines()
    line_ind = 2
    while line_ind < len(src):
        if line_ind // 4 % 1000 == 0:
            print(line_ind)
        line = src[line_ind]
        words = ['[CLS]'] + line.split() + ['[SEP]']
        tokens = tokenizer.tokenize(' '.join(words))
        seg_ids = [1] * len(tokens)
        word_ids = tokenizer.convert_tokens_to_ids(tokens)
        word_tensor = torch.tensor([word_ids])
        seg_tensor = torch.tensor([seg_ids])
        encoded_layers, _ = model(word_tensor, seg_tensor)
        if len(tokens) != len(words):
            pointer = 1
            _data = []
            for x in range(1, len(words) - 1):
                start = pointer
                t = len(words[x])
                c = 0
                #while tokens[pointer+1][:2] == '##':
                while c < t:
                    if tokens[pointer+1][:2] == '##':
                        c += len(tokens[pointer]) - 1
                    else:
                        c += len(tokens[pointer])
                    pointer += 1
                vec = encoded_layers[-1][0][start:pointer+1].sum(0) / (pointer - start + 1)
                _data.append(vec.unsqueeze(0))
            data.append(torch.cat(_data, dim=0).data.numpy())
            assert len(words) - 2 == data[-1].shape[0]
        else:
            data.append(encoded_layers[-1][0][1:-1].data.numpy())
        line_ind += 4
    data = np.vstack(data)
    np.save('data/T1.{}.test.mor'.format(lang), data)
