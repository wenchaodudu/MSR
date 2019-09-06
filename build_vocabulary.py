import pdb
import os
import sys
import pickle
import json


lang = 'es'

def update(dic, tok):
    if tok not in dic:
        dic[tok] = len(dic) + 1

word_dic = {}
pos_dic = {}
dep_dic = {}
trg_word_dic = {}
#trg_word_dic['<eof>'] = 1
trg_pos_dic = {}
TRACK = 'T2'

for src in os.listdir('data'):
    if TRACK == 'T1':
        if lang in src and src[-3:] == 'src' and src[:2] == TRACK:
        #if lang in src and src[-3:] == 'src' and src[:2] == 'T5':
        #if lang in src and src[:2] == TRACK:
            data = pickle.load(open(os.path.join('data', src), 'rb'))
            for _, datapoint in enumerate(data):
                if _ % 1000 == 0:
                    print(_)
                for word in datapoint:
                    if word is not None:
                        update(word_dic, word[0])
                        update(pos_dic, word[2])
                        update(dep_dic, word[5])
                        
    elif TRACK == 'T2':
        #if lang in src and src[-3:] == 'src' and src[:2] == TRACK:
        #if lang in src and src[-3:] == 'src' and src[:2] == 'T5':
        if lang in src and src[:2] == TRACK and 'st2' not in src:
            data = pickle.load(open(os.path.join('data', src), 'rb'))
            for _, datapoint in enumerate(data):
                if _ % 1000 == 0:
                    print(_)
                for word in datapoint:
                    if word is not None:
                        if 'src' in src:
                            update(word_dic, word[0])
                            update(pos_dic, word[2])
                            update(dep_dic, word[5])
                        elif 'trg' in src and 'test' not in src:
                            update(trg_word_dic, word[0].lower())
                            update(trg_pos_dic, word[2])
print(len(word_dic))
json.dump({'word': word_dic,
           'pos': pos_dic,
           'dep': dep_dic,
           'trg_word': trg_word_dic,
           'trg_pos': trg_pos_dic
           },
          open('{}.{}.dictionary.json'.format(TRACK, lang), 'w'))


pos_dic1 = {}
pos_dic2 = {}
form_dic = {}
char_dic = {}
char_dic['<start>'] = 1
char_dic['<eou>'] = 2

for src in os.listdir('data'):
    if lang in src and src[-3:] == 'mor':
        data = pickle.load(open(os.path.join('data', src), 'rb'))
        for _, datapoint in enumerate(data):
            if _ % 1000 == 0:
                print(_)
            for word in datapoint:
                for c in word[0]:
                    update(char_dic, c)
                for c in word[1]:
                    update(char_dic, c)
                update(pos_dic1, word[2])
                update(pos_dic2, word[3])
                if word[4] != '_':
                    for form in word[4].split('|'):
                        if form[:4] != 'lin=' and form[:8] != 'original':
                            update(form_dic, form)

json.dump({'pos1': pos_dic1,
           'pos2': pos_dic2,
           'char': char_dic,
           'form': form_dic
           },
          open('T1.{}.mor.dictionary.json'.format(lang), 'w'))
