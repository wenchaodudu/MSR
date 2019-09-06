import pdb
import sys
import os


lang = sys.argv[1]
src_dir = 'T1-test'
if lang == 'zh':
    lines = open('T1.{}.test.txt'.format(lang)).readlines()
    for src in os.listdir(src_dir):
        if src[:2] == lang:
            i = src.find('-ud-test')
            if i == -1:
                i = src.find('.conllu')
            output = open('T1-submission/{}.txt'.format(src[:-7]), 'w')
            datapoint = []
            sid = 1
            lind = 2
            while lind < len(lines):
                output.write('#sent_id = {}\n'.format(sid))
                output.write('#text = {}\n'.format(lines[lind].strip()))
                output.write('\n')
                sid += 1
                lind += 4

else:
    words = open('T1.{}.test.morph.txt'.format(lang)).readlines()
    ind = 0
    for src in os.listdir(src_dir):
        if src[:2] == lang:
            i = src.find('-ud-test')
            if i == -1:
                i = src.find('.conllu')
            output = open('T1-submission/{}.txt'.format(src[:-7]), 'w')
            datapoint = []
            sid = 1
            for l, line in enumerate(open(os.path.join(src_dir, src)).readlines()):
                if l % 10000 == 0:
                    print(l)
                if line[0] == '#':
                    continue
                line = line.split('\t')
                if '-' in line[0] or '.' in line[0]:
                    print(line[0])
                    continue
                if len(line) <= 1:
                    output.write('#sent_id = {}\n'.format(sid))
                    output.write('#text = {}\n'.format(' '.join(datapoint)))
                    output.write('\n')
                    datapoint = []
                    sid += 1
                else:
                    datapoint.append(words[ind].split()[1])
                    ind += 1

    print(ind, len(words))

