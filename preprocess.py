import pdb
import sys
import os
import pickle
from copy import copy


TRAIN = 2
TRACK = 3
lang = sys.argv[1]

src_data = []
trg_data = []

if TRACK == 1:
    if TRAIN == 1:
        src_dir = 'T1-train'
        for src in os.listdir(src_dir):
            if src[:2] == lang:
                datapoint = []
                for l, line in enumerate(open(os.path.join(src_dir, src)).readlines()):
                    if l % 10000 == 0:
                        print(l)
                    line = line.split('\t')
                    if len(line) <= 1:
                        truth = [0 for x in datapoint]
                        for _, data in enumerate(datapoint):
                            truth[data[3]] = _
                        src_data.append(datapoint)
                        trg_data.append(truth)
                        datapoint = []
                    else:
                        data = [line[x] for x in [1, 2, 3, 5, 6, 7, 5]]
                        data[3] = int(data[3].split('=')[-1]) - 1
                        data[4] = int(data[4]) - 1
                        data[6] = [token for token in data[6].split('|') if 'lin' in token]
                        data[6] = 0 if len(data[6]) == 0 else int(data[6][0].split('=')[-1])
                        if data[6] > 1:
                            data[6] = 2
                        if data[6] == -1:
                            data[6] = 3
                        if data[6] < -1:
                            data[6] = 4
                        datapoint.append(data)

        pickle.dump(src_data, open('data/T1.{}.train.src'.format(lang), 'wb'))
        pickle.dump(trg_data, open('data/T1.{}.train.trg'.format(lang), 'wb'))
    elif TRAIN == 2:
        src_dir = 'UD-dev'
        for src in os.listdir(src_dir):
            if src[:2] == lang:
                datapoint = []
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
                        truth = [0 for x in datapoint]
                        for _, data in enumerate(datapoint):
                            truth[data[3]] = _
                        src_data.append(datapoint)
                        trg_data.append(truth)
                        datapoint = []
                    else:
                        data = [line[x] for x in [2, 1, 3, 5, 6, 7, 9]]
                        data[3] = len(datapoint)
                        data[4] = int(data[4]) - 1
                        data[6] = len(datapoint) - data[4]
                        if data[6] > 1:
                            data[6] = 2
                        if data[6] == -1:
                            data[6] = 3
                        if data[6] < -1:
                            data[6] = 4
                        datapoint.append(data)

        pickle.dump(src_data, open('data/T1.{}.dev.src'.format(lang), 'wb'))
        pickle.dump(trg_data, open('data/T1.{}.dev.trg'.format(lang), 'wb'))
    else:
        src_dir = 'T1-test'
        for src in os.listdir(src_dir):
            if src[:2] == lang:
                datapoint = []
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
                        truth = [0 for x in datapoint]
                        for _, data in enumerate(datapoint):
                            truth[data[3]] = _
                        src_data.append(datapoint)
                        trg_data.append(truth)
                        datapoint = []
                    else:
                        data = [line[x] for x in [1, 2, 3, 5, 6, 7, 5]]
                        data[3] = len(datapoint)
                        data[4] = int(data[4]) - 1
                        data[6] = [token for token in data[6].split('|') if 'lin' in token]
                        data[6] = 0 if len(data[6]) == 0 else int(data[6][0].split('=')[-1])
                        if data[6] > 1:
                            data[6] = 2
                        if data[6] == -1:
                            data[6] = 3
                        if data[6] < -1:
                            data[6] = 4
                        datapoint.append(data)

        pickle.dump(src_data, open('data/T1.{}.test.src'.format(lang), 'wb'))
        pickle.dump(trg_data, open('data/T1.{}.test.trg'.format(lang), 'wb'))
elif TRACK == 2:
    if TRAIN == 1:
        src_dir = 'T2-train'
        orig_src_data = pickle.load(open('data/T1.en.train.src', 'rb'))
        en_lst = [f for f in os.listdir(src_dir) if f[:2] == 'en']
        en_lst = [en_lst[x] for x in [0, 3, 2, 1]]
        for src in en_lst:
            datapoint = []
            for l, line in enumerate(open(os.path.join(src_dir, src)).readlines()):
                if l % 10000 == 0:
                    print(l)
                line = line.split('\t')
                if len(line) <= 1:
                    complete = orig_src_data[len(src_data)]
                    truth = [None for x in complete]
                    for data in datapoint:
                        if data[4] > -1:
                            data[4] = datapoint[data[4]][3]
                    for data in datapoint:
                        if data[3] > -1:
                            truth[data[3]] = data
                    missing = [data for data in complete if truth[data[3]] is None]
                    for data in missing:
                        data[4] = complete[data[4]][3]
                    src_data.append(truth)
                    trg_data.append(missing)
                    datapoint = []
                else:
                    data = [line[x] for x in [1, 2, 3, 5, 6, 7, 5]]
                    original_id = data[3].find('original_id')
                    if original_id == -1:
                        data[3] = -1
                        data[4] = -1
                    else:
                        original_id = data[3][original_id:].split('|')[0]
                        data[3] = int(original_id.split('=')[1]) - 1
                        data[4] = int(data[4]) - 1
                    data[6] = 0
                    datapoint.append(data)

        pickle.dump(src_data, open('data/T2.{}.train.src'.format(lang), 'wb'))
        pickle.dump(trg_data, open('data/T2.{}.train.trg'.format(lang), 'wb'))
    elif TRAIN == 2:
        src_dir = 'T2-dev'
        orig_src_data = pickle.load(open('data/T1.en.dev.src', 'rb'))
        en_lst = [f for f in os.listdir(src_dir) if f[:2] == 'en']
        en_lst = [en_lst[x] for x in [3, 2, 0, 1]]
        for src in en_lst:
            datapoint = []
            datapoint = []
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
                    complete = orig_src_data[len(src_data)]
                    truth = [None for x in complete]
                    matched = [False for x in complete]
                    for data in datapoint:
                        if data[0] != '_' and data[4] == -1:
                            match = [c for c in complete if c[4] == -1]
                            assert len(match) == 1
                            data[3] = match[0][3]
                            data[4] = match[0][4]
                            matched[data[3]] = True
                    have_children = [False for x in complete]
                    for data in complete:
                        have_children[data[4]] = True
                    for data in datapoint:
                        if data[0] != '_' and data[4] != -1:
                            match = [c for c in complete if c[0] == data[0] and c[2] == data[2] and complete[c[4]][0] == datapoint[data[4]][0] and not matched[c[3]]]
                            if len(match) > 1:
                                _match = [c for c in match if have_children[c[3]]]
                                if _match:
                                    match = _match
                            data[3] = match[0][3]
                            data[4] = match[0][4]
                            matched[data[3]] = True
                    for data in datapoint:
                        if data[3] > -1:
                            truth[data[3]] = data
                    assert sum(1 for m in matched if m) == sum(1 for d in datapoint if d[0] != '_')
                    empty_par = [d for d in truth if d and d[4] != -1 and truth[d[4]] is None]
                    missing = [data for data in complete if truth[data[3]] is None]
                    if empty_par:
                        pdb.set_trace()
                    src_data.append(truth)
                    trg_data.append(missing)
                    datapoint = []   
                else:
                    data = [line[x] for x in [1, 2, 3, 5, 6, 7, 5]]
                    data[3] = -1
                    if data[0] == '_':
                        data[4] = -1
                    else:
                        data[4] = int(data[4]) - 1
                    data[6] = 0
                    datapoint.append(data)

        pickle.dump(src_data, open('data/T2.{}.dev.src'.format(lang), 'wb'))
        pickle.dump(trg_data, open('data/T2.{}.dev.trg'.format(lang), 'wb'))
    else:
        src_dir = 'T2-test'
        for src in os.listdir(src_dir):
            if src[:2] == lang:
                datapoint = []
                for l, line in enumerate(open(os.path.join(src_dir, src)).readlines()):
                    if l % 10000 == 0:
                        print(l)
                    line = line.split('\t')
                    if len(line) <= 1:
                        truth = [0 for x in datapoint]
                        for _, data in enumerate(datapoint):
                            truth[data[3]] = _
                        src_data.append(datapoint)
                        trg_data.append(truth)
                        datapoint = []
                    else:
                        data = [line[x] for x in [1, 2, 3, 5, 6, 7, 5]]
                        data[3] = len(datapoint)
                        data[4] = int(data[4]) - 1
                        data[6] = 0
                        datapoint.append(data)

        pickle.dump(src_data, open('data/T2.{}.test.src'.format(lang), 'wb'))
        pickle.dump(trg_data, open('data/T2.{}.test.trg'.format(lang), 'wb'))
elif TRACK == 3:
    if TRAIN == 1:
        src_dir = 'T2-train'
        orig_src_data = pickle.load(open('data/T1.en.train.src', 'rb'))
        en_lst = [f for f in os.listdir(src_dir) if f[:2] == 'en']
        en_lst = [en_lst[x] for x in [0, 3, 2, 1]]
        for src in en_lst:
            datapoint = []
            for l, line in enumerate(open(os.path.join(src_dir, src)).readlines()):
                if l % 10000 == 0:
                    print(l)
                line = line.split('\t')
                if len(line) <= 1:
                    complete = orig_src_data[len(src_data)]
                    truth = [None for x in complete]
                    for data in datapoint:
                        if data[4] > -1:
                            data[4] = datapoint[data[4]][3]
                    for data in datapoint:
                        if data[3] > -1:
                            truth[data[3]] = data
                    missing = [data for data in complete if truth[data[3]] is None]
                    for data in missing:
                        data[4] = complete[data[4]][3]

                    for data in missing:
                        truth[data[3]] = copy(data)
                        truth[data[3]][2] = 0
                        truth[data[3]][-1] = 0
                    assert None not in truth
                    src_data.append(truth)
                    trg_data.append(list(range(len(truth))))
                    datapoint = []
                else:
                    data = [line[x] for x in [1, 2, 3, 5, 6, 7, 5]]
                    original_id = data[3].find('original_id')
                    if original_id == -1:
                        data[3] = -1
                        data[4] = -1
                    else:
                        original_id = data[3][original_id:].split('|')[0]
                        data[3] = int(original_id.split('=')[1]) - 1
                        data[4] = int(data[4]) - 1
                    data[6] = 0
                    datapoint.append(data)

        pickle.dump(src_data, open('data/T3.en.train.src', 'wb'))
        pickle.dump(trg_data, open('data/T3.en.train.trg', 'wb'))
    elif TRAIN == 2:
        src_dir = 'T2-dev'
        orig_src_data = pickle.load(open('data/T1.en.dev.src', 'rb'))
        en_lst = [f for f in os.listdir(src_dir) if f[:2] == 'en']
        en_lst = [en_lst[x] for x in [3, 2, 0, 1]]
        for src in en_lst:
            datapoint = []
            datapoint = []
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
                    complete = orig_src_data[len(src_data)]
                    truth = [None for x in complete]
                    matched = [False for x in complete]
                    for data in datapoint:
                        if data[0] != '_' and data[4] == -1:
                            match = [c for c in complete if c[4] == -1]
                            assert len(match) == 1
                            data[3] = match[0][3]
                            data[4] = match[0][4]
                            matched[data[3]] = True
                    have_children = [False for x in complete]
                    for data in complete:
                        have_children[data[4]] = True
                    for data in datapoint:
                        if data[0] != '_' and data[4] != -1:
                            match = [c for c in complete if c[0] == data[0] and c[2] == data[2] and complete[c[4]][0] == datapoint[data[4]][0] and not matched[c[3]]]
                            if len(match) > 1:
                                _match = [c for c in match if have_children[c[3]]]
                                if _match:
                                    match = _match
                            data[3] = match[0][3]
                            data[4] = match[0][4]
                            matched[data[3]] = True
                    for data in datapoint:
                        if data[3] > -1:
                            truth[data[3]] = data
                    assert sum(1 for m in matched if m) == sum(1 for d in datapoint if d[0] != '_')
                    empty_par = [d for d in truth if d and d[4] != -1 and truth[d[4]] is None]
                    missing = [data for data in complete if truth[data[3]] is None]
                    if empty_par:
                        pdb.set_trace()

                    for data in missing:
                        truth[data[3]] = data
                        truth[data[3]][2] = 0
                        truth[data[3]][-1] = 0
                    assert None not in truth           
                    src_data.append(truth)
                    trg_data.append(list(range(len(truth))))
                    datapoint = []   
                else:
                    data = [line[x] for x in [1, 2, 3, 5, 6, 7, 5]]
                    data[3] = -1
                    if data[0] == '_':
                        data[4] = -1
                    else:
                        data[4] = int(data[4]) - 1
                    data[6] = 0
                    datapoint.append(data)

        pickle.dump(src_data, open('data/T3.en.dev.src', 'wb'))
        pickle.dump(trg_data, open('data/T3.en.dev.trg', 'wb'))
elif TRACK == 4:
    if TRAIN == 1:
        src_dir = 'T1-train'
        for src in os.listdir(src_dir):
            if src[:2] == lang:
                datapoint = []
                for l, line in enumerate(open(os.path.join(src_dir, src)).readlines()):
                    if l % 10000 == 0:
                        print(l)
                    line = line.split('\t')
                    if len(line) <= 1:
                        src_data.append(datapoint)
                        datapoint = []
                    else:
                        data = [line[x] for x in range(1, 6)]
                        datapoint.append(data)

        pickle.dump(src_data, open('data/T1.{}.train.mor'.format(lang), 'wb'))
    elif TRAIN == 2:
        src_dir = 'UD-dev'
        for src in os.listdir(src_dir):
            if src[:2] == lang:
                datapoint = []
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
                        src_data.append(datapoint)
                        datapoint = []
                    else:
                        data = [line[x] for x in [2, 1, 3, 4, 5]]
                        datapoint.append(data)
        pickle.dump(src_data, open('data/T1.{}.dev.mor'.format(lang), 'wb'))
    else:
        src_dir = 'T1-test'
        for src in os.listdir(src_dir):
            if src[:2] == lang:
                datapoint = []
                for l, line in enumerate(open(os.path.join(src_dir, src)).readlines()):
                    if l % 10000 == 0:
                        print(l)
                    line = line.split('\t')
                    if len(line) <= 1:
                        src_data.append(datapoint)
                        datapoint = []
                    else:
                        data = [line[x] for x in range(1, 6)]
                        datapoint.append(data)
        pickle.dump(src_data, open('data/T1.{}.test.mor'.format(lang), 'wb'))
