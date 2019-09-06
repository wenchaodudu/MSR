""""Evaluates generated sentences.

Please, install NLTK: https://www.nltk.org/install.html
sudo pip install -U nltk


python eval.py <system-dir> <reference-dir>

e.g.
python bin/eval.py system_out_dev Finall4/Sentences/dev/

Author: Bernd Bohnet, bohnetbd@gmail.com
"""

import codecs
import collections
import io
import os
import sys
import pdb
try:
  from nltk.metrics import *
  import nltk.translate.nist_score as ns
  import nltk.translate.bleu_score as bs
except ImportError:
  print('Please install nltk (https://www.nltk.org/)')
  print("For instance: 'sudo pip install -U nltk\'")
  exit()


def read_corpus(filename, ref=False, normalize=True):
  """Reads a corpus

  Args:
    filename: Path and file name for the corpus.

  Returns:
    A list of the sentences.
  """
  data = []
  with open(filename, 'r') as f:
    for line in codecs.getreader('utf-8')(f, errors='ignore'):  # type: ignore
      line = line.rstrip()
      if line.startswith(u'# text'):
        split = line.split(u'text = ')
        if len(split) > 1:
          text = split[1]
        else:
          text = ''
        if normalize:
          text = text.lower()
        if ref:
          data.append([text.split()])
        else:
          data.append(text.split())
  return data

def read(output):
    lines = open(output).readlines()
    lines = [l.lower() for l in lines]
    for _, line in enumerate(lines):
        words = line.split()
        words = ['is' if w == "'s" else w for w in words]
        words = ['am' if w == "'m" else w for w in words]
        words = ['have' if w == "'ve" else w for w in words]
        words = ['are' if w == "'re" else w for w in words]
        words = ['will' if w == "'ll" else w for w in words]
        words = ['would' if w == "'d" else w for w in words]
        words = ['not' if w == "n't" else w for w in words]
        lines[_] = ' '.join(words)
    ref = []
    hyp = []
    for x in range(len(lines) // 4):
        ref.append([lines[x*4+0].split()])
        hyp.append(lines[x*4+2].split())
    return ref, hyp

def main():
    '''
    arguments = sys.argv[1:]
    num_args = len(arguments)
    if num_args != 2:
    print('Wrong number few arguments.')
    print(sys.argv[0], 'system-dir', 'referene-dir')
    exit()
    system_path = arguments[0]
    ref_path = arguments[1]
    '''
    output_path = sys.argv[1]

    # For all files in system path.
    # read files
    ref, hyp = read(output_path)

    # NIST score
    nist = ns.corpus_nist(ref, hyp, n=4)

    # BLEU score
    chencherry = bs.SmoothingFunction()
    bleu = bs.corpus_bleu(ref, hyp, smoothing_function=chencherry.method2)
    print ('BLEU', round(bleu, 3))
    total_len = 0.0
    edi = 0.0
    for r, h in zip(ref, hyp):
      try:
        total_len += max(len(r[0][0]), len(h[0]))
        edi += edit_distance(r[0][0], h[0])
      except:
        #print('r', r[0])
        #print('h', h)
        print("ERROR")
        pass
    print ('DIST', round(1-edi/total_len,3))
    print ('NIST', round(nist, 6))
    print ('')


if __name__ == "__main__":
    main()
