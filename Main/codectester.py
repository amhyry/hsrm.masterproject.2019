import sys
import locale
import io
import sys
import os
import platform
import math
from datetime import datetime
import nltk
import random
import pickle
from ClassifierBasedGermanTagger.ClassifierBasedGermanTagger import ClassifierBasedGermanTagger


sys.stdout = io.TextIOWrapper(sys.stdout.detach(), encoding = 'utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.detach(), encoding = 'utf-8')

MODULE = '/users/tom/desktop/pattern'
import sys;
if MODULE not in sys.path: sys.path.append(MODULE)
from pattern.de import singularize, conjugate, predicative

def lemma_via_patternlib(token, pos):
    if pos == 'NP':  # singularize noun
        return singularize(token)
    elif pos.startswith('V'):  # get infinitive of verb
        return conjugate(token)
    elif pos.startswith('ADJ') or pos.startswith('ADV'):  # get baseform of adjective or adverb
        return predicative(token)

    return token

def test0_system_stdout_environment():
    """ Test this with/without stdout, err changing above """

    #string = u'안녕세계'
    #print(string)

    print(sys.getdefaultencoding())
    print(sys.stdout.encoding)
    # corp = nltk.corpus.ConllCorpusReader('.', 'tiger_release_aug07.corrected.16012013.conll09',
    #                                      ['ignore', 'words', 'ignore', 'ignore', 'pos'],
    #                                      encoding='utf-8')
    #
    #
    # tagged_sents = list(corp.tagged_sents())
    # random.shuffle(tagged_sents)
    #
    # # set a split size: use 90% for training, 10% for testing
    # split_perc = 0.1
    # split_size = int(len(tagged_sents) * split_perc)
    # train_sents, test_sents = tagged_sents[split_size:], tagged_sents[:split_size]

    #tagger = ClassifierBasedGermanTagger(train=train_sents)


if __name__ == '__main__':
    if sys.stdout.isatty():
        default_encoding = sys.stdout.encoding
    else:
        default_encoding = locale.getpreferredencoding()
    print(default_encoding)
    test0_system_stdout_environment()
    print(lemma_via_patternlib("kauft", "Ver"))

        #print(i*math.pow(10, x.index(i)+1))
        #l = l + [i * 2]

        #print(i)








    #print(u"some unicode text \N{EURO SIGN}")
    #print(b"some utf-8 encoded bytestring \xe2\x82\xac".decode('utf-8'))
