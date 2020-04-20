# coding: utf8
import json

#import warnings
#warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
# languange processing imports
import nltk
# preprocessing imports
from sklearn.preprocessing import LabelEncoder

from collections import Counter
import multiprocessing
import re
import string
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import pprint
import sys
import locale
import io
import JsonToSentencesConverter as Crawler
import os
from datetime import datetime
import pandas as pd

from sklearn.manifold import TSNE
import re
import matplotlib.pyplot as plt

# model imports
from gensim.models.ldamulticore import LdaMulticore

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression


from gensim.models import TfidfModel
from gensim.models import Word2Vec
from gensim.corpora import Dictionary
from nltk.corpus import brown
from gensim.summarization import keywords
from gensim import corpora
from gensim.models.ldamodel import LdaModel
import numpy as np



# hyperparameter training imports
from sklearn.model_selection import GridSearchCV





# visualization imports
from IPython.display import display
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import base64
import io
#%matplotlib inline
sns.set()  # defines the style of the plots to be seaborn style



#from . import Crawler

sys.stdout = io.TextIOWrapper(sys.stdout.detach(), encoding = 'utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.detach(), encoding = 'utf-8')

class NLP_NeuralNet:
    """Class docstrings go here.
    ...

    Args:
        json_file : str
            filename/filepath to jsonfile, which contains sentences
        vocabulary_size : int
            [optional] Size of Vocabulary

    Methods
        _initialize(sentences_list=None):
            Prints the animals name and what sound it makes
    """

    def __init__(self, test = True):
        self._initialized = False
        print('Start')
        #self.source = ""
        #self.destination = ""
        #self.dir_up = ""
        self.smalldata = test
        self.modelstr = "_33001"

        if(os.name == "posix"):
            self.dir_up = '..'
        if(os.name == "nt"):
            self.dir_up = '.'
#        print("Path: " + os.name + dir_up)
        #version = sum([int(item) for item in datetime.now().strftime('%m_%d_%H_%M').split('_')])
        version = datetime.now().strftime('%m_%d_%H_%M')

        if self.smalldata == True:
            self.modelstr = "_1001_" + version
            self.source = self.dir_up + '/Data/sampleFromDataCrowlerindeed1001.json'
            self.destination =  self.dir_up + '/Data/data_for_voc_1001.json'
        else:
            self.modelstr = "_33001_" + version
            self.source = self.dir_up + '/Data/sampleFromDataCrowlerindeed33001.json'
            self.destination = self.dir_up + '/Data/data_for_voc_33001.json'

    def convert(self):
        Crawler.converter(self.source, self.destination)

    def loadData(self):
        with open(self.destination, encoding='utf-8') as json_file:
            self.data = json.load(json_file)
            print('daten geladen')
        return self.data

    def makeLDA(self, data, load=None):
        #for a in data:
        #print(a.decode("utf8","replace"))
        #doc_complete = dict2listConverter(data)
            #only_str_docs = removeEmptyListsFromDocs(doc_complete)
        if load == None:
            doc_clean = [clean(doc).split() for doc in data]
            #print(doc_clean)
            print("Dokumente gesaeubert")
            dictionary = corpora.Dictionary(doc_clean)
            print("Dictionary angelegt")
            print(dictionary)
            doc_term_matrix = [dictionary.doc2bow(doc) for doc in doc_clean]
            print("Matrix erstellt")
            #print(doc_term_matrix)
            ldamodel = LdaModel(doc_term_matrix, num_topics=3, id2word = dictionary, passes=50)
            ldamodel.save(self.dir_up + '/Data/Model/lda_model'+self.modelstr)
        else:
            load = self.dir_up + '/Data/Model/' + load
            ldamodel = LdaModel.load(load)
            #ldamodel = LdaModel.load(dir_up + '/Data/Model/ldamodel')
        print(ldamodel.print_topics(num_topics=3, num_words=10))
        self.lda = ldamodel
        return ldamodel

    def makeW2V(self, data, load=None):
        if load == None:

            EMB_DIM = 300
            data_clean = [doc.split() for doc in data]
    #print(data_clean[1])
    #print(multiprocessing.cpu_count())
            w2v = Word2Vec(data_clean, size=EMB_DIM, window=5, min_count=5, negative=15, iter=10, workers=multiprocessing.cpu_count())
            w2v.save(self.dir_up+'/Data/Model/wtov_model'+self.modelstr)
        else:
            load = self.dir_up + '/Data/Model/' + load
            #ldamodel = LdaModel.load(load)
            #w2v = Word2Vec.load(dir_up+'/Data/Model/wtov_model'+modelstr)
            w2v = Word2Vec.load(load)
        self.w2v = w2v
        return w2v
        #wv = w2v.wv




stop = set(stopwords.words('german'))
exclude = set(string.punctuation)
lemma = WordNetLemmatizer()

def word_frequency_barplot(df, nr_top_words=50):
    """ df should have a column named count.
    """
    fig, ax = plt.subplots(1,1,figsize=(20,5))

    sns.barplot(list(range(nr_top_words)), df['count'].values[:nr_top_words], palette='hls', ax=ax)

    ax.set_xticks(list(range(nr_top_words)))
    ax.set_xticklabels(df.index[:nr_top_words], fontsize=14, rotation=90)
    return ax

def removeEmptyListsFromDocs(docs):
    result = []
    for doc in docs:
        if isinstance(doc,str):
            result.append(doc)
    return result


def clean(doc):
    stop_free = " ".join([i for i in doc.lower().split() if i not in stop])
    punct_free = ''.join([ch for ch in stop_free if ch not in exclude])
    normalized = " ".join([lemma.lemmatize(word) for word in punct_free.split()])
    return normalized

def umlauteConverter(text):
    text = re.sub('[%s]' % re.escape(string.punctuation), ' ', text)
    text = text.encode("utf-8",'replace')
    text = text.decode('utf-8','replace')
    return text

def sentence_nomalizer(text):
    result = []
    text = umlauteConverter(text)
    #text = [w for w in text.split() ]
    #for value in text:
    #    if len(value) >= 1: result.append(value)

    return text

def dict2listConverter(dataAsDict):
    result = []
    for key in dataAsDict.keys():
        result.append(dataAsDict[key])
    return result

def remove_ascii_words(df):
    """ removes non-ascii characters from the 'texts' column in df.
    It returns the words containig non-ascii characers.
    """
    our_special_word = 'qwerty'
    non_ascii_words = []
    for i in range(len(df)):
        for word in df.loc[i, 'text'].split(' '):
            if any([ord(character) >= 128 for character in word]):
                non_ascii_words.append(word)
                df.loc[i, 'text'] = df.loc[i, 'text'].replace(word, our_special_word)
    return non_ascii_words




def w2v_preprocessing(df):
    """ All the preprocessing steps for word2vec are done in this function.
    All mutations are done on the dataframe itself. So this function returns
    nothing.
    """
    df['text'] = df.text.str.lower()
    df['document_sentences'] = df.text.str.split('.')  # split texts into individual sentences
    df['tokenized_sentences'] = list(map(lambda sentences:
                                         list(map(nltk.word_tokenize, sentences)),
                                         df.document_sentences))  # tokenize sentences
    df['tokenized_sentences'] = list(map(lambda sentences:
                                         list(map(get_good_tokens, sentences)),
                                         df.tokenized_sentences))  # remove unwanted characters
    df['tokenized_sentences'] = list(map(lambda sentences:
                                         list(filter(lambda lst: lst, sentences)),
                                         df.tokenized_sentences))  # remove empty lists

def lda_get_good_tokens(df):
    df['text'] = df.text.str.lower()
    df['tokenized_text'] = list(map(nltk.word_tokenize, df.text))
    df['tokenized_text'] = list(map(get_good_tokens, df.tokenized_text))

def get_good_tokens(sentence):
    replaced_punctation = list(map(lambda token: re.sub('[^0-9A-Za-z!?]+', '', token), sentence))
    removed_punctation = list(filter(lambda token: token, replaced_punctation))
    return removed_punctation


def giveLdaTopicsWithoutStopwords(lda):
    #print(lda.print_topics(num_topics=3, num_words=10))
    #print(stop)
    tmp = lda.print_topics(num_topics=3, num_words=1)
    print ("Ganz NEU: ",tmp)

if __name__ == '__main__':

    #print(destination)

    #x = NLP_NeuralNet(test = True) # test auf false und es werden 33001 daten geladen
    #x.convert()
    #data = x.loadData()
    #print(data)
    #lda = x.makeLDA(data, load='lda_model_33001')
    #w2v = x.makeW2V(data, load='wtov_model_1001')
    #doc_clean = [clean(doc).split() for doc in data]
    #doc_clean = [doc.split() for doc in data]
    #print(doc_clean)
    print("Dokumente gesaeubert")
    #dictionary = corpora.Dictionary(doc_clean)
    print("Dictionary angelegt")
    #print(dictionary)


    train_data = pd.read_json('../Data/data_for_voc_1001.json', encoding='utf-8')

    print(train_data.shape)
    train_data.rename(columns={0:'text'},inplace=True)

    print(train_data.head(3))

    #print(train_data)

    lda_get_good_tokens(train_data)

    #doc_term_matrix = [dictionary.doc2bow(doc) for doc in doc_clean]
    print("Matrix erstellt")


    #story_topics = pd.DataFrame(dict(story_id_codes= np.arange(dat['doc_topic_dists'].shape[0])))


    #train_data.tokenized_text.values

    non_ascii_words = remove_ascii_words(train_data)

    print("Replaced {} words with characters with an ordinal >= 128 in the train data.".format(len(non_ascii_words)))

    w2v_preprocessing(train_data)
    lda_get_good_tokens(train_data)

    tokenized_only_dict = Counter(np.concatenate(train_data.tokenized_text.values))

    tokenized_only_df = pd.DataFrame.from_dict(tokenized_only_dict, orient='index')
    tokenized_only_df.rename(columns={0: 'count'}, inplace=True)

    tokenized_only_df.sort_values('count', ascending=False, inplace=True)
    ax = word_frequency_barplot(tokenized_only_df)
    ax.set_title("Word Frequencies", fontsize=16);
    plt.show()
    '''
    lda = x.makeLDA(data, load='lda_model_1001_08_25_15_13')
    w2v = x.makeW2V(data, load='wtov_model_1001_08_25_15_13')
    #giveLdaTopicsWithoutStopwords(lda)
    #lda = x.makeLDA(data)
    #w2v = x.makeW2V(data)

    wv = w2v.wv

    print(wv.vocab)

    vocab = list(wv.vocab)
    '''
    '''
    X = w2v[vocab]
    tsne = TSNE(n_components=2)
    X_tsne = tsne.fit_transform(X)
    df = pd.DataFrame(X_tsne, index=vocab, columns=['x', 'y'])
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    ax.scatter(df['x'], df['y'])
    for word, pos in df.iterrows():
        ax.annotate(word, pos)
    plt.show()
    '''


    #kwords = []
    #i = 0
    #j = 0
    #for key in data:
    #    i+=1
    #    print(key)
    #    print(data[key])
    #    if isinstance(data[key],list) :
    #        j +=1
    #        continue
    #
    #    #print(type(data[key]))
    #    #print(keywords("".join(data[key])))
    #    text = sentence_nomalizer(data[key])
    #    print(text)
    #    print(keywords(text))
    #    kwords.append(keywords(text))

    #EMB_DIM = 300
    #data_clean = [doc.split() for doc in data]
    #print(data_clean[1])
    #print(multiprocessing.cpu_count())
    #w2v = Word2Vec(data_clean, size=EMB_DIM, window=5, min_count=5, negative=15, iter=10, workers=multiprocessing.cpu_count())
    #w2v.save(dir_up+'/Data/Model/wtov_model'+modelstr)
    #w2v = Word2Vec.load(dir_up+'/Data/Model/wtov_model'+modelstr)
    #wv = w2v.wv

    #print(wv.vocab)

    #with open('./Data/keywords.json', 'w', encoding='utf8') as json_file:
    #    data = json.dumps(kwords, ensure_ascii=False)
    #    json_file.write(str(data))

    #result = wv.similar_by_word('Sprachen')
    #print("Most Similar to 'Sprachen':\n", result[:20])

    #downloader.list(show_packages=False)

    #print(downloader.supported_tasks(lang="en"))
    #print(downloader.supported_languages_table(task="ner2"))

    #value = downloader.download("morph2.fy")
    #value2 = downloader.download("morph2.en")
    #print("Download value: ", value)

    #for sentence in sentences[:100]:
    #    if Text(sentence).language.code == "de":
    #        print(Text(sentence).entities)

    #for sentence in sentences[:100]:
    #    for word in Text(sentence).words:
            #if Text(sentence).language.code == "de":
    #        print(word, " -> ", word.morphemes)

    #for sentence in sentences:
    #    if Text(sentence).language.code == "de":
    #        print("{:<16}{}".format("Word", "POS Tag")+"\n"+"-"*30)
    #        for word, tag in Text(sentence).pos_tags:
    #            print(u"{:<16}{:>2}".format(word,tag))

    #print("{:<16}{}".format("Word", "POS Tag")+"\n"+"-"*30)
    #for word, tag in text.pos_tags:
    #    print(u"{:<16}{:>2}".format(word, tag))

    import sys

    print(sys.version)


    #result =
