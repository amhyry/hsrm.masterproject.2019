# coding: utf8
#import warnings
#warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
#our own classes
import JsonToSentencesConverter as Crawler
from ClassifierBasedGermanTagger.ClassifierBasedGermanTagger import ClassifierBasedGermanTagger

#basics
import pprint
import sys
import locale
import io
import multiprocessing
import re
import string
import os
from datetime import datetime
import numpy as np
import pandas as pd
import pickle

import json
import re
#new import
from collections import Counter
from pattern.de import singularize, conjugate, predicative
#gensim
from gensim.models import TfidfModel
from gensim.models import Word2Vec
from gensim.corpora import Dictionary
from gensim.summarization import keywords
from gensim import corpora
from gensim.models.ldamodel import LdaModel
#new import
from gensim.models.ldamulticore import LdaMulticore

import nltk
from nltk.corpus import brown
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem import SnowballStemmer

from sklearn.manifold import TSNE
# hyperparameter training imports
from sklearn.model_selection import GridSearchCV
#new import
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

#plotting
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.cm as cm

# visualization imports
from IPython.display import display
import matplotlib.image as mpimg
import base64
import io

#polyglot


#%matplotlib inline
sns.set()  # defines the style of the plots to be seaborn style
#MODULE = '/Library/Frameworks/Python.framework/Versions/3.6/lib/python3.6/site-packages/pattern/'
MODULE = 'C:/Users/wdering/AppData/Local/Programs/Python/Python37/Lib/site-packages/pattern/'
if MODULE not in sys.path: sys.path.append(MODULE)
from pattern.de import singularize, conjugate, predicative

#correct the individual encoding
sys.stdout = io.TextIOWrapper(sys.stdout.detach(), encoding = 'utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.detach(), encoding = 'utf-8')
pd.set_option('display.max_columns', None)
#%% Start


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
            self.destination = self.dir_up + '/Data/data_for_voc_33001_senc.json'

    def convert(self):
        Crawler.converter(self.source, self.destination)
        print(self.source + " konvertiert in " + self.destination)

    def loadData(self):
        with open(self.destination, encoding='utf-8') as json_file:
            self.data = json.load(json_file)
        self.DF_data = pd.read_json(self.destination, encoding='utf-8')
        self.DF_data.rename(columns={0:'text'},inplace=True)
        print(self.destination + ' Daten geladen in data und DF_data')
        return self.data, self.DF_data

    def makeBOW(self, data, load=None):
        if load == None:
            print("TODO:: vielleicht aus Vocabulary laden?")
        else:
            load = self.dir_up + '/Data/' + load
            with open(load, encoding='utf-8') as json_file:
                bow = json.load(json_file)

        print(load + ' Bag of words geladen')
        self.bow = bow
        return bow

    def makeLDA(self, data, load=None):
        if load == None:
            print("LDA Model wird erzeugt... Daten vorbereiten.")
            doc_clean = [clean(doc).split() for doc in data]
            print("Model wird erzeugt... Dictionary wird angelegt.")
            dictionary = corpora.Dictionary(doc_clean)
            print("Model wird erzeugt... Dictionary angelegt. Erzeuge Matrix...")
            #print(dictionary)
            doc_term_matrix = [dictionary.doc2bow(doc) for doc in doc_clean]
            print("Model wird erzeugt... Matrix erstellt. Beginne Training.")
            #print(doc_term_matrix)
            ldamodel = LdaModel(doc_term_matrix, num_topics=3, id2word = dictionary, passes=50)
            print("LDA Model wurde erzeugt. Model abspeichern...")
            ldamodel.save(self.dir_up + '/Data/Model/lda_model'+self.modelstr)
            print("LDA Model erstellt unter: "+ self.dir_up + '/Data/Model/lda_model'+self.modelstr)
        else:
            load = self.dir_up + '/Data/Model/' + load
            ldamodel = LdaModel.load(load)
            #ldamodel = LdaModel.load(dir_up + '/Data/Model/ldamodel')
        #print(ldamodel.print_topics(num_topics=3, num_words=10))
        self.lda = ldamodel
        print(load + " ldamodel  erfolgreich geladen!")
        return ldamodel

    def makeW2V(self, data, load=None):
        if load == None:
            print("W2V Model wird erzeugt... Daten vorbereiten.")
            EMB_DIM = 300
            #data_clean = [doc.split() for doc in data]
            print("Model wird erzeugt... Daten vorbereitet. Beginne training...")
            w2v = Word2Vec(data, size=EMB_DIM, window=5, min_count=5, negative=15, iter=10, workers=multiprocessing.cpu_count())
            #w2v = Word2Vec(data_clean, size=EMB_DIM, window=5, min_count=5, negative=15, iter=10, workers=multiprocessing.cpu_count())
            print("W2V Model wurde erzeugt. Model abspeichern...")
            w2v.save(self.dir_up+'/Data/Model/wtov_model'+self.modelstr)
            print("W2V Model erstellt unter: "+ self.dir_up+'/Data/Model/wtov_model'+self.modelstr)
            load = self.dir_up+'/Data/Model/wtov_model'+self.modelstr
        else:
            load = self.dir_up + '/Data/Model/' + load
            #w2v = Word2Vec.load(dir_up+'/Data/Model/wtov_model'+modelstr)
            w2v = Word2Vec.load(load)
        self.w2v = w2v
        print(load + " w2vmodel erfolgreich geladen!")
        return w2v
        #wv = w2v.wv



#end of class-------------------------------------------------------------------
stop = set(stopwords.words('german'))
exclude = set(string.punctuation)
lemma = WordNetLemmatizer()

def removeEmptyListsFromDocs(docs):
    result = []
    for doc in docs:
        if isinstance(doc,str):
            result.append(doc)
    return result


def clean(doc):
    stop_free = " ".join([i for i in doc.split() if i.lower() not in stop])
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


#NEUE FUNKTIONS, für plotting---------------------------------------------------
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
#ARNOLDS IMPORTANTS-------------------------------------------------------------
def get_good_tokens(sentence):
    replaced_punctation = list(map(lambda token: re.sub('[^0-9A-Za-zßÄÖÜöäü!?]+', '', token), sentence))
    removed_punctation = list(filter(lambda token: token, replaced_punctation))
    #punct_free = ''.join([ch for ch in stop_free if ch not in exclude])
    return removed_punctation

def preprocessing_tokenize(df):
    df['text_lower'] = df.text.str.lower()
    #df['text'] = df.text.str.lower() #Wladi, wenn wir direkt am anfang lowern, wird jedes POS falsch!
    df['punctuation_free'] = list(map(lambda token: re.sub('[^0-9A-Za-zßÄÖÜöäü !?]+', '', token), df['text']))

    df['tokenized_text_punctuation_free'] = list(map(nltk.word_tokenize, df.text))
    df['tokenized_text_punctuation_free'] = list(map(get_good_tokens, df.tokenized_text_punctuation_free))

    df['tokenized_text'] = list(map(nltk.word_tokenize, df.text_lower))
    df['tokenized_text'] = list(map(get_good_tokens, df.tokenized_text))
    #
    # df['document_sentences'] = df.text.str.split('.')
    # df['tokenized_sentences'] = list(map(lambda sentences:
    #                                      list(map(nltk.word_tokenize, sentences)),
    #                                      df.document_sentences))  # tokenize sentences
    # df['tokenized_sentences'] = list(map(lambda sentences:
    #                                      list(map(get_good_tokens, sentences)),
    #                                      df.tokenized_sentences))  # remove unwanted characters
    # df['tokenized_sentences'] = list(map(lambda sentences:
    #                                      list(filter(lambda lst: lst, sentences)),
    #                                      df.tokenized_sentences))  # remove empty lists

def preprocessing_remove_stopwords(df):
    our_special_word = 'qwerty'
    # Luckily nltk already has a set of stopwords that we can remove from the texts.
    stopwords = nltk.corpus.stopwords.words('german')
    # we'll add our own special word in here 'qwerty'
    stopwords.append(our_special_word)
    stopwords.append("sowie")
    stopwords.append("?")
    stopwords.append("!")
    df['stopwords_removed'] = list(map(lambda doc:
                                       [word for word in doc if word not in stopwords],
                                       df['tokenized_text']))

def preprocessing_pos_tags(df):
    print(os.getcwd())
    with open('./nltk_german_classifier_data.pickle', 'rb') as f:
        tagger = pickle.load(f)
    df['pos_tags'] = list(map(lambda sentence:
                                     tagger.tag(sentence),
                                     df.tokenized_text_punctuation_free))


def preprocessing_primitive_form(df):
    #df['primitive_form'] = list(map(lambda sentence:
    #                                [lemma_via_patternlib(tuple_[0], tuple_[1]) for tuple_ in sentence],
    #                                df['pos_tags']))

    df['only_nouns'] = list(map(lambda sentence:
                                     [word[0].lower() for word in sentence if re.match('^N', word[1])],
                                     df['pos_tags']))
    df['all_other_words'] = list(map(lambda sentence:
                                     [word[0].lower() for word in sentence if not re.match('^N', word[1])],
                                     df['pos_tags']))




def preprocessing_stem_words(df):
    lemm = nltk.stem.WordNetLemmatizer()
    df['lemmatized_text'] = list(map(lambda sentence:
                                     list(map(lemm.lemmatize, sentence)),
                                     df.stopwords_removed))

    stemmer = SnowballStemmer("german")
    df['ger_stemmed_text'] = list(map(lambda sentence:
                                  list(map(stemmer.stem, sentence)),
                                  df.lemmatized_text))

    p_stemmer = nltk.stem.porter.PorterStemmer()
    df['stemmed_text'] = list(map(lambda sentence:
                                  list(map(p_stemmer.stem, sentence)),
                                  df.lemmatized_text))

def preprocessing_doc_to_bow(df):
    dictionary = Dictionary(documents=df.stemmed_text.values)
    df['bow'] = list(map(lambda doc: dictionary.doc2bow(doc), df.stemmed_text))

def preprocessing_generate_Dictionary(df):
    dictionary = Dictionary(documents=df.only_nouns.values)
    print("Found {} words.".format(len(dictionary.values())))
    dictionary.filter_extremes(no_above=0.8, no_below=3)
    dictionary.compactify()  # Reindexes the remaining words after filtering
    print("Left with {} words.".format(len(dictionary.values())))
    #print(dictionary)
    return dictionary

def lemma_via_patternlib(token, pos):
    if pos == 'NP':  # singularize noun
        return singularize(token)
    elif pos.startswith('V'):  # get infinitive of verb
        return conjugate(token)
    elif pos.startswith('ADJ') or pos.startswith('ADV'):  # get baseform of adjective or adverb
        return predicative(token)
    return token

#NOBODY IMPORTANTS-------------------------------------------------------------

#NOBODY IMPORTANTS-------------------------------------------------------------

def document_to_bow(df):
    df['bow'] = list(map(lambda doc: dictionary.doc2bow(doc), df.stemmed_text))

def word_frequency_barplot(df, nr_top_words=50):
    """ df should have a column named count.
    """
    fig, ax = plt.subplots(1,1,figsize=(20,5))

    sns.barplot(list(range(nr_top_words)), df['count'].values[:nr_top_words], palette='hls', ax=ax)

    ax.set_xticks(list(range(nr_top_words)))
    ax.set_xticklabels(df.index[:nr_top_words], fontsize=14, rotation=90)
    return ax

def word_frequency_barplot_pretty(df, nr_top_words=50):
    """ df should have a column named count.
    """
    fig, ax = plt.subplots(1,1,figsize=(20,5))

    #sns.barplot(range(len(df)), list(df.values()), tick_label=list(df.keys()))
    #sns.barplot(list(range(nr_top_words)), df['count'].values[:nr_top_words], palette='hls', ax=ax)
    sns.barplot(list(range(nr_top_words)), list(df.values())[:50], palette='hls', ax=ax)

    ax.set_xticks(list(range(nr_top_words)))

    ax.set_xticklabels(list(sorted_bow.keys())[:50], fontsize=14, rotation=90)
    return ax

def tsne_plot_similar_words(title, labels, embedding_clusters, word_clusters, a, filename=None):
    plt.figure(figsize=(16, 9))
    colors = cm.rainbow(np.linspace(0, 1, len(labels)))
    for label, embeddings, words, color in zip(labels, embedding_clusters, word_clusters, colors):
        x = embeddings[:, 0]
        y = embeddings[:, 1]
        plt.scatter(x, y, c=color, alpha=a, label=label)
        for i, word in enumerate(words):
            plt.annotate(word, alpha=0.5, xy=(x[i], y[i]), xytext=(5, 2), textcoords='offset points', ha='right', va='bottom', size=8)
    plt.legend(loc=4)
    plt.title(title)
    plt.grid(True)
    if filename:
        plt.savefig(filename, format='png', dpi=150, bbox_inches='tight')
    plt.show()

#end----------------------------------------------------------------------------



if __name__ == '__main__':
    my_NLP_nn = NLP_NeuralNet(test = True) # test auf false und es werden 33001 daten geladen
    #my_NLP_nn.convert() #Wenn neue Daten benötigt werden.

    #data, DF_data = my_NLP_nn.loadData()
    #lda = x.makeLDA(data, load='lda_model_33001')
    #w2v = x.makeW2V(data, load='wtov_model_1001')
    dir_up = ".."
    #w2v = Word2Vec.load(dir_up+'/Data/Model/wtov_model_33001_08_26_09_18')
    #bow = my_NLP_nn.makeBOW(data, load='bag_of_words.vocab') #
    #lda = my_NLP_nn.makeLDA(data, load='lda_model_1001_08_25_15_13') #
    #w2v = my_NLP_nn.makeW2V(data, load='wtov_model_1001_08_25_15_13') #


    #lda = x.makeLDA(data)
    #w2v = x.makeW2V(data)
###Whatever comes after here, to test and etc.
#plot frequencies of words
    #DF_data = DF_data[::10]
    #DF_data = DF_data.iloc[0:5]
    # print(DF_data.shape)

    #==============WICHTIG======================================================
    #preprocessing_tokenize(DF_data)
    #preprocessing_remove_stopwords(DF_data)
    #preprocessing_stem_words(DF_data)

    #DF_data.to_pickle("./DATA_FRAME_1001_08_31_07_37.pkl")
    #DF_data = pd.read_pickle("./DATA_FRAME_1001_08_31_07_37.pkl")
    #print(DF_data.head(10))
    #preprocessing_pos_tags(DF_data)


    #preprocessing_doc_to_bow(DF_data)

    #preprocessing_primitive_form(DF_data)
    #DF_data.to_pickle("./DATA_FRAME_1001_08_31_07_37.pkl")
    #print("Start Saving Pickle...")
    #DF_data.to_pickle("./DATA_FRAME_1001.pkl")
    #print("Saved!")
    #print(os.getcwd())
    DF_data = pd.read_pickle("./Main/DATA_FRAME_1001.pkl")
    #print(DF_data.head(2))
    #preprocessing_primitive_form(DF_data)

    #print(DF_data.primitive_form.values.tolist())
    #w2v = my_NLP_nn.makeW2V(DF_data.primitive_form.values.tolist())

    #print(DF_data.primitive_form.values.tolist())
    #print(np.concatenate(DF_data.only_nouns.values).tolist())

    #w2v = my_NLP_nn.makeW2V(DF_data.only_nouns.values.tolist(), load='wtov_model_1001_08_29_17_59')
    w2v = Word2Vec.load('./Data/Model/wtov_model_33001_08_26_09_18')
    # print(type(w2v.wv.vocab))

    #DF_data.load("./DATA_FRAME_1001.pkl")
    preprocessing_primitive_form(DF_data)
    #
    dictionary = preprocessing_generate_Dictionary(DF_data)
    # print(dictionary)
    #
    #keys = ['Aufgaben', 'Kenntnisse', 'Studium', 'Profil', 'Erfahrung', 'Wirtschaftsinformatik']
    #print(w2v.wv['Erfahrung'])
    keys = ['Sprachen', 'Fähigkeiten', 'Kenntnisse', 'Wissen']
    #print(np.concatenate(DF_data.only_nouns.values).tolist())
    embedding_clusters = []
    word_clusters = []
    i = 0
    for word in keys:
        embeddings = []
        words = []

        #for similar_word, _ in w2v.wv.most_similar(word, topn=30):
        for similar_word, _ in w2v.wv.most_similar(word, topn=200):
            if similar_word.lower() in np.concatenate(DF_data.only_nouns.values).tolist() and len(words) == 30:
                print(word, similar_word)
                i += 1
                words.append(similar_word)
                embeddings.append(w2v.wv[similar_word])
        embedding_clusters.append(embeddings)
        word_clusters.append(words)

    #print(embedding_clusters)
    embedding_clusters = np.array(embedding_clusters)
    n, m, k = embedding_clusters.shape
    tsne_model_2d = TSNE(perplexity=15, n_components=2, init='pca', n_iter=3500, random_state=32)
    embeddings_2d = np.array(tsne_model_2d.fit_transform(embedding_clusters.reshape(n * m, k))).reshape(n, m, 2)
    #
    print("soweit sogut")
    tsne_plot_similar_words('Similar words from Indeed', keys, embeddings_2d, word_clusters, 0.7, 'similar_words.png')





    #print(DF_data.head(2))
    #
    # cleansed_words_df = pd.DataFrame.from_dict(dictionary.token2id, orient='index')
    # cleansed_words_df.rename(columns={0: 'id'}, inplace=True)
    # cleansed_words_df['count'] = list(map(lambda id_: dictionary.dfs.get(id_), cleansed_words_df.id))
    # del cleansed_words_df['id']
    # cleansed_words_df.sort_values('count', ascending=False, inplace=True)
    # ax = word_frequency_barplot(cleansed_words_df)
    # ax.set_title("Document Frequencies (Number of documents a word appears in)", fontsize=16);

    #lda_get_good_tokens(DF_data)
    #non_ascii_words = remove_ascii_words(DF_data)
    #print(non_ascii_words)
    #print("Replaced {} words with characters with an ordinal >= 128 in the train data.".format(len(non_ascii_words)))
    #w2v_preprocessing(DF_data)
    #lda_get_good_tokens(DF_data)
    #remove_stopwords(DF_data)
    #stem_words(DF_data)

     # first five rows of dataframe
    #print(DF_data.head(5))

    # tokenized_only_dict = Counter(np.concatenate(DF_data.only_nouns.values))
    #
    # print(np.concatenate(DF_data.only_nouns.values))
    # #print(tokenized_only_dict)
    # #document_to_bow(DF_data)
    # tokenized_only_df = pd.DataFrame.from_dict(tokenized_only_dict, orient='index')
    # tokenized_only_df.rename(columns={0: 'count'}, inplace=True)
    # tokenized_only_df.sort_values('count', ascending=False, inplace=True)
    #
    # ax = word_frequency_barplot(tokenized_only_df)
    # ax.set_title("Word Frequencies", fontsize=16);
    # plt.show()

#end plot frequencies of words
    '''
    wv = w2v.wv

    print(wv.vocab)
    sorted_bow = sorted(bow.items(), key=lambda x: x[1], reverse=True)[:100]

    print(type(sorted_bow))
    print(sorted_bow)
    sorted_bow = dict(sorted_bow)

    #df = pd.DataFrame(bow.values(), index=bow.keys(), columns=['x', 'y'])
    #df.head(10)

    ax = word_frequency_barplot_pretty(sorted_bow)
    #ax = word_frequency_barplot(tokenized_only_df)
    ax.set_title("Word Frequencies", fontsize=16);
    plt.show()

    #plt.bar(range(len(sorted_bow)), list(sorted_bow.values()), tick_label=list(sorted_bow.keys()))
    #plt.show()

    zwischenergebnis = {}
    topics = ['Sprachen', 'Fähigkeiten', 'Kenntnisse', 'Wissen', 'Profil']
    for topic in topics:
        zwischenergebnis[topic] = wv.similar_by_word(topic)[:30]
    print(zwischenergebnis)

    vocab = list(wv.vocab)
    '''

    print("Ende")
    #vocab2 = [word[0] for word in nltk.pos_tag(nltk.word_tokenize(vocab)) if "NN" in word[1]]
    #print(vocab2)
    #X = w2v[vocab]
    #tsne = TSNE(n_components=2)
    #X_tsne = tsne.fit_transform(X)
    #df = pd.DataFrame(X_tsne, index=vocab, columns=['x', 'y'])
    #fig = plt.figure()
    #ax = fig.add_subplot(1, 1, 1)

    #ax.scatter(df['x'], df['y'])
    #for word, pos in df.iterrows():
    #    ax.annotate(word, pos)
    #plt.show()


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
