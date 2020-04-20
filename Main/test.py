# coding: utf8
import json

#import warnings
#warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
from gensim.models import TfidfModel
from gensim.models import Word2Vec
from gensim.corpora import Dictionary
from nltk.corpus import brown
from gensim.summarization import keywords
from gensim import corpora
from gensim.models.ldamodel import LdaModel
import multiprocessing
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import pprint
import sys
import locale
import io
import JsonToSentencesConverter as Crawler
import os
from datetime import datetime
from sklearn.manifold import TSNE
from sklearn.feature_extraction.text import CountVectorizer
import lda
import numpy as np
import bokeh.plotting as bp
from bokeh.plotting import save, show
from bokeh.io import show
from bokeh.models import HoverTool
import matplotlib.pyplot as plt
import matplotlib.cm as cm
#from . import Crawler
import matplotlib.patheffects as PathEffects
import tempfile
import imageio
import shutil
from gensim.models import LsiModel
from gensim.models.coherencemodel import CoherenceModel



sys.stdout = io.TextIOWrapper(sys.stdout.detach(), encoding = 'utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.detach(), encoding = 'utf-8')


#from polyglot.text import Text
#from polyglot.downloader import downloader

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
    #stop_free = " ".join([i for i in doc.lower().split() if i not in stop])
    stop_free = " ".join([i for i in doc.split() if i.lower() not in stop])
    #punct_free = ''.join([ch for ch in stop_free if ch not in exclude])
    #only_noun = " ".join(word[0] for word in nltk.pos_tag(nltk.word_tokenize(punct_free)) if "NN" in word[1])
    normalized = " ".join([lemma.lemmatize(word) for word in stop_free.split()])
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

def giveLdaTopicsWithoutStopwords(lda, data, dict):
    #print(lda.print_topics(num_topics=3, num_words=10))
    #print(stop)
    tmp = lda.print_topics(num_topics=3, num_words=100)
    #for ele in tmp:
        #print (ele)

    for i in lda.show_topics(num_words=4):
        print(i)

    import pyLDAvis.gensim
    lda_display = pyLDAvis.gensim.prepare(lda, data, dict)
    lda_display.save_json()


def printAs3DPlot(data):
    import plotly.plotly as py
    import plotly.graph_objs as go
    trace1=go.Scatter3d(x=Xe, y=Ye, z=Ze, mode='lines',line=dict(color='rgb(125,125,125)', width=1), hoverinfo='none')
    trace2=go.Scatter3d(x=Xn, y=Yn, z=Zn, mode='markers', name='actors', marker=dict(symbol='circle', size=6, color=group, colorscale='Viridis', line=dict(color='rgb(50,50,50)', width=0.5)), text=labels,hoverinfo='text')


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

def tsne_plot_similar_words_png(title, embedding_clusters, a, filename):
    plt.figure(figsize=(16, 9))
    colors = cm.rainbow(np.linspace(0, 1, len(embedding_clusters)))
    i = 1
    for embeddings, color in zip(embedding_clusters, colors):
        x = embeddings[:, 0]
        y = embeddings[:, 1]
        plt.scatter(x, y, c=color, alpha=a)
        plt.text(x.mean(), y.mean(), str(i), color='white', weight='bold', fontsize=13, path_effects=[PathEffects.withStroke(linewidth=3, foreground="black", alpha=0.7)])
        i += 1
    plt.title(title)
    plt.grid(True)
    plt.xlim(-200, 200)
    plt.ylim(-200, 200)
    plt.savefig(filename, format='png', dpi=150, bbox_inches='tight')

def compute_coherence_values(dictionary, doc_term_matrix, doc_clean, stop, start=2, step=3):
    """
    Input   : dictionary : Gensim dictionary
              doc_term_matrix : Gensim corpus
              doc_clean : List of input texts
              stop : Max num of topics
    purpose : Compute c_v coherence for various number of topics
    Output  : model_list : List of LSA topic models
              coherence_values : Coherence values corresponding to the LDA model with respective number of topics
    """
    coherence_values = []
    model_list = []
    for num_topics in range(start, stop, step):
        # generate LSA model
        model = LsiModel(doc_term_matrix, num_topics=num_topics, id2word = dictionary)  # train model
        model_list.append(model)
        coherencemodel = CoherenceModel(model=model, texts=doc_clean, dictionary=dictionary, coherence='c_v')
        coherence_values.append(coherencemodel.get_coherence())
    return model_list, coherence_values

if __name__ == '__main__':
    print('Start')
    #parameters
    smalldata = True # or false if you want to run no test
    source = ""
    destination = ""
    dir_up = ""
    modelstr = "_33001"

    if(os.name == "posix"):
        dir_up = '..'

    if(os.name == "nt"):
        dir_up = '.'
    #print("Path: " + os.name + dir_up)

    if smalldata == True:
        modelstr = "_1001_" + datetime.now().strftime('%m_%d_%H_%M')
        source = dir_up + '/Data/sampleFromDataCrowlerindeed1001.json'
        destination =  dir_up + '/Data/data_for_voc_1001.json'
    else:
        modelstr = "_33001_" + datetime.now().strftime('%m_%d_%H_%M')
        source = dir_up + '/Data/sampleFromDataCrowlerindeed33001.json'
        destination = dir_up + '/Data/data_for_voc_33001.json'

    #no need to to this, because 33001 and 1001 are already converted -> only needed for new data
    #Crawler.converter(source, destination)
    with open(destination, encoding='utf-8') as json_file:
        data = json.load(json_file)

    #data = brown.sents()

    #print('daten geladen')
    #for a in data:
        #print(a.decode("utf8","replace"))
    #doc_complete = dict2listConverter(data)
    #only_str_docs = removeEmptyListsFromDocs(doc_complete)
    #print(downloader.supported_languages_table("pos2"))
    #print(data[:3])
    doc_clean = [clean(doc).split() for doc in data]
    #print(doc_clean)
    #print(doc_clean)
    #print("Dokumente gesaeubert")
    dictionary = corpora.Dictionary(doc_clean)
    #print("Dictionary angelegt")
    #print(dictionary)
    doc_term_matrix = [dictionary.doc2bow(doc) for doc in doc_clean]
    #corpus = [dictionary.doc2bow(doc) for doc in doc_clean]
    #print("Matrix erstellt")
    #print(doc_term_matrix)
    #start,stop,step=2,12,1
    #model_list, coherence_values = compute_coherence_values(dictionary, doc_term_matrix,doc_clean,stop, start, step)
    # Show graph
    #x = range(start, stop, step)
    #plt.plot(x, coherence_values)
    #plt.xlabel("Number of Topics")
    #plt.ylabel("Coherence score")
    #plt.legend(("coherence_values"), loc='best')
    #plt.show()


    ################TNSE######################
    w2v = Word2Vec.load(dir_up+'/Data/Model/wtov_model_33001_08_26_09_18')
    #keys = ['Sprachen', 'Fähigkeiten', 'Kenntnisse', 'Wissen', 'Programmieren']
    print(w2v.wv['Erfahrung'])
    keys = ['Aufgaben', 'Kenntnisse', 'Studium', 'Profil', 'Erfahrung', 'Wirtschaftsinformatik']
    embedding_clusters = []
    word_clusters = []
    for word in keys:
        embeddings = []
        words = []
        for similar_word, _ in w2v.wv.most_similar(word, topn=30):
        #for similar_word, _ in w2v.wv.most_similar(word, topn=200):
            words.append(similar_word)
            embeddings.append(w2v.wv[similar_word])
        embedding_clusters.append(embeddings)
        word_clusters.append(words)

    #print(embedding_clusters)
    embedding_clusters = np.array(embedding_clusters)
    n, m, k = embedding_clusters.shape
    tsne_model_2d = TSNE(perplexity=15, n_components=2, init='pca', n_iter=3500, random_state=32)
    embeddings_2d = np.array(tsne_model_2d.fit_transform(embedding_clusters.reshape(n * m, k))).reshape(n, m, 2)


    #tsne_plot_similar_words('Similar words from Indeed', keys, embeddings_2d, word_clusters, 0.7, 'similar_words.png')


    #images = []
    #for i in range(1, 30):
    #    fname = os.path.join(dirpath, str(i) + '.png')
    #    tsne_model_2d_gif = TSNE(perplexity=i, n_components=2, init='pca', n_iter=3500, random_state=32)
    #    embeddings_2d_gif = np.array(tsne_model_2d_gif.fit_transform(embedding_clusters.reshape(n * m, k))).reshape(n, m, 2)
    #    tsne_plot_similar_words_png('Vizualizing similar words from Indeed using t-SNE (perplexity={})'.format(i), embeddings_2d_gif, 0.6, fname)
    #    images.append(imageio.imread(fname))
    #imageio.mimsave("2d1.gif", images, duration = 0.5)
    #shutil.rmtree(dirpath)
    #ldamodel = LdaModel(doc_term_matrix, num_topics=3, id2word = dictionary, passes=50)
    #ldamodel.save(dir_up + '/Data/Model/lda_model_noun_'+modelstr)
    ###################################
    # = LdaModel.load(dir_up + '/Data/Model/lda_model_33001_08_25_12_43')
    #ldamodel = LdaModel.load(dir_up + '/Data/Model/lda_model_noun__33001_08_26_11_29')

    #from sklearn.datasets import fetch_20newsgroups
    #remove = ('headers', 'footers', 'quotes')

    # fetch train and test data
    #newsgroups_train = fetch_20newsgroups(subset='train', remove=remove)
    #newsgroups_test = fetch_20newsgroups(subset='test', remove=remove)

    # a list of 18,846 cleaned news in string format
    # only keep letters & make them all lower case
    #news = [' '.join(raw.lower().split()) for raw in newsgroups_train.data + newsgroups_test.data]

    #print('Start')

    #with open('./Data/sampleFromDataCrowlerindeed1001.json', encoding='utf-8') as json_file:
    #    data = json.load(json_file)

    #sentences = []
    #for key in data:
    #    if isinstance(data[key],str):
    #        sentences.append(data[key])

    #print(news[1])
    #print(sentences)
    #news = [clean(doc)for doc in sentences]
    #print(news)
    #n_topics = 10 # number of topics
    #n_iter = 500 # number of iterations
    #data = [' '.join(doc) for doc in data]
    #cvectorizer = CountVectorizer(min_df=5)
    #cvz = cvectorizer.fit_transform(news)

    #print("train an LDA model")
    #lda_model = lda.LDA(n_topics=n_topics, n_iter=n_iter)
    #X_topics = lda_model.fit_transform(cvz)

    #print("TSNE")
    #tsne_model = TSNE(n_components=2, verbose=1, random_state=0, angle=.99, init='pca')
    #tsne_lda = tsne_model.fit_transform(X_topics)

    #print("x = ",tsne_lda[:, 0])
    #print("y = ",tsne_lda[:, 1])

    #print(ldamodel.print_topics(num_topics=3, num_words=10))
    #n_top_words = 5 # number of keywords we show
    #colormap = np.array(["#1f77b4", "#aec7e8", "#ff7f0e", "#ffbb78", "#2ca02c","#98df8a", "#d62728", "#ff9896", "#9467bd", "#c5b0d5"])#,"#8c564b", "#c49c94", "#e377c2", "#f7b6d2", "#7f7f7f","#c7c7c7", "#bcbd22", "#dbdb8d", "#17becf", "#9edae5"])

    #_lda_keys = []
    #for i in range(X_topics.shape[0]):
    #    _lda_keys +=  X_topics[i].argmax(),

    #print(_lda_keys)
    #topic_summaries = []
    #topic_word = lda_model.topic_word_  # all topic words
    #vocab = cvectorizer.get_feature_names()
    #for i, topic_dist in enumerate(topic_word):
    #    topic_words = np.array(vocab)[np.argsort(topic_dist)][:-(n_top_words + 1):-1] # get!
    #    topic_summaries.append(' '.join(topic_words)) # append!

    #title = '20 newsgroups LDA viz'
    #num_example = len(X_topics)

    #plot_lda = bp.figure(plot_width=1400, plot_height=1100, title=title,tools="pan,wheel_zoom,box_zoom,reset,hover,previewsave", x_axis_type=None, y_axis_type=None, min_border=1)
    #SOURCE = bp.ColumnDataSource({"content": news[:num_example],"topic_key": _lda_keys[:num_example]})
    #SOURCE = bp.ColumnDataSource({"x": news[:num_example],"y": _lda_keys[:num_example]})

    #plot_lda.scatter(x=tsne_lda[:, 0], y=tsne_lda[:, 1], color=colormap[_lda_keys][:num_example],source=SOURCE)

    # randomly choose a news (within a topic) coordinate as the crucial words coordinate
    #topic_coord = np.empty((X_topics.shape[1], 2)) * np.nan
    #for topic_num in _lda_keys:
    #    if not np.isnan(topic_coord).any():
    #        break
    #    topic_coord[topic_num] = tsne_lda[_lda_keys.index(topic_num)]

    # plot crucial words
    #for i in range(X_topics.shape[1]):
    #    plot_lda.text(topic_coord[i, 0], topic_coord[i, 1], [topic_summaries[i]])

    # hover tools
    #hover = plot_lda.select(dict(type=HoverTool))
    #hover.tooltips = {"content": "@content - topic: @topic_key"}

    #show(plot_lda)
    # save the plot
    #save(plot_lda, '{}.html'.format(title))



    #dictionary = Dictionary.load(dir_up + '/Data/bow_33001_08_26_14_30')
    #for
    #print(dictionary.token2bow)
    #giveLdaTopicsWithoutStopwords(ldamodel, corpus, dictionary) #what is this? and where can i find it?

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



    ###################################
    #EMB_DIM = 300
    #data_clean = [doc.split() for doc in data]
    #print(doc_clean[1])
    #print(multiprocessing.cpu_count())
    #w2v = Word2Vec(doc_clean, size=EMB_DIM, window=5, min_count=5, negative=15, iter=10, workers=multiprocessing.cpu_count())
    #w2v.save(dir_up+'/Data/Model/wtov_model'+modelstr)
    #w2v = Word2Vec.load(dir_up+'/Data/Model/wtov_model_33001_08_26_09_18')
    #wv = w2v.wv

    #print(wv.vocab)

    #with open('./Data/keywords.json', 'w', encoding='utf8') as json_file:
    #    data = json.dumps(kwords, ensure_ascii=False)
    #    json_file.write(str(data))

    #result = wv.similar_by_word('Sprachen')
    #print("Most Similar to 'Sprachen':\n", result[:20])


    #zwischenergebnis = {}
    #topics = ['Sprachen', 'Fähigkeiten', 'Kenntnisse', 'Wissen', 'Programmieren']
    #for topic in topics:
    #    zwischenergebnis[topic] = wv.similar_by_word(topic)[:3]
    #    print("Topic: ", topic)
    #    print(zwischenergebnis[topic])
    #print(zwischenergebnis)

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

    #import sys

    #print(sys.version)


    #result =
