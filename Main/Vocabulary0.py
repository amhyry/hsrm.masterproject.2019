'''
Created on 24 May 2019

@author: arnoldriemer
'''

from collections import Counter
import json
import numpy as np
from nltk import SnowballStemmer
from nltk.corpus import stopwords
import re
import string
from tempfile import TemporaryFile
import hickle as hkl
import gensim
import polyglot
from polyglot.text import Text
import time
#from keras.preprocessing.text import Tokenizer
#from nltk import wordpunct_tokenize
#from nltk.stem import WordNetLemmatizer
#from pattern.de import singularize, conjugate, predicative

class Vocabulary:
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

    def __init__(self, json_file = None, vocabulary_size = None):
        self._initialized = False
        self.vocabulary_size = vocabulary_size
        self.BAG_OF_WORDS_FILE_FULL_PATH = "bag_of_words.vocab"
        self.word_2_index = {}
        self.index_2_word = {}

        self.word_2_index["START"] = 1
        self.word_2_index["UNKOWN"] = -1
        self.MaxSentenceLength = None
        self.json_file = json_file
        #self._initialize(Vocabulary.load_from_Json(json_file))
        if not self.json_file == None:
            self._initialize(Vocabulary.load_from_Json(json_file)) #load-real
        else:
            self._initialize(Vocabulary.load_from_Json()) #load-sample

    def _initialize(self, sentences_list):
        """Will initialize a File and create the Indexes
        Args:
            sentences_array : list
                List of sentences
        Raises:
            NotImplementedError
                You Dumb!
        """
        if sentences_list is None:
            raise NotImplementedError("You dumb!")
        self.sentence_list_normalizer(sentences_list) # auskommentieren
        self._prepare_Bag_of_Words_File(sentences_list)
        self.sentences_list = sentences_list
        #self._create_Vocab_Indexes()
        self.MaxSentenceLength = max([len(txt.split(" ")) for txt in sentences_list])
        self._initialized = True

    def Get_Top_Words(self, number_words = None):
        """Will initialize a File and create the Indexes
        Args:
            number_words : int
                Number of unique words in bag_of_words.vocab file.
        Returns:
            most_popular_words
        """
        if number_words == None:
            number_words = self.vocabulary_size
        chars = json.loads(open(self.BAG_OF_WORDS_FILE_FULL_PATH).read())
        counter = Counter(chars)
        most_popular_words = {key for key, _value in counter.most_common(number_words)}
        return most_popular_words

    def _prepare_Bag_of_Words_File(self,sentences_list):
        """ Do File
        Args:
            sentences_list : list
                sentences
        Returns:
            most_popular_words
        """
        counter = Counter()
        for s in sentences_list:
            counter.update(s.split(" "))
        with open(self.BAG_OF_WORDS_FILE_FULL_PATH, 'w') as output_file:
            output_file.write(json.dumps(counter))

    def _create_Vocab_Indexes(self):
        """ Do File
        Args:
            sentences_list : list
                sentences
        Returns:
            most_popular_words
        """
        if self.vocabulary_size == None:
            self.vocabulary_size = len(json.loads(open(self.BAG_OF_WORDS_FILE_FULL_PATH).read()))

        INPUT_WORDS = self.Get_Top_Words(self.vocabulary_size)
        #word to int
        print(len(INPUT_WORDS))


        for i, word in enumerate(INPUT_WORDS):
            self.word_2_index[word] = i
        #int to word
        for word, i in self.word_2_index.items():
            self.index_2_word[i] = word

    def _w_to_one_hot_v(self, word):
        """ Do File
        Args:
            word : list
                sentences
        Returns:
            vector
        """
        vector = np.zeros(self.vocabulary_size)
        vector[self.word_2_index[word]] = 1
        return vector

    def TransformSentencesToId(self, sentences):
        """ Do File
        Args:
            word : list
                sentences
        Returns:
            vector
        """
        vectors = []
        for r in sentences:
            words = r.split(" ")
            vector = np.zeros(len(words))

            for t, word in enumerate(words):
                if word in self.word_2_index:
                    vector[t] = self.word_2_index[word]
                else:
                    pass
                    #vector[t] = 2 #unk
            vectors.append(vector)
        return vectors

    #def TransformIdToSentences(self, sentences):
    def ReverseTransformSentencesToId(self, sentences):

        """ Do File
        Args:
            word : list
                sentences
        Returns:
            vector
        """
        vectors = []
        for r in sentences:
            words = r.split(" ")
            vector = np.zeros(len(words))

            for t, word in enumerate(words):
                if word in self.word_2_index:
                    vector[t] = self.word_2_index[word]
                else:
                    pass
                    #vector[t] = 2 #unk
            vectors.append(vector)
        return vectors

    def Get_SkipGram_Target_Words(self, sentences = None, Frame = 5):
        """Get Word Pairs in Frame
        Args:
            sentences: List
            Window_size: List

        Returns:
            A Integer with the size unique words in bag of words, unique items stored in bag_of_words.vocab file!
        """

        SKIP_GRAM_INPUT_WORD_LIST = []

        if sentences == None:
            sentences = self.sentences_list

        for sentence in sentences:
            sentence_tokenized = sentence.split(" ")

            for index, target_word in enumerate(sentence_tokenized):
                FROM_INDEX = max(index-Frame,0)
                TO_INDEX = min(index+1+Frame,len(sentence_tokenized))

                for contextWord in sentence_tokenized[FROM_INDEX:TO_INDEX]:
                    if contextWord != target_word:
                        SKIP_GRAM_INPUT_WORD_LIST.append((target_word,contextWord))

        return SKIP_GRAM_INPUT_WORD_LIST

    def Get_SkipGram_Target_Words_OneHotEncoded_XY(self, sentences = None, Frame = 5):
        """Get training_data
        Args:
            sentences: List
            Window_size: List

        Returns:
            x_train
            y_train

        """
        if sentences == None:
            sentences = self.sentences_list

        Skip_Gram_Target_Words = self.Get_SkipGram_Target_Words(sentences, Frame)

        X,Y = [],[]

        for target_word, context_word in Skip_Gram_Target_Words:
            X.append(self._w_to_one_hot_v(target_word))
            Y.append(self._w_to_one_hot_v(context_word))
        return np.asarray(X), np.asarray(Y)

    def get_bag_size(self):
        """Get Size of Bag of Words
        Returns:
            A Integer with the size unique words in bag of words, unique items stored in bag_of_words.vocab file!
        """
        chars = json.loads(open(self.BAG_OF_WORDS_FILE_FULL_PATH).read())
        return len(chars)

    @staticmethod
    def load_from_Json(json_file_path = 'sample-file.json'):
        """Load a List from a Json-File
        Args:
            json_file_path: string
                path to a json-file
        Returns:
            A list containing all data in the file.
            example:
             list = [ "machine learning engineers can build great data models",
                        "the more data you have the better your model",
                        "these predictions sound right, but it is all about your data",
                        "your data can provide great value"
                        "(...)"
                        "Smart-Products-Loesungen zur Vernetzung von Produkten im Internet of Things sprechen Sie an?",
                        "Sie wollen die neuesten Trends und Technologien nicht nur beobachten, sondern begleiten..."
                    ]

        """
        with open(json_file_path, encoding='utf-8') as json_file:
            data = json.load(json_file)
            return data

    @staticmethod
    def write_to_Json(sentences, json_file_path = 'sample-file.json'):
        """write a list to a Json-File
        Args:
            sentences: list
                   Example
                        list = [ "machine learning engineers can build great data models",
                        "the more data you have the better your model",
                        "these predictions sound right, but it is all about your data",
                        "your data can provide great value"
                        "(...)"
                        "Smart-Products-Loesungen zur Vernetzung von Produkten im Internet of Things sprechen Sie an?",
                        "Sie wollen die neuesten Trends und Technologien nicht nur beobachten, sondern begleiten..."
                    ]
            json_file_path: string
                path to a json-file
        """
        j = json.dumps(sentences)
        f = open(json_file_path, 'w')
        f.write(j)
        f.close()

    @staticmethod
    def stemma(sentence):
        list_sentences = []
        snowball = SnowballStemmer(language='german')
        for word in sentence.split():
            list_sentences.append(snowball.stem(word))
        return ' '.join(list_sentences)

    @staticmethod
    def stemma_List(sentences):
        for elem in sentences:
            sentences[sentences.index(elem)] = Vocabulary.stemma(elem)
        return sentences

    @staticmethod
    def umlauteConverter(text):
        text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
        #print('1 Print: ',text)
        text = text.encode('utf-8')
        text = text.replace(b'\xc3\x83\xc2\xa4', b'ae')
        text = text.replace(b'\xc3\x83\xc2\xbc', b'ue')
        text = text.replace(b'\xc3\x83\xc2\xb6', b'oe')
        text = text.replace(b'\xc3\x83', b'ss')
        text = text.replace(b'\xc3\xa2', b'')
        text = text.replace(b'\x82', b'')
        #print('2 Print: ',text)
        return text.decode('unicode_escape')


    @staticmethod
    def sentence_list_normalizer(sentences):
        result = {}
        print(len(sentences))
        i = 0
        j = 0
        for elem in sentences:
            j += 1
            if (j % 100000) == 1:
                print(j)
            try:
                if "de" == Text(elem).language.code:
                #time.sleep(30)
                #print(sentences.index(elem), elem)
                #print(elem.encode('utf-8'))
                    sentences[sentences.index(elem)] = Vocabulary.sentence_nomalizer(elem)
            except:
                i+=1

        print("Exceptions: ", i)
        return sentences

    @staticmethod
    def sentence_nomalizer(text):
        result = []
        text = Vocabulary.umlauteConverter(text)
        text = str(text).lower()
        stop_words = set(stopwords.words("german")) # Filter out any stop words
        #print('3 Print: ',text)
        text = Vocabulary.stemma(text)
        text = [w for w in text.split() if (not w in stop_words) ]

        #print('4 Print: ',text)
        for value in text:
            if len(value) >= 1: result.append(value)

        #print('5 Print: ',result)
        #time.sleep(20)
        return ' '.join(result)

    @staticmethod
    def dataOK(text):
        if re.search(r'\d', text): return False
        if re.search('(m/w/d)', text): return False
        if re.search('willkommen', text): return False
        if re.search('unser', text): return False
        if re.search('wir', text): return False
        if re.search('www.', text): return False
        if re.search('.de', text): return False
        if re.search('.com', text): return False
        if re.search('bewerb', text): return False
        return True

    @staticmethod
    def preprocessJson(data):
        sentences = []
        i = 0
        for key in data:
            i+=1
            #sentences.extend(tokenize.sent_tokenize(data[key]))
            dataAsList = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', data[key])
            for sentence in dataAsList:
                if Vocabulary.dataOK(sentence):
                    #print('1 Print: ',sentence)
                    sentence = Vocabulary.sentence_nomalizer(sentence)
                    #print('2 Print: ',sentence)
            #        if sentence and 1<=10:

                    sentences.append(sentence)
        return sentences

    def save_SkipGram_Target_Words_OneHotEncoded_XY_trainingdata(self, sentences_list, frame):
        x_train, y_train = self.Get_SkipGram_Target_Words_OneHotEncoded_XY(sentences_list, frame)
        self.data = { 'x_train' : x_train, 'y_train' : y_train}

        # Dump data to file
        hkl.dump( self.data, 'new_data_file.hkl' )
        print("Data succesfully stored in file")
        # Load data from file
    def load_skipGram_Target_words(self):
        self.data = hkl.load( 'new_data_file.hkl' )
        print("Data succesfully loaded")

    def getTrainData(self):
        return self.data['x_train'] , self.data['y_train']

    def save_SkipGram_Target_Words_OneHotEncoded_XY(self, sentences_list, frame):
        pass
        outfile_x_train = "outfile_x_train.npy"
        outfile_y_train = "outfile_y_train.npy"
        x_train, y_train = self.Get_SkipGram_Target_Words_OneHotEncoded_XY(sentences_list, frame)
        np.save(outfile_x_train, x_train)
        np.save(outfile_y_train, y_train)



def start():
    data = Vocabulary.load_from_Json("sampleFromDataCrowlerindeed12001.json")
    data = Vocabulary.preprocessJson(data)
    Vocabulary.write_to_Json(data, "inserate_normalisiert_VOC.json")



if __name__ == '__main__':

    #s = "Das ist ein Satz den ich mir ausgedacht habe"

    #data = Vocabulary.load_from_Json("sampleFromDataCrowlerindeed12001.json")
    #data = Vocabulary.preprocessJson(data)
    #Vocabulary.write_to_Json(data, "inserate_normalisiert.json")

    #SENTENCES = Vocabulary.load_from_Json('test_sentences_for_first_1000.json')
    #SENTENCES = Vocabulary.load_from_Json()
    #print(SENTENCES)

    #vocab = Vocabulary(None)
    #print(vocab.Get_Top_Words(26))


    #start()

    #vocab = Vocabulary()
    #vocab = Vocabulary('inserate_normalisiert_VOC.json')
    print("Start")
    vocab = Vocabulary('./Data/data_for_voc_33001.json')
    print("Daten gelagen")
    #Skip_Gram_Target_Words = vocab.Get_SkipGram_Target_Words(None, 2)
    #X_train, Y_train = vocab.Get_SkipGram_Target_Words_OneHotEncoded_XY(None,5)
    #print(X_train.shape)
    #print(Y_train.shape)
    #print(Skip_Gram_Target_Words)

    print("Vocabulary of {0} words".format(len(vocab.Get_Top_Words())))

    #Skip_Gram_Target_Words = vocab.Get_SkipGram_Target_Words(SENTENCES, WINDOW_SIZE=5)
    #Skip_Gram_Target_Words = vocab.Get_SkipGram_Target_Words(None,3)
    #for target, context in Skip_Gram_Target_Words:
    #    print("({0}, {1})".format(target,context))
    #x_train, y_train = vocab.Get_SkipGram_Target_Words_OneHotEncoded_XY(None, 3)
    vocab.save_SkipGram_Target_Words_OneHotEncoded_XY_trainingdata(None, 3)
    #vocab.load_skipGram_Target_words()
    #x_train, y_train = vocab.getTrainData()

    #print("Done")
    #print(vocab.word_2_index["haben"]) 72
    #print(vocab._w_to_one_hot_v("haben"))
    #print(x_train.shape)
    #print(y_train.shape)

    '''
            SENTENCES = [
             "machine learning engineers can build great data models",
             "the more data you have the better your model",
             "these predictions sound right, but it is all about your data",
             "your data can provide great value"
            ]  '''
