from nltk.corpus import stopwords
import json, re, string, io


def umlauteConverter(text):
    #text = re.sub('[%s]' % re.escape(string.punctuation), ' ', text)
    text = text.encode("utf-8",'replace')
    text = text.decode('utf-8','replace')
    return text

def sentence_nomalizer(text):
    result = []
    text = umlauteConverter(text)
    text = [w for w in text.split() ]
    for value in text:
        if len(value) >= 1: result.append(value)
    return " ".join(result)

def converter(source, destination):
    #source = '../Data/sampleFromDataCrowlerindeed1001.json'
    #destination = '../Data/data_for_voc_1001.json'

    print('Start')

    with open(source, encoding='utf-8') as json_file:
        data = json.load(json_file)

    sentences = []
    i = 0
    j = 0
    for key in data:
        i+=1
        if isinstance(data[key],list) :
            j +=1
            continue

        dataAsList = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', data[key])
        for sentence in dataAsList:
            sentence = sentence_nomalizer(sentence)
            if sentence:# and 1<=10:
                sentences.append(sentence)
    print(j)

    with io.open(destination, 'w', encoding='utf8') as json_file:
        data = json.dumps(sentences, ensure_ascii=False)
        #data = json.dumps(sentences)
        json_file.write(str(data))

    print('Ende')
    print("testest")

if __name__ == '__main__':
    converter('../Data/sampleFromDataCrowlerindeed1001.json', '../Data/data_for_voc_1001.json')
