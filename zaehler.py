# coding: utf8
import json

with open('./Data/sampleFromDataCrowlerindeed33001.json', encoding='utf-8') as json_file:
    data = json.load(json_file)

sentences = []
for key in data:
    if isinstance(data[key],str):
        sentences.append(data[key])

print(len(sentences))
#print(sentences[:4])
