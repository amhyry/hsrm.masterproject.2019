# -*- coding: iso-8859-1 -*-
from bs4 import BeautifulSoup # For HTML parsing
try:
    import urllib.request as urllib2
except ImportError:
    import urllib2 # Website connections
import re # Regular expressions
from time import sleep # To prevent overwhelming the server between connections
from collections import Counter # Keep track of our term counts
from nltk.corpus import stopwords # Filter out stopwords, such as 'the', 'or', 'and'
import pandas as pd
import sys
import json
import codecs
#import html2text

def text_cleaner(website):
    '''
    This function just cleans up the raw html so that I can look at it.
    Inputs: a URL to investigate
    Outputs: Cleaned text only
    '''
    try:
        site = urllib2.urlopen(website).read()#.decode('utf-8') # Connect to the job posting
        print(type(site))
    except:
        print("Fehler beim lesen")
        return None, None  # Need this in case the website isn't there anymore or some other weird connection problem

    soup_obj = BeautifulSoup(site, features='html.parser') # Get the html from the site

    #print(soup_obj.find())

    for script in soup_obj(["script", "style"]):
        script.extract() # Remove these two elements from the BS4 object


    body = soup_obj.find(id = 'jobDescriptionText')
    text = body.get_text()
    result_text = text
    lines = (line.strip() for line in text.splitlines())
    result_lines = []
    for line in lines:
        if line and line[-1] != ".":
            line = line + "."
        result_lines.append(line)
        #print(line, str(i))

    result_text = ' '.join(result_lines)
    return result_text

    #if "li" in soup_obj.contents:
    #    return tabel_li_extraktion(soup_obj)
    #else:
    #    return div_extraktion(soup_obj)

def tabel_li_extraktion(soup_obj):
    result_text = ""

    body = soup_obj.find(id = 'jobDescriptionText')

    for ul in body.find_all('ul'):
        result_text += ul.text + "\t\n\t"

    print("table: ",type(result_text))
    return result_text

def div_extraktion(soup_obj):
    result_text = ""

    body = soup_obj.find(id = 'jobDescriptionText')
    text = body.get_text()#find_all('div')[2].get_text()
    #lines = (line.strip() for line in text.splitlines()) # break into lines
    #chunks = (phrase.strip() for line in lines for phrase in line.split("  ")) # break multi-headlines into a line each

    #def chunk_space(chunk):
    #    chunk_out = chunk + ' ' # Need to fix spacing issue
    #    return chunk_out

    #text = ''.join(chunk_space(chunk) for chunk in chunks if chunk).encode('utf-8') # Get rid of all blank lines and ends of line

    result_text = text.decode('utf-8')
    #print("div: ",type(result_text))
    return result_text

def getURLs2Json(job_URLS):
    print(len(job_URLS))
    error_url = list()
    final_description = {}
    for j in range(0,len(job_URLS)):
        print(j)
        if job_URLS[j] is not None:
            value = text_cleaner(job_URLS[j])

        if value is not None:
            print("good")
            final_description[j] = value
        else:
            print("bad")
            error_url.append(job_URLS[j])

        if (j % 1000 == 1):
            #print(final_description[j])
            d = json.dumps(final_description)
            f = open('sampleFromDataCrowlerindeed'+str(j)+'.json', 'w')
            f.write(str(d))
            f.close()
            #with codecs.open('sampleFromDataCrowlerindeed'+str(j)+'.json', 'w', encoding='iso-8859-1') as f:
                #json.dump(final_description, f, ensure_ascii=False)#, encoding='utf8')
                #f.write(unicode(data))

    j = json.dumps(final_description)
    f = open('sampleFromDataCrowlerindeed.json', 'w')
    f.write(j)
    f.close()

    j = json.dumps(error_url)
    f = open('URLlistFromDataCrowlerindeed.json', 'w')
    f.write(j)
    f.close()

    print('Done with collecting the job postings!')
    print('There were', len(final_description), 'jobs successfully found.')

#seattle_info = skills_info(city='Frankfurt am Main')
#URLlist = getURLsFromIDEED()


with open('data/onlyLinks.json') as json_file:
    data = json.load(json_file)
print("URLliste fertig!")

getURLs2Json(data)



#sample = text_cleaner('https://de.indeed.com/rc/clk?jk=b065e9ad343976f0&fccid=14251a9dde577135&vjs=3')
#print(sample)
#print(sample[:20])
