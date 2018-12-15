# -*- coding: utf-8 -*-
from __future__ import unicode_literals
"""
Tools for text extraction and analysis
"""

import collections
from os import listdir
from os.path import join
from nltk.stem.porter import PorterStemmer
from krovetzstemmer import Stemmer
import re
import nltk
from tqdm import tqdm
import ntpath
import gzip


"""
Write at the beginning of a file.
"""
class Prepender:

    def __init__(self, fname, mode='w'):
        self.__write_queue = []
        self.__f = open(fname, mode)

    def write(self, s):
        self.__write_queue.insert(0, s)

    def close(self):
        self.__exit__(None, None, None)

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        if self.__write_queue: 
            self.__f.writelines(self.__write_queue)
        self.__f.close()


"""
Add a line to the beginning of a file.
"""
def line_prepender(filename, line):
    with open(filename, 'r+') as f:
        content = f.read()
        f.seek(0, 0)
        f.write(line.rstrip('\r\n') + '\n' + content)


"""
It return the file name extracted from a path
"""
def path_leaf(path):
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)

"""
removes the file extension:
example: file.txt becomes file
return: file name without extension
"""
def remove_extension(file):
    return file.split('.')[0]


"""
Cleans the input text of special characters
return cleaned text
"""
def escape(input):
    return input.translate({
        ord('('): None,
        ord(')'): None,
        ord('\''): None,
        ord('\"'): None,
        ord('.'): ' ',
        ord(':'): ' ',
        ord('\t'): ' ',
        ord('/'): ' ',
        ord('&'): ' ',
        ord(','): ' ',
        ord('^'): ' ',
        ord('-'): ' ',
        ord('?'): ' ',
        ord('!'): ' ',
        ord('+'): ' ',
        ord(';'): ' ',
        ord('`'): None,
        ord('$'): None,
        ord('€'): None,
        ord('<'): ' ',
        ord('>'): ' ',
        ord('%'): ' ',
        ord('#'): ' ',
        ord('_'): ' ',
        ord('@'): ' ',
        ord('~'): ' ',
        ord('='): None,
        ord('*'): None,
    })


"""
Performs stemming according to the selected algo
return stemed text
"""
def stem(algo, text):
    if algo == "krovetz":
        stemmer = Stemmer()
        return stemmer.stem(text)
    elif algo == "porter":
        stm = PorterStemmer()
        return stm.stem(text)
    print("ERROR STEMMING: {t} unkown.".format(t=algo))


"""
Performs cleaning and stemming 
return cleaned and stemmed text
"""
def clean(text_to_clean, steming, stoplist):
    prog = re.compile("[_\-\(]*([A-Z]\.)*[_\-\(]*")
    tex = []
    for w in text_to_clean.split():
        if prog.match(w):
            w = w.replace('.', '')
        tex.append(w)
    text = " ".join(tex)
    text = ' '.join(escape(text).split())
    text = " ".join(nltk.word_tokenize(text))
    text = " ".join([stem(steming, w) for w in text.split() if w not in stoplist])
    return text


"""
Performs stemming and dot "." removal for a word 
return checked and stemmed word
"""
def check_word(word, stemming):
    prog1 = re.compile("([A-Z]\.)+")
    prog2 = re.compile("([A-Z]|[a-z])+(\_|\-)+([A-Z]|[a-z])*")
    if prog1.match(word):
        w=word.replace('.','')
        return [stem(stemming, w)]
    if prog2.match(word):
        words = [stem(stemming, w) for w in word.replace('_', ' ').replace('-', ' ').split()]
        return words
    return [stem(stemming, word)]


""" 
Extract TREC topics on the pathTop parameter as dictionnary. 
return dictionnary of queries.
ex: {0:"this is a text of the topic"}
"""
def extract_Trec_Topics(path_top):
    print("Extraction de : %s" % path_top)
    nb = 0
    topics = {}
    for f in listdir(path_top):
        f = open(join(path_top,f), 'r')   # Reading file
        l = f.readline().lower()
        # extracting topics
        while l != "":
            if l != "":
                num = 0
                while (l.startswith("<num>") == False) and (l != ""):
                    l = f.readline().lower()
                num = l.replace("<num>", "").replace("number:", "").replace("\n", "").replace(" ", "")
                while (l.startswith("<title>")==False) and (l!=""):
                    l = f.readline().lower()
                titre = ""
                while (not l.startswith("</top>")) and (not l.startswith("<desc>")) and (l!=""):
                    titre = titre + " " + l.replace("<title>", "")
                    l = f.readline().lower()
                if titre != "" and num != 0:
                    topics[str(int(num))] = titre.replace("\n", "").replace("topic:", "").replace("\t", " ")
                    nb += 1
            else: 
                print("Fin.\n ")
        f.close()
    return collections.OrderedDict(sorted(topics.items()))


""" 
Extract TREC million queries on the path_top parameter as dictionnary. 
return: dictionnary of queries.
ex: {0:"this is a text of the query"}
"""
def extract_trec_million_queries(path_top):
    topics = {}
    for f in listdir(path_top):
        print("Processing file ", f)
        if ".gz" not in f:
            input = open(join(path_top, f), 'r')   # Reading file
        else:
            input = gzip.open(join(path_top, f))
        for line in tqdm(input.readlines()):
            l = line.decode("iso-8859-15")
            query = l.strip().split(":")
            q = "mq" + str(int(query[0]))
            q_text = query[-1]  # last token string
            topics[q] = q_text
    return collections.OrderedDict(sorted(topics.items()))


"""
Read the qrels file to a dictionary.
Return dictionary of: {(q_id, d_id):rel} 
"""
def get_qrels(qrels_file):
        print("Reading Qrels ... ")
        qdr = {}
        with open(qrels_file, 'r') as qrels:
            for line in tqdm(qrels):
                if line is not None:
                    q = str(int(line.strip().split()[0]))
                    doc = line.strip().split()[2]
                    rel = int(line.strip().split()[3])
                    qdr[(q, doc)] = rel
        print("Qrels ok.")
        return collections.OrderedDict(sorted(qdr.items()))


"""
Computes a list of relevant judged documents in the trec-like qrels_file
return lists of relevant documents for each query
"""
def relDocs_perQuery(qrels_file):
    print("Relevant documents per-query ...")
    q_relDoc = {}
    with open(qrels_file) as qrels:
        for l in qrels:
            if l.strip().split()[0] not in q_relDoc:
                q_relDoc[l.strip().split()[0]] = []
            if int(l.strip().split()[3])==1:
                q_relDoc[l.strip().split()[0]].append(l.strip().split()[2])
    return q_relDoc


"""
Computes a list of judged documents per query in the trec-like qrels_file
return lists of documents for each query
"""
def docs_perQuery(qrels_file):
    print("Relevant documents per-query ...")
    q_relDoc = {}
    with open(qrels_file) as qrels:
        for l in qrels:
            if l.strip().split()[0] not in q_relDoc:
                q_relDoc[l.strip().split()[0]] = []
            q_relDoc[l.strip().split()[0]].append(l.strip().split()[2])
    return q_relDoc


"""
read unique values from column n in the file f
"""
def read_values(f, n):
    inf = open(f, "r")
    lines = inf.readlines()
    result = []
    for x in lines:
            result.append(x.split(' ')[n])
    inf.close()
    return set(result)


"""
devide list seq into num different sub-lists
return: list of folds
"""
def chunkIt(seq, num=5):
    avg = len(seq) / float(num)
    out = []
    last = 0.0
    while last < len(seq):
        out.append(seq[int(last):int(last + avg)])
        last += avg
    return out


def get_text_of_a_passage(doc_id, index, id2token, passage):
    """
    Get the text corresponding to a retrieved passage from indri
    :param doc_id: int
    :param id2token: list
    :param passage: list
    :return: string
    """
    doc = [x for x in index.document(doc_id)[1]]
    passage_txt = " ".join([id2token[x] for x in doc[passage[0]:passage[1]] if x != 0])
    return passage_txt


def sortFile(run):
    """
    Sort lines of a file run
    :param run: str
    :return:
    """
    f = open(run, 'r')
    out = f.read()
    f.close()
    del f
    lines = out.split('\n')
    del out
    tf = sorted(lines)
    ft = open(run + "_trie", 'w')
    for l in tqdm(tf):
        ft.write(l + '\n')
    ft.close()

