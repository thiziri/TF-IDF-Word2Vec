# Querying with classical IR and w2vec similarities

import pyndri
import sys
import json
import numpy as np
from tqdm import tqdm
from tools4text import *
from function_tools import *
from nltk.corpus import stopwords
from gensim.models import KeyedVectors as Word2Vec
from os.path import join
from pprint import pprint
from collections import defaultdict

if __name__ == "__main__":
    config_file = sys.argv[1]
    with open(config_file, 'r') as conf:
        config = json.load(conf)
    pprint(config)

    print("\nIndex reading ... \n")
    index = pyndri.Index(config["indexFolder"])

    documents = [document_id for document_id in range(index.document_base(), index.maximum_document())]
    ext_Documents = {}
    for id_d in documents:
        ext_Documents[id_d], _ = index.document(id_d)
    id2tf = index.get_term_frequencies()
    _, id2token, id2df = index.get_dictionary()
    print("Index OK.")

    print("\nWord2vec loading ...")
    w2v_model = Word2Vec.load_word2vec_format(config["w2v_model"], binary=True)
    print("\nWord2Vec OK.")
    stop = set(stopwords.words('english')) if config["stoplist"] else []
    kind = "_ptf" if config["if_pseudo_tf"] else "_tf"

    print("Compute frequency in documents...")
    id2dtf = {}  # id2dtf[doc][w_id]: tf(w, doc)
    for doc in tqdm(documents):
        doct = [x for x in index.document(doc)[1] if x > 0]
        id2dtf[doc] = {}
        termFreq_d = defaultdict(int)
        if config["if_pseudo_tf"]:
            for t in doct:
                termFreq_d[t] = pseudo_frequency(id2token, t, doct, w2v_model, config["alpha"])
        else:
            for t in doct:  # compute frequency of each word in that doc
                termFreq_d[t] += 1
        id2dtf[doc] = termFreq_d
    np.save(join(config["output"]), "id2dtf"+kind+".npy", id2dtf)  # load with: np.load("file.npy").item()

    if config["cptf"]:
        print("Compute pseudo frequency in collection...")
        id2ctf = {}  # id2dtf[doc][w_id]: tf(w, doc)
        for t in tqdm(id2tf):
            id2ctf[t] = pseudo_frequency(id2token, t, list(id2tf.keys()), w2v_model, config["alpha"])
        np.save(join(config["output"]), "id2ctf" + kind + ".npy", id2ctf)

    print("Done.")
