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
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import as_completed


def process_docs(doc_grp, index_grp, if_pseudo_tf_grp, alpha_grp, w2v_model_grp, id2token_grp):
    """
    Compute the term frequencies at each document from the documents_grp
    :param documents_grp: list
    :param index_grp: object
    :param if_pseudo_tf_grp: bool
    :param alpha_grp: float
    :param w2v_model_grp: object
    :param id2token_grp: dict
    :return: dict
    """
    id2dtf_grp = {}
    doct_grp = [x for x in index_grp.document(doc_grp)[1] if x > 0]
    id2dtf_grp[doc_grp] = {}
    termFreq_d_grp = defaultdict(int)
    if if_pseudo_tf_grp:
        for t in doct_grp:
            termFreq_d_grp[t] = pseudo_frequency(id2token_grp, t, doct_grp, w2v_model_grp, alpha_grp)
    else:
        for t in doct_grp:  # compute frequency of each word in that doc
            termFreq_d_grp[t] += 1
    id2dtf_grp[doc_grp] = termFreq_d_grp
    return id2dtf_grp


def chunk_data(data, n):
    """
    Chunk a list of data into sub-lists of n elements each one
    :param data: list
    :param n: int
    :return: list
    """
    chunks = [data[x:x + n] for x in range(0, len(data), n)]
    return chunks


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
    w2v_model = Word2Vec.load_word2vec_format(config["w2v_model"], binary=config["binary"])  # parameterizable
    print("\nWord2Vec OK.")
    stop = set(stopwords.words('english')) if config["stoplist"] else []
    kind = "_ptf" if config["if_pseudo_tf"] else "_tf"

    print("Compute frequency in documents...")
    n_threads = config["num_threads"]
    documents_grps = chunk_data(documents, n_threads)  # split documents into groups of documents to process in parallel

    id2dtf = {}
    for documents_grp in tqdm(documents_grps):
        res_list = []
        with ThreadPoolExecutor(max_workers=len(documents_grp)) as executor:
            futures = [executor.submit(process_docs, document, index, config["if_pseudo_tf"], config["alpha"],
                                       w2v_model, id2token) for document in documents_grp]
            # print(futures)
            for future in as_completed(futures):
                # print(future.result())
                res_list.append(future.result())
        for res in res_list:
            id2dtf.update(res)
        # break

    np.save(join(config["output"], "id2dtf"+kind+".npy"), id2dtf)  # load with: np.load("file.npy").item()

    if config["cptf"]:
        print("Compute pseudo frequency in collection...")
        id2ctf = {}  # id2dtf[doc][w_id]: tf(w, doc)
        for t in tqdm(id2tf):
            id2ctf[t] = pseudo_frequency(id2token, t, list(id2tf.keys()), w2v_model, config["alpha"])
        np.save(join(config["output"]), "id2ctf" + kind + ".npy", id2ctf)

    print("Done.")
