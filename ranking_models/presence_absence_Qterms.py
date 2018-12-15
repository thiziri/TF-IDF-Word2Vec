# Querying with classical IR and w2vec similarities

import pyndri
import sys
import json
from tqdm import tqdm
from tools4text import *
from function_tools import *
from pyndri import TFIDFQueryEnvironment, OkapiQueryEnvironment
from nltk.corpus import stopwords
from gensim.models import KeyedVectors as Word2Vec
from os.path import join
from pprint import pprint

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
    token2id, id2token, id2df = index.get_dictionary()
    id2tf = index.get_term_frequencies()
    avgdl = sum([index.document_length(document_id) for document_id in documents]) / len(documents)
    print("Index OK.")

    print("\nWord2vec loading ...")
    w2v_model = Word2Vec.load_word2vec_format(config["w2v_model"], binary=True)
    print("\nWord2Vec OK.")

    print("Cleaning topics ...")
    stop = set(stopwords.words('english')) if config["stoplist"] else []
    tops = {}
    topics = extract_Trec_Topics(config["topics"])
    for top in tqdm(topics):
        toptext = clean(topics[top], config["steming"], stop)
        tops[top] = toptext
    print("Topics read OK.\n")

    qio = index
    # set the query environment
    model = config["model"]
    if model == "tfidf":
        qio = TFIDFQueryEnvironment(index)
        mod = "tfidf_BM25"
    elif model == "okapi":
        qio = OkapiQueryEnvironment(index)
        mod = "okapi_BM25"
    else:
        mod = "LM"
    print("\n\nModel : ", mod)

    resRun = join(config["outputfolder"],
                  mod + config["collection_name"] + "_with_simCos") + "_alpha" + str(config["alpha"]) + \
             config["sim_model"]
    run = open(resRun, 'w')

    tops = {"298": tops["298"]}  # test with 1 query only
    resTop = get_top_results(tops, qio, config["num_res"])  # pre-rank results
    for top in tops:  # re-ranking of the top documents for the topic top
        # print("results ", top, resTop[top])
        print('topic {docno} : {doctext}'.format(docno=top, doctext=tops[top]))

        top_t = [token2id[w] for w in tops[top].split() if w in token2id and w not in stop]  # get topic word ids
        # print(top_t)

        # set the model parameters:
        parameters = {}
        if config["model"] == "tfidf":
            parameters = {'k1': config['k1'], 'b': config['b'], 'avgdl': avgdl, 'id2df': id2df}
        elif config["model"] == "okapi":
            parameters = {'k1': config['k1'], 'b': config['b'], 'avgdl': avgdl, 'k3': config['k3'], 'id2df': id2df}
        else:  # LM
            parameters = {'clmbda': config["c_lambda"], 'dlmbda': config["d_lambda"], 'id2tf': id2tf}

        # run the corresponding model:
        sc_bm25 = {}
        if config["sim_model"] == "eq2":
            sc_bm25 = allD_allQ_sim(resTop[top], index, top_t, id2token, w2v_model, len(documents), config["alpha"],
                                    config["model"], parameters, config["if_pseudo_tf"])
        elif config["sim_model"] == "eq3":
            sc_bm25 = allD_QinD_notQinD_sim(resTop[top], index, top_t, id2token, w2v_model, len(documents),
                                            config["alpha"], config["lambda"], config["model"], parameters,
                                            config["if_pseudo_tf"])
        elif config["sim_model"] == "eq4":
            sc_bm25 = QinD_QinDothers_allD_QnotInDsim(resTop[top], index, top_t, id2token, w2v_model, len(documents),
                                                      config["alpha"], config["lambda1"], config["lambda2"],
                                                      config["model"], parameters, config["if_pseudo_tf"])
        else:
            print("Model not set: parameter sim_model")
            exit(-1)
        for d in sc_bm25:
            run.write("{t}\tQ0\t{d}\t0\t{sc}\tBM25_impl\n".format(t=int(top), d=ext_Documents[d], sc=sc_bm25[d]))
        # break
    run.close()
    sortFile(resRun)


