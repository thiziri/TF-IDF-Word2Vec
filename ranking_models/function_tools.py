# function tools

from math import log
from tqdm import tqdm
import math
from collections import defaultdict


def manhattan_distance(start, end):
    """
    manhattan distance between 2 points start and end
    :param start: tuple
    :param end:  tuple
    :return: float
    """
    return sum(abs(e - s) for s, e in zip(start, end))


def sigmoid(x):
    """
    sigmoid value
    :param x: float
    :return: float
    """
    return 1.0 / (1.0 + math.exp(-20 * (x + 0.75)))


def cossim(wt, wd, puis, model):
    """
    cosine similarity between two words wt and wd, raised to power puis
    :param wt: str
    :param wd: str
    :param puis: int
    :param model: w2vec model
    :return:
    """
    # cos with w2v
    if wt == wd:
        return 1.0
    try:
        s = model.similarity(wt, wd)
    except:
        return 0.0

    if s < 0 and puis % 2 == 0:
        return -pow(s, puis)
    # print(wt, wd, pow(s, puis))
    return pow(s, puis)

def cossim0(wt, wd, puis, model):
    if wt == wd:
        return 1.0
    return 0.0


def tfidf_unit_score(df, nc, avgdl, tf, dl, k1=1.2, b=0.75):
    """
    compute the tf-idf score of a word
    :param df: int
    :param nc: int
    :param k1: float
    :param b: float
    :param avgdl: float
    :param tf: int
    :param dl: int
    :return: float
    """
    idf = log((nc - df + 0.0) / df + 0.0)
    return idf * ((tf * (k1 + 0)) / (tf + k1 * (1 - b + b * (dl / avgdl))))


def okapi_BM25_unit_score(df, nc, avgdl, tf, dl, qtf, k1=1.2, b=0.75, k3=7):
    """
    Okapi bm25 model
    :param df: int
    :param nc: int
    :param avgdl: float
    :param tf: int
    :param dl: int
    :param qtf: int
    :param k1: float
    :param b: float
    :param k3: int
    :return: float
    """
    idf = log((nc+1.0)/(df+0.5))
    qtw = 1.0
    if qtf > 0:
        qtw = ((idf*k3*qtf)/(qtf+k3))/qtf
    return qtw*idf*(tf*(k1+1))/(tf+k1*(1-b+b*(dl/avgdl)))


def lm_unit_score(clmbda, dlmbda, tfq_d, dl, tfq_c, cl):
    """
    Language model with JM smoothing
    :param clmbda: float
    :param dlmbda: float
    :param tfq_d: int
    :param dl: int
    :param tfq_c: int
    :param cl: int
    :return: float
    """
    return log(((dlmbda*(tfq_d/dl))/(clmbda*(tfq_c/cl)))+(1-clmbda-dlmbda)*tfq_d+1)  # with context and +1


def get_top_results(topics, qio, num):
    """
    pre-rank results of the different topics
    :param topics: dict
    :param qio: object
    :param num: int
    :return: dict
    """
    resTop = {}
    for top in tqdm(topics):
        # print('topic {docno} : {doctext}'.format(docno=top, doctext=topics[top]))
        r = []
        # get the top num documents with qio
        results = qio.query(topics[top], results_requested=int(num))
        for int_document_id, _ in results:
            r.append(int_document_id)
        resTop[top] = r
    print("pre-ranking ok.")
    return resTop


def pseudo_frequency(id2token, w, doc, model, alpha):
    """
    Computes the pseudo frequency based on sum of the w2vec similarities
    :param id2token: dict
    :param w: str
    :param doc: list
    :param model: object
    :param alpha: int
    :return: float
    """
    return sum([cossim(id2token[w], id2token[wd], alpha, model) for wd in doc])


def allD_allQ_sim(resTop, index, top_t, id2token, w2v_model, cl, alpha, unit_score, parameters, if_psedo_tf):
    """
    Similarity score using the equation [2] in the CORIA'18 paper
    :param resTop: list
    :param index: object
    :param top_t: list
    :param id2token: dict
    :param w2v_model: object
    :param cl: int
    :param alpha: int
    :param unit_score: str
    :param parameters: dict
    :param if_psedo_tf: bool
    :return: dict
    """
    sc_bm25 = {}
    # print(top_t)
    for d in resTop:
        termFreq_d = defaultdict(int)
        dl = index.document_length(d)
        doc = [x for x in index.document(d)[1] if x > 0]  # get document words
        # print(d, len(doc))
        sc_bm25[d] = 0.0
        if if_psedo_tf:
            for t in doc:
                termFreq_d[t] = pseudo_frequency(id2token, t, doc, w2v_model, alpha)
        else:
            for t in doc:  # compute frequency of each word in that doc
                termFreq_d[t] += 1

        for tq in top_t:
            for t_d in termFreq_d:
                if unit_score == "tfidf":
                    sc_bm25[d] += tfidf_unit_score(parameters["id2df"][t_d], cl, parameters['avgdl'], termFreq_d[t_d],
                                                   dl, parameters["k1"], parameters["b"]) * cossim(id2token[tq],
                                                                                                   id2token[t_d],
                                                                                                   alpha,
                                                                                                   w2v_model)
                elif unit_score == "okapi":
                    sc_bm25[d] += okapi_BM25_unit_score(parameters["id2df"][t_d], cl, parameters['avgdl'],
                                                        termFreq_d[t_d], dl, top_t.count(tq), parameters["k1"],
                                                        parameters["b"], parameters["k3"]) * cossim(id2token[tq],
                                                                                                    id2token[t_d],
                                                                                                    alpha,
                                                                                                    w2v_model)
                else:
                    sc_bm25[d] += lm_unit_score(parameters['clmbda'], parameters['dlmbda'], termFreq_d[t_d], dl,
                                                parameters["id2tf"][t_d], cl) * cossim(id2token[tq],
                                                                                       id2token[t_d],
                                                                                       alpha,
                                                                                       w2v_model)
                # break
        # print(sc_bm25[d])
        # break
    return sc_bm25


def allD_QinD_notQinD_sim(resTop, index, top_t, id2token, w2v_model, cl, alpha, lamda, unit_score, parameters,
                          if_psedo_tf):
    """
    Similarity score using the equation [3] in the CORIA'18 paper
    :param resTop: list
    :param index: object
    :param top_t: list
    :param id2token: dict
    :param w2v_model: object
    :param cl: int
    :param alpha: int
    :param lamda: float
    :param unit_score: str
    :param parameters: dict
    :param if_psedo_tf: bool
    :return: dict
    """
    sc_bm25 = {}
    for d in resTop:
        s_in = 0.0
        s_out = 0.0
        termFreq_d = defaultdict(int)
        dl = index.document_length(d)
        doc = [x for x in index.document(d)[1] if x > 0]
        sc_bm25[d] = 0.0
        if if_psedo_tf:
            for t in doc:
                termFreq_d[t] = pseudo_frequency(id2token, t, doc, w2v_model, alpha)
        else:
            for t in doc:  # compute frequency of each word in that doc
                termFreq_d[t] += 1

        termsIn = [w for w in set(top_t) & set(termFreq_d.keys())]

        for tq in top_t:
            if tq in termsIn:
                for t_d1 in termFreq_d:
                    if unit_score == "tfidf":
                        s_in += tfidf_unit_score(parameters["id2df"][t_d1], cl, parameters['avgdl'], termFreq_d[t_d1],
                                                 dl, parameters["k1"], parameters["b"]) * cossim(id2token[tq],
                                                                                                 id2token[t_d1],
                                                                                                 alpha,
                                                                                                 w2v_model)
                    elif unit_score == "okapi":
                        s_in += okapi_BM25_unit_score(parameters["id2df"][t_d1], cl, parameters['avgdl'],
                                                      termFreq_d[t_d1], dl, top_t.count(tq), parameters["k1"],
                                                      parameters["b"], parameters["k3"]) * cossim(id2token[tq],
                                                                                                  id2token[t_d1],
                                                                                                  alpha,
                                                                                                  w2v_model)
                    else:
                        s_in += lm_unit_score(parameters['clmbda'], parameters['dlmbda'], termFreq_d[t_d1], dl,
                                              parameters["id2tf"][t_d1], cl) * cossim(id2token[tq],
                                                                                      id2token[t_d1],
                                                                                      alpha,
                                                                                      w2v_model)
            else:
                for t_d in termFreq_d:
                    if unit_score == "tfidf":
                        s_out += tfidf_unit_score(parameters["id2df"][t_d], cl, parameters['avgdl'], termFreq_d[t_d],
                                                  dl, parameters["k1"], parameters["b"]) * cossim(id2token[tq],
                                                                                                  id2token[t_d],
                                                                                                  alpha,
                                                                                                  w2v_model)
                    elif unit_score == "okapi":
                        s_out += okapi_BM25_unit_score(parameters["id2df"][t_d], cl, parameters['avgdl'],
                                                       termFreq_d[t_d], dl, top_t.count(tq), parameters["k1"],
                                                       parameters["b"], parameters["k3"]) * cossim(id2token[tq],
                                                                                                   id2token[t_d],
                                                                                                   alpha,
                                                                                                   w2v_model)
                    else:
                        s_out += lm_unit_score(parameters['clmbda'], parameters['dlmbda'], termFreq_d[t_d], dl,
                                               parameters["id2tf"][t_d], cl) * cossim(id2token[tq],
                                                                                      id2token[t_d],
                                                                                      alpha,
                                                                                      w2v_model)
        sc_bm25[d] += lamda * s_in + (1 - lamda) * s_out
    return sc_bm25


def QinD_QinDothers_allD_QnotInDsim(resTop, index, top_t, id2token, w2v_model, cl, alpha, lamda1, lamda2, unit_score,
                                    parameters, if_psedo_tf):
    """
    Similarity score using the equation [4] in the CORIA'18 paper
    :param resTop: list
    :param index: object
    :param top_t: list
    :param id2token: dict
    :param w2v_model: object
    :param cl: int
    :param alpha: int
    :param lamda1: float
    :param lamda2: float
    :param unit_score: str
    :param parameters: dict
    :param if_psedo_tf: bool
    :return: dict
    """
    sc_bm25 = {}
    for d in resTop:
        s_bm25 = 0.0
        s_in = 0.0
        s_out = 0.0
        termFreq_d = defaultdict(int)
        dl = index.document_length(d)
        doc = [x for x in index.document(d)[1] if x > 0]
        sc_bm25[d] = 0.0
        if if_psedo_tf:
            for t in doc:
                termFreq_d[t] = pseudo_frequency(id2token, t, doc, w2v_model, alpha)
        else:
            for t in doc:  # compute frequency of each word in that doc
                termFreq_d[t] += 1

        termsIn = [w for w in set(top_t) & set(termFreq_d.keys())]

        for tq in top_t:
            if tq in termsIn:
                for t_d1 in termFreq_d:
                    if t_d1 == tq:
                        if unit_score == "tfidf":
                            s_bm25 += tfidf_unit_score(parameters["id2df"][t_d1], cl, parameters['avgdl'],
                                                     termFreq_d[t_d1],
                                                     dl, parameters["k1"], parameters["b"])
                        elif unit_score == "okapi":
                            s_bm25 += okapi_BM25_unit_score(parameters["id2df"][t_d1], cl, parameters['avgdl'],
                                                          termFreq_d[t_d1], dl, top_t.count(tq), parameters["k1"],
                                                          parameters["b"], parameters["k3"])
                        else:
                            s_bm25 += lm_unit_score(parameters['clmbda'], parameters['dlmbda'], termFreq_d[t_d1], dl,
                                                  parameters["id2tf"][t_d1], cl)
                    else:
                        if unit_score == "tfidf":
                            s_in += tfidf_unit_score(parameters["id2df"][t_d1], cl, parameters['avgdl'],
                                                     termFreq_d[t_d1],
                                                     dl, parameters["k1"], parameters["b"]) * cossim(id2token[tq],
                                                                                                     id2token[t_d1],
                                                                                                     alpha,
                                                                                                     w2v_model)
                        elif unit_score == "okapi":
                            s_in += okapi_BM25_unit_score(parameters["id2df"][t_d1], cl, parameters['avgdl'],
                                                          termFreq_d[t_d1], dl, top_t.count(tq), parameters["k1"],
                                                          parameters["b"], parameters["k3"]) * cossim(id2token[tq],
                                                                                                      id2token[t_d1],
                                                                                                      alpha,
                                                                                                      w2v_model)
                        else:
                            s_in += lm_unit_score(parameters['clmbda'], parameters['dlmbda'], termFreq_d[t_d1], dl,
                                                  parameters["id2tf"][t_d1], cl) * cossim(id2token[tq],
                                                                                          id2token[t_d1],
                                                                                          alpha,
                                                                                          w2v_model)
            else:
                for t_d in termFreq_d:
                    if unit_score == "tfidf":
                        s_out += tfidf_unit_score(parameters["id2df"][t_d], cl, parameters['avgdl'], termFreq_d[t_d],
                                                  dl, parameters["k1"], parameters["b"]) * cossim(id2token[tq],
                                                                                                  id2token[t_d],
                                                                                                  alpha,
                                                                                                  w2v_model)
                    elif unit_score == "okapi":
                        s_out += okapi_BM25_unit_score(parameters["id2df"][t_d], cl, parameters['avgdl'],
                                                       termFreq_d[t_d], dl, top_t.count(tq), parameters["k1"],
                                                       parameters["b"], parameters["k3"]) * cossim(id2token[tq],
                                                                                                   id2token[t_d],
                                                                                                   alpha,
                                                                                                   w2v_model)
                    else:
                        s_out += lm_unit_score(parameters['clmbda'], parameters['dlmbda'], termFreq_d[t_d], dl,
                                               parameters["id2tf"][t_d], cl) * cossim(id2token[tq],
                                                                                      id2token[t_d],
                                                                                      alpha,
                                                                                      w2v_model)
        sc_bm25[d] += (lamda1 * s_bm25) + (lamda2 * s_in) + (1 - lamda1 - lamda2) * s_out
    return sc_bm25
