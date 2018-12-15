from gensim.models import KeyedVectors as Word2Vec
from tqdm import tqdm

import sys
sys.path.append('../../for_matchZoo')
from tools4text import clean

def w2v_model_accuracy(model, questions):

    accuracy = model.accuracy(questions)
    
    sum_corr = len(accuracy[-1]['correct'])
    sum_incorr = len(accuracy[-1]['incorrect'])
    total = sum_corr + sum_incorr
    percent = lambda a: a / total * 100
    
    print('Total sentences: {}, Correct: {:.2f}%, Incorrect: {:.2f}%'.format(total, percent(sum_corr), percent(sum_incorr)))

# read the evaluation file, get it at:
# https://word2vec.googlecode.com/svn/trunk/questions-words.txt
questions = 'questions-words.txt'
evals = open(questions, 'r').readlines()
num_sections = len([l for l in evals if l.startswith(':')])
print('total evaluation sentences: {} '.format(len(evals) - num_sections))
# total evaluation sentences: 19544

# clean questions
print("Preprocessing questions ...")
out = open("pre_processed_"+questions, "w")
for l in tqdm(evals):
	if l.startswith(':'):
		out.write(l)
	else:
		pre_processed = clean(l, "krovetz", [])
		out.write(pre_processed+"\n")

questions = "pre_processed_"+questions

# load the pre-trained model of GoogleNews dataset (100 billion words), get it at:
# https://code.google.com/p/word2vec/#Pre-trained_word_and_phrase_vectors 
google = Word2Vec.load_word2vec_format('/home/thiziri/Documents/DOCTORAT/osirim_data/projets/iris/PROJETS/WEIR/collections/constructed/local_embeddings/Robust/Robust_skipgram_wordEmbedding_dim300_win10_minCount5.txt', binary=False) #('GoogleNews-vectors-negative300.bin', binary=True)
# test the model accuracy*
w2v_model_accuracy(google, questions)
#Total sentences: 7614, Correct: 74.26%, Incorrect: 25.74%


# *took around 1hr45mins on Mac Book Pro (3.1 GHz Intel Core i7)
