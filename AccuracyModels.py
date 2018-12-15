# evaluation of word2vec models accuracy

import docopt
from os import listdir
from os.path import isfile, join
from gensim.models import Word2Vec
import patoolib
import os
import shutil


args = docopt.docopt("""
    Usage:
        AccuracyModels.py <w2vec_modelsFolder> <outputFolder> <questionsWords>
       
    """)
    
modelsFolder=args["<w2vec_modelsFolder>"]
outputFolder=args["<outputFolder>"]
questionsWords=args["<questionsWords>"]

for f in listdir(modelsFolder):
    mod = join(modelsFolder, f)
    test = True
    try :
        test = patoolib.extract_archive(mod)
    except:
        test = False
    if test:
        if isfile(test):
            modC = test
        else:
            modC = listdir("./"+test)[0]
    else:
        modC = f
    
    print("Evaluation de ", modC)
    
    # Load the model
    
    print("Loading the word2vec Model")
    model = Word2Vec.load_word2vec_format(modC, binary=False)  # text format

    print("Evaluation ...")
    res = model.accuracy(questionsWords)
    print("Accuracy OK \n")

    cat = [res[i]['section'] for i in range(len(res))]
    correct = 0
    incorrect = 0
    acc = {}
    for i in range(len(res)):
        correct += len(res[i]['correct'])
        incorrect += len(res[i]['incorrect'])
        acc[cat[i]] = (correct/float(correct+incorrect))*100

    # Save the results for f
    print("Writing results ...")
    pathAcc = join(outputFolder, modC+".accuracy")
    out = open(pathAcc, 'w')
    out.write(str(acc))
    out.close()
    if test:
        if isfile(modC):
            os.remove(modC)
        else:
            shutil.rmtree(modC)
print("Job finished :)")
