# collection processing: documents to vectors

import logging
logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.INFO)
from os import listdir
from os.path import isfile, join
from re import sub
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
import nltk
from nltk.corpus import stopwords
import numpy
import docopt
import os

args = docopt.docopt("""
    Usage:
        doc2vec.py [options] <collection> <indexFolder>
    
    Options:
        --dim NUM    Dimentions for the word embeddings generation [default: 300]
        --win NUM    Window size [default: 5]
        --minc NUM   Minimum frequency for concidered words [default: 5]
        
    """)
    
stoplist = stopwords.words('english')

dimConcept = int(args["--dim"])
win = int(args["--win"])
minc = int(args["--minc"])
print("Execution with :\n dim = %d \n win = %d \n minc = %d" %(dimConcept, win, minc))

doc_freq = {}  # dictionnaire des termes de la collection avec leurs fréquences globales
documents = {}  # dictionnaire des termes du document avec leurs fréquences
documentsVec = {}  # dictionnaire des vecteurs des documents
index = args["<indexFolder>"]+str(dimConcept)+'_win'+str(win)+'_min'+str(minc)
os.mkdir(index)
collection = args["<collection>"]

documentsDict = index +'/DocsIndex'
doc_freqTerm = index +"/collIndex"
vecDocs = index + "/doc2vecRepresentation"
matDoc = index +"/matDocuments"
corpusAsSentences = index + "/CorpusSentences.txt"  # où stocker le corpus sous-forme de phrases


#1 Traitement de la collection

print("Traitement de la collection ...\n %s" %(collection))
out_file = open(corpusAsSentences, 'w')
for f in listdir(collection): 
    if isfile(join(collection, f)):
        in_file = open(join(collection, f), 'r')
        text = in_file.read()
        in_file.close()
        if text != "":
            print("document: %s" % f)
            sents = nltk.sent_tokenize(text)
            doc_aux = []
            for s in sents:
                s=sub(r'[^a-zA-Z]+',' ', s) 
                s=s.lower()
                out_file.write("%s\n" %(s))
                d=[word for word in s.split()] # récupérer la liste des termes de la sentence s
                doc_aux+=d   # liste de tous les termes du document avec répétitions
            doc={w:doc_aux.count(w) for w in doc_aux}
            documents[f]=doc
            for word in doc:
                if(word not in doc_freq):
                    doc_freq[word]=1
                else:
                    doc_freq[word]+=1        
out_file.close()

print("Création d'index ...")
#sauvegarder dans un dossier
os.mkdir(documentsDict)
for f in documents:
    out_file=open(join(documentsDict,f), 'w')
    out_file.write(str(documents[f]))
    out_file.close()

out_file=open(doc_freqTerm, 'w')
out_file.write(str(doc_freq))
out_file.close()
print("corpus traité avec succès \n")

#2 apprentissage
print("Apprentissage ...")
sentences = LineSentence(corpusAsSentences)
model = Word2Vec(sentences, size=dimConcept, window=win, min_count=minc, workers=4) # lancer la génération du vocabulaire
model.save_word2vec_format(index+'/word2vec'+str(dimConcept)+'_win'+str(win)+'_min'+str(minc)+'.txt', fvocab=None, binary=False)
print("vocabulaire ok \n")

#3 representation des documents en vecteurs  
print("Documents to vectors ...")
os.mkdir(matDoc)
for f in listdir(collection): # ici lire tte la collection 
    doc=documents[f]
    tdoc=0 # taille du document à calculer :
    tdoc=sum(doc.values())
    vec_doc=numpy.zeros(dimConcept)
    mat_doc={}
    for word in doc:
        if(word in model.vocab):
            wordV=numpy.array(model[word])
            #vec_doc+=((doc[word]+1)/(doc_freq[word]+1))*(wordV/LA.norm(wordV))  somme pondérée des vecteurs normalisés des termes du document eq1'
            vec_doc+=((doc[word]+1)/(doc_freq[word]+1))*wordV
            mat_doc[word]=list(wordV*((doc[word]+1)/(doc_freq[word]+1)))
    documentsVec[f]=list(vec_doc/tdoc) # matrice de la collection : chaque vecteur document est une liste
    out_file=open(join(matDoc,f), 'w') # matrice d'un document : chaque mot est une liste
    out_file.write(str(mat_doc))
    out_file.close()
print("sauvegarde des représentations des documents ...")
#sauvegarder dans un dossier
os.mkdir(vecDocs)
for f in documentsVec:
    out_file=open(join(vecDocs,f), 'w')
    out_file.write(str(documentsVec[f]))
    out_file.close()
print("documents vector representation ok \n")


