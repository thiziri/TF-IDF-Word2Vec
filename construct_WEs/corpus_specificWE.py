# Construct a corpus-specifique word embeddings
import gensim, logging, sys, pprint
logging.basicConfig(format='%(asctime)s : %(levelname)s: %(message)s', level=logging.INFO)
from os.path import join, isfile
import os.path
from gensim.models.word2vec import LineSentence
from gensim.models import Word2Vec
import numpy, ast
import docopt
import json
import pyndri
from nltk.corpus import stopwords

if __name__ == "__main__":
	print("\n----BEGIN----\n")
	args = docopt.docopt("""
	    Usage:
	        corpus_specificWE.py <configuration_file>
	        
	    """)


	config_file = sys.argv[1]
	config = json.load(open(config_file))
	print(json.dumps(config, indent=2))
	configuration = config["word2vec_config"].copy()
	print("Word2Vec will be trained with the following configuration:")
	print(json.dumps(configuration, indent=2))
	stopWordsList=set(stopwords.words('english')) if not bool(config["stop_file"]) else set([line.strip() for line in open(config["stop_file"]).readlines()])

	text_in = ""
	if bool(config['index']):
		print("Index reading ...")
		index = pyndri.Index(config["index"])
		_,id2token,_ = index.get_dictionary()
		documents = [document_id for document_id in range(index.document_base(), index.maximum_document())]

		text_in = os.path.join(config["out"],"Sentences.txt")
		intxt = open(text_in,"w") #construct a file of text lines, each line is a document content as one sentence
		for id_d in documents:
			_, terms = index.document(id_d)
			txt_line = ""
			if config["stopping"]:
				txt_line = " ".join([id2token[w] for w in terms if w!=0 and id2token[w] not in stopWordsList])
			else:
				txt_line = " ".join([id2token[w] for w in terms if w!=0])
			intxt.write(txt_line+"\n")
		intxt.close()
		print("Sentences of the corpus are in : {txt}".format(txt=text_in))

	elif bool(config["text_file"]):
		print("Text input from: ",config["text_file"])
		text_in = config["text_file"]
	else:
		print("Error reading text input.")
		exit()

	sentences = LineSentence(text_in)

	#Training word embeddings 
	
	model = Word2Vec(sentences,
		size=configuration["size"],
		alpha=configuration["alpha"],
		window=configuration["window"],
		min_count=configuration["min_count"],
		max_vocab_size=configuration["max_vocab_size"],
		sample=configuration["sample"],
		seed=configuration["seed"],
		workers=configuration["workers"],
		min_alpha=configuration["min_alpha"],
		sg=configuration["sg"],
		hs=configuration["hs"],
		negative=configuration["negative"],
		cbow_mean=configuration["cbow_mean"],
		iter=configuration["iter"],
		null_word=configuration["null_word"],
		trim_rule=configuration["trim_rule"],
		sorted_vocab=configuration["sorted_vocab"],
		batch_words=configuration["batch_words"])
	#, compute_loss=configuration["compute_loss"])
	# save the trained model:
	model_name = "skipgram" if configuration["sg"]==1 else "cbow"
	model.wv.save_word2vec_format(os.path.join(config["out"], config["data_name"]+model_name+"_wordEmbedding_dim"+str(configuration["size"])+"_win"+str(configuration["window"])+"_minCount"+str(configuration["min_count"])+".txt"),fvocab=None,binary=False)
	
	print("Finished.")



