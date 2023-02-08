import sys
import os
import os.path
import json
#from lexrank import STOPWORDS, LexRank
from lexrank_bert.summarizer import LexRankBERT
from nltk.tokenize import word_tokenize, sent_tokenize
import rouge
from random import shuffle
from tqdm import tqdm
import argparse

def load_train(train_file, max_pairs=10000, load_dec=False):
	data = json.load(open(train_file))
	qa_pairs = []
	for entity in tqdm(data):
		for qa in entity["qa_pair"]:
			if not load_dec:
				q = qa["question"]
				a = qa["answer"]
				qa_pairs.append("{} {}".format(q, a))
			else:
				qa_pairs.append(qa["declaritive"])
	shuffle(qa_pairs)
	if max_pairs != -1:
		qa_pairs = qa_pairs[:max_pairs]
	return qa_pairs

def load_test(test_file, load_dec=False):
	data = json.load(open(test_file))
	qa_pairs_per_entity = {}
	for entity in tqdm(data):
		asin = entity["asin"]
		qa_pairs = []
		for qa in entity["qa_pair"]:
			if not load_dec:
				q = qa["question"]
				a = qa["answer"]
				qa_pairs.append("{} {}".format(q, a))
			else:
				qa_pairs.append(qa["declaritive"])
		qa_pairs_per_entity[asin] = {"qa_pairs":qa_pairs, "gold":entity["summary"]}
	return qa_pairs_per_entity

def get_statistics(data, tokenizer):
	nums = []
	for asin, info in data.items():
		nums.append(len(tokenizer(info["gold"][0])))
	return sum(nums)/len(nums)

def truancate_summary(ranked_qas, avg_num_dev, tokenizer):
	summary = ""
	summary_length = 0
	i = 0
	while i < len(8): #and summary_length < avg_num_dev:
		summary += "     " + ranked_qas[i]
		summary_length += len(tokenizer(ranked_qas[i]))
		i += 1
	return summary

def run_lexrank(train_file, dev_file, test_file, load_dec=False, use_bert=False, threshold=None, train_max_pairs=10000):
	print('Load data...')
	train_data = load_train(train_file, max_pairs=train_max_pairs, load_dec=load_dec)
	dev_data = load_test(dev_file, load_dec=load_dec)
	test_data = load_test(test_file, load_dec=load_dec)
	tokenizer = word_tokenize

	print('Computing IDF scores...')
	if use_bert:
		lxr = LexRankBERT()
	else:
		lxr = LexRank(train_data, stopwords=STOPWORDS['en'])

	print("Get statistics from dev...")
	avg_num_dev = get_statistics(dev_data, tokenizer)

	print('Summarizing...')
	gens = []
	for asin, info in tqdm(test_data.items()):
		qa_pair_scores = lxr.rank_sentences(info["qa_pairs"], threshold=threshold)
		ranked_qas = [qa for qa, score in 
	                sorted(zip(info["qa_pairs"], qa_pair_scores), reverse=True, key=lambda x:x[1])]
		summary = truancate_summary(ranked_qas, avg_num_dev, tokenizer)
		gens.append({"pred": summary, "ref": info["gold"]})

	print('Saving...')
	with open("../generation/{}lexrank{}.json".format("led_" if load_dec else "", "_bert" if use_bert else ""), "w") as file:
		file.write(json.dumps(gens, indent=4))
		file.close()

	#evaluator = rouge.Rouge(metrics=["rouge-n", "rouge-l"], max_n=2, limit_length=False,
	#                    apply_avg=True, stemming=True)
	#scores = evaluator.get_scores(gen, refs)
	#for metric in ("rouge-1", "rouge-2", "rouge-l"):
	#	for measure in ("p", "r", "f"):

if __name__=="__main__":
	# parser = argparse.ArgumentParser()
	# parser.add_argument("--data_dir", default='data/', type=str)
	# parser.add_argument("--train_max_pairs", default=10000, type=int)
	# parser.add_argument("--use_bert", type=str2bool, nargs='?',const=True,default=False)
	# parser.add_argument("--use_declaritive", type=str2bool, nargs='?',const=True,default=False)
	# args = parser.parse_args()

	data_dir = "data/"

	train_file = os.path.join(data_dir, "qa_summary_filtered_train_dec.json")
	dev_file = os.path.join(data_dir, "qa_summary_filtered_val_dec.json")
	test_file = os.path.join(data_dir, "qa_summary_filtered_test_dec.json")
	train_max_pairs=10000

	run_lexrank(train_file, dev_file, test_file, load_dec=False, use_bert=True, train_max_pairs=train_max_pairs)
