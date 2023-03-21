import os
import json
import hashlib
import pandas as pd
from pprint import pprint
from tqdm import tqdm

'''
load original amazon_qa data from json file
save in one json file
'''
def load_amazon_qa():
	qa_dir = os.listdir('amazon_qa_dataset')
	qa_dicts = []
	for each in qa_dir:
		category = each.replace(".json","").split("QA_")[1]
		with open(os.path.join('amazon_qa_dataset',each)) as f:
			data = f.readlines()
			for each in data:
				qa = eval(each)
				prod_id = qa['asin']
				questions = qa['questions']
				qa_pairs = []
				for each_q in questions:
					qid = each_q['askerID']
					answers = each_q['answers']
					for each_a in answers:
						qaid = qid+'_'+each_a['answererID']
						qa_pair = {"qid":qid, "qaid":qaid, "questionType":each_q["questionType"],
		 							"question":each_q["questionText"],"answer":each_a["answerText"]}
						qa_pairs.append(qa_pair)
				qa_dicts.append({"asin":prod_id,"category":category,"qa_pair":qa_pairs})		
	with open('amazon_qa_ori.json','w') as f:
		json.dump(qa_dicts,f, indent=2)

def get_qa_annotate():
	with open('amazon_qa_ori_filter_item.json') as f:
		data = json.loads(f.read())
	with open('summary/qa_annotate.json') as f:
		ref_id = json.loads(f.read())
	for each_item in tqdm(data):
		filter_qa = []
		for each_qa in each_item["qa_pair"]:
			str2hash = each_item["asin"]+each_item["category"]+each_qa['question']+each_qa['answer']
			result = hashlib.md5(str2hash.encode())
			qa_id = result.hexdigest()
			if qa_id in ref_id.keys():
				if ref_id[qa_id] != "":
					if "qa_index" in ref_id[qa_id].keys():
						each_qa["qa_index"]=ref_id[qa_id]["qa_index"]
					each_qa["annotation"]=ref_id[qa_id]["annotation"]
				filter_qa.append(each_qa)
		each_item["qa_pair"]=filter_qa
	with open('qa_summary_filtered.json','w') as f:
		json.dump(data,f, indent=2)	

def get_summary():
	with open('amazon_qa_ori.json') as f:
		data = json.loads(f.read())
	with open('summary/prod_summary.json') as f:
		ref_id = json.loads(f.read())
	
	filter_item = []
	for each_item in tqdm(data):
		str2hash = each_item["asin"]+each_item["category"]
		result = hashlib.md5(str2hash.encode())
		sum_id = result.hexdigest()
		if sum_id in ref_id.keys():
			each_item["summary"]=ref_id[sum_id]
			filter_item.append(each_item)
	
	with open('amazon_qa_ori_filter_item.json','w') as f:
		json.dump(filter_item,f, indent=2)

def split_data():
	with open('qa_summary_filtered.json') as f:
		data = json.loads(f.read())
	with open('summary/split_dicts.json') as f:
		ref_id = json.loads(f.read())

	train = []
	dev = []
	test = []
	for each_item in tqdm(data):
		str2hash = each_item["asin"]+each_item["category"]
		result = hashlib.md5(str2hash.encode())
		split_id = result.hexdigest()
		if split_id in ref_id['train']:
			train.append(each_item)
		elif split_id in ref_id['dev']:
			dev.append(each_item)
		elif split_id in ref_id['test']:
			test.append(each_item)

	print(len(train),len(dev),len(test))
       
	with open('qa_summary_filtered_train.json','w') as f:
		json.dump(train,f, indent=2)
	with open('qa_summary_filtered_val.json','w') as f:
		json.dump(dev,f, indent=2)
	with open('qa_summary_filtered_test.json','w') as f:
		json.dump(test,f, indent=2)

if __name__ == '__main__':
	load_amazon_qa()
	get_summary()
	get_qa_annotate()
	split_data()