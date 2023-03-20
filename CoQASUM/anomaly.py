import os
import json
import hashlib
from pprint import pprint
 

def mask_data():
	'''load CoQASUM data'''
	# with open('amazon_qa_summary_filtered.json') as f:
	# 	data = json.loads(f.read())
	# 	print(len(data))
	# 	hash_summary(data)
	# 	hash_qa_rewrite(data)

	'''load train dev test data'''
	split_dcits = {}
	for each in ['train','dev','test']:
		with open('qa_summary_filtered_'+each+'.json') as f:
			data = json.loads(f.read())
			print(len(data))
			split_dcits[each] = hash_train_dev_test(data)
	'''save split dicts'''
	with open('split_dicts.json','w') as f:
		json.dump(split_dcits,f)

def hash_train_dev_test(data):
	split = []
	for qa_data in data:
		str2hash = qa_data['asin']+qa_data['category']
		result = hashlib.md5(str2hash.encode())
		split_id = result.hexdigest()
		split.append(split_id)
	return split

def hash_qa_rewrite(data):
	anno_dict = {}
	for qa_data in data:
		for qa_pair in qa_data['qa_pair']:
			# if 'annotation' in qa_pair.keys():
			str2hash = qa_data['asin']+qa_data['category']+qa_pair['question']+qa_pair['answer']
			result = hashlib.md5(str2hash.encode())
			qa_hash = result.hexdigest()
			if 'annotation' in qa_pair.keys():
				# print(qa_pair["qa_index"], "annotation")
				if "qa_index" in qa_pair.keys():
					anno_dict[qa_hash] = {"qa_index":qa_pair["qa_index"],"annotation":qa_pair['annotation']}
				else:
					anno_dict[qa_hash] = {"annotation":qa_pair['annotation']}
			else:
				# print(qa_pair["qa_index"], "no annotation")
				anno_dict[qa_hash] = ""
	with open('qa_annotate.json','w') as f:
		json.dump(anno_dict,f, indent=4)

def hash_summary(data):
	sum_dict = {}
	for qa_data in data:
		str2hash = qa_data['asin']+qa_data['category']
		result = hashlib.md5(str2hash.encode())
		# print(result)
		sum_id = result.hexdigest()
		sum_dict[sum_id] = qa_data["summary"]
	
	'''
	add idx for summary
	save to json file
	'''
	with open('prod_summary.json','w') as f:
		json.dump(sum_dict,f, indent=4)

def load_amazon_qa():
	qa_dir = os.listdir('data')
	for each in qa_dir:
		category = each.replace(".json","").split("qa_")[1]
		with open(os.path.join('data',each)) as f:
			for line in f:
				qa_data = json.load(line)
				str2hash = qa_data['asin']+qa_data['question']
				result = hashlib.md5(str2hash.encode())
				qa_data['id'] = result.hexdigest()
				qa_data['category'] = category
				yield qa_data


if __name__ == '__main__':
	mask_data()