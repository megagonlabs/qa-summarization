import os, json, csv
import nltk, re
import itertools, random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pprint import pprint
from random import seed
from random import randint
from glob import glob
from functools import reduce
from itertools import islice
from collections import Counter
#from sumeval.metrics.rouge import RougeCalculator
from tqdm import tqdm
from nltk.tokenize import word_tokenize, sent_tokenize

def longestCommonSubsequence(s1, s2):

	n = len(s1)
	m = len(s2)

	res = ''

	dp = [[0] * (m + 1) for _ in range(n + 1)]

	for i in range(n + 1):
		for j in range(m + 1):
			if i == 0 or j == 0:
				dp[i][j] = 0
			elif s1[i - 1] == s2[j - 1]:
				dp[i][j] = 1 + dp[i - 1][j - 1]
			else:
				dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

	return dp[n][m] # for returning the length of lcs

def mturk_annotation_to_json():

	with open('8qa_ref.json','r',encoding='utf-8') as ref:
		ref_data = json.load(ref)
		print(len(ref_data))
	# exit()

	qa_list = []
	# qa_summary_data = {}
	path = 'mturk_results/post-edit_all/'

	cnt = 0
	items_list = []
	for filename in tqdm(glob(os.path.join(path, '*.json'))):
		hit_id = filename.split('/')[2].replace('.json','')
		# print('filename',filename.split('/')[2])
		cnt+=1
		with open(filename, 'r', encoding='utf-8') as f:
			data = json.load(f)
			# iterate 3 annotator
			for assign_id, data_info in data.items():
				worker_id = data_info[0]
				qa_info = json.loads(data_info[1])
				# print('each annotator json file')
				for each_item in qa_info:
					# reset qa dict
					qa_summary_data = {}
					# duplicted product id
					leng = int(len(each_item['prod_idx'])/2)
					if len(str(each_item['prod_idx']))>10:
						each_item['prod_idx']=each_item['prod_idx'][:leng]
						if leng < 10:
							each_item['prod_idx'] = '0'+each_item['prod_idx']
					# print('each item json file')
					# print(ref_data[each_item['prod_idx']])
					# exit()
					# annotated summary
					qa_pair = []
					err_list = []
					# correct QA decoding
					for i in range(len(each_item['ques'])):
						if each_item['ques'][i] not in ref_data[each_item['prod_idx']]['question']:
							# print('question:',i, each_item['ques'][i])
							# print('correct question:',ref_data[each_item['prod_idx']]['question'][i])
							each_item['ques'][i] = ref_data[each_item['prod_idx']]['question'][i]
						if each_item['ans'][i] not in ref_data[each_item['prod_idx']]['answer']:
							# print('answer:',i, each_item['ans'][i])
							# print('correct answer:',ref_data[each_item['prod_idx']]['answer'][i])
							each_item['ans'][i] = ref_data[each_item['prod_idx']]['answer'][i]
						qa_pair.append({'qa_index':worker_id+'#'+str(i),'wid':worker_id,'question':each_item['ques'][i],'answer':each_item['ans'][i],
									'qa_rewrite':each_item['qa_summary'][i]})
					# exit()
					# add new item
					if each_item['prod_idx'] not in set(items_list):
						qa_summary_data['asin']=each_item['prod_idx']
						qa_summary_data['category']=each_item['prod_cat']
						items_list.append(each_item['prod_idx'])
						qa_summary_data['qa_pair']=qa_pair
						qa_list.append(qa_summary_data)
					# existed item
					else:
						for each_qa in qa_list:
							if each_qa['asin']==each_item['prod_idx']:
								each_qa['qa_pair']+=qa_pair

	print(len(qa_list))
	print(len(set(items_list)))
	json.dump(qa_list, open('qa_annotated_summary.json', 'w', encoding='utf-8'), indent=2) 


def eval_4_standard():

	with open('qa_annotated_summary.json', 'r', encoding='utf-8') as f:
		qa_info = json.load(f)

	# error check
	worker_eval = {}
	item_worker_eval = {}
	worker_data, all_worker_data = [], []
	error, qa_count = 0, 0
	yn_list = ['Yes', 'yes', 'No', 'no']
	first_prons = ['I', 'Me', 'me', 'My', 'my', 'Mine', 'mine', 'Our', 'our', 'Ours', 'ours', 'We', 'we']
	third_prons = ["It", "it","Its","It's"]
	q_term = ['who', 'which', 'when', 'where', 'why', 'how', 'what','am', 'are', 'is', 'was', 'were',
			  'have', 'has', 'had', 'shall', 'will', 'should', 'would', 'do', 'does', 'did']
	'''
	check 1. LCS 2. Yes/No 3. QA format 4. It/it 5. First pronouns 6. ignore Q
	'''
	# item_summary_file = 'item_summary.json'
	item_sum = {}
	worker_list, item_list = [], []
	cnt = 0
	# iter item
	for item in tqdm(qa_info):
		# cnt+=1
		# pprint(item)
		# print(type(item))
		# exit()

		# iter qa pairs
		for each_qa in item['qa_pair']:
			# pprint(each_qa)
			lcs_qa = longestCommonSubsequence(each_qa['question']+' '+each_qa['answer'],each_qa['qa_rewrite'])
			lcs_sim = lcs_qa/len(each_qa['question']+' '+each_qa['answer'])
			clean_q = re.sub(r'[^\w\s]','',each_qa['question'])
			error = 0
			# 1. LCS
			if (lcs_sim > 0.7 or lcs_sim < 0.3):
				error += 2
				# print(lcs_sim)
				# print('error1')
				# print(row['Q-A post-edit'])
			# 2. Yes/No
			if (set(each_qa['qa_rewrite'].split()) & set(yn_list)):
				error += 2
				# print('error2')
				# print(row['Q-A post-edit'])
			# 3. QA format
			if (each_qa['qa_rewrite'].split()[0].lower() in q_term) or ('?' in each_qa['qa_rewrite']):
				error += 2
				# print('error3')
				# print(row['Q-A post-edit'])
			# 4. It/it
			if (each_qa['qa_rewrite'].split()[0] in third_prons):
				error += 1
				# print('error4')
				# print(row['Q-A post-edit'])
			# 5. First pronouns
			if (set(each_qa['qa_rewrite'].split()) & set(first_prons)):
				error += 2
				# print('error5')
				# print(row['Q-A post-edit'])
			# 6. ignore Q (maybe less error weight)
			if not any(ext in each_qa['qa_rewrite'].split() for ext in clean_q.split()):
				error += 1
				# print('error6')
				# print(row['Q-A post-edit'])
			each_qa['error score']=error
			
	json.dump(qa_info, open('qa_annotated_summary_full.json', 'w', encoding='utf-8'), indent=2) 


def best_rewrite():

	with open('qa_annotated_summary_full.json', 'r', encoding='utf-8') as f:
		qa_info = json.load(f)

	
	best_item_worker = {}
	best_rewrite_list = []
	for item in qa_info:
		best_rewrite_dict = {}
		best_rewrite_dict['asin'] = item['asin']
		best_rewrite_dict['category'] = item['category']
		max_score = 999
		best_worker = ''
		for qa1,qa2,qa3,qa4,qa5,qa6,qa7,qa8 in zip(*[iter(item['qa_pair'])]*8):
			score = qa1['error score']+qa2['error score']+qa3['error score']+qa4['error score']+qa5['error score']+qa6['error score']+qa7['error score']+qa8['error score']
			if score < max_score:
				best_rewrite_dict['QA1']=qa1['qa_rewrite']
				best_rewrite_dict['QA2']=qa2['qa_rewrite']
				best_rewrite_dict['QA3']=qa3['qa_rewrite']
				best_rewrite_dict['QA4']=qa4['qa_rewrite']
				best_rewrite_dict['QA5']=qa5['qa_rewrite']
				best_rewrite_dict['QA6']=qa6['qa_rewrite']
				best_rewrite_dict['QA7']=qa7['qa_rewrite']
				best_rewrite_dict['QA8']=qa8['qa_rewrite']
				best_rewrite_dict['error score'] = score
				max_score = score
				best_worker = qa1['wid']
		if max_score>15:
			print(best_rewrite_dict['asin'], max_score)
			# print(best_rewrite_dict)
		best_item_worker[item['asin']]=best_worker
		best_rewrite_list.append(best_rewrite_dict)
		# pprint(best_rewrite_list)
	# print(len(best_rewrite_list))
	# json.dump(best_rewrite_list, open('mturk_best_rewrite.json', 'w', encoding='utf-8'), indent=2)
	print(len(best_item_worker))
	json.dump(best_item_worker, open('mturk_best_worker_ref.json', 'w', encoding='utf-8'), indent=2)

def merge_qa_summary():

	with open('mturk_best_worker_ref.json', 'r', encoding='utf-8') as f:
		best_worker = json.load(f)

	with open('qa_annotated_ref.json', 'r', encoding='utf-8') as f:
		qa_annotated = json.load(f)

	# qa_annotated_dict={}
	# for item in qa_annotated:
	# 	qa_annotated_dict[item['asin']]=item['qa_pair']

	# json.dump(qa_annotated_dict, open('qa_annotated_ref.json', 'w', encoding='utf-8'), indent=2)

	with open('amazon_qa_summary.json', 'r', encoding='utf-8') as f:
		qa_augmented = json.load(f)

	cnt = 0
	for item in qa_augmented:
		# iter augmented qa
		for each_aug_qa in item['qa_pair']:
			# each_aug_qa['annotation']={'rewrite':''}
			for each_annoted_qa in qa_annotated[item['asin']]:
				if each_aug_qa['question']==each_annoted_qa['question'] and each_aug_qa['answer']==each_annoted_qa['answer']:
					# add rewrite info
					# print(each_aug_qa['question'],each_aug_qa['answer'])
					# print(each_annoted_qa['question'],each_annoted_qa['answer'])
					if each_annoted_qa['wid']==best_worker[item['asin']]:
						cnt+=1
						if 'annotation' not in each_aug_qa:
							each_aug_qa['annotation']={'rewrite':[{'edit':each_annoted_qa['qa_rewrite'],
														   'is_selected':'True',
														   'error score':each_annoted_qa['error score'],
														   'wid':each_annoted_qa['wid']}]}
						else:
							each_aug_qa['annotation']['rewrite']+=[{'edit':each_annoted_qa['qa_rewrite'],
														   'is_selected':'True',
														   'error score':each_annoted_qa['error score'],
														   'wid':each_annoted_qa['wid']}]
					else:
						if 'annotation' not in each_aug_qa:
							each_aug_qa['annotation']={'rewrite':[{'edit':each_annoted_qa['qa_rewrite'],
														   'is_selected':'False',
														   'error score':each_annoted_qa['error score'],
														   'wid':each_annoted_qa['wid']}]}
						else:
							each_aug_qa['annotation']['rewrite']+=[{'edit':each_annoted_qa['qa_rewrite'],
														   'is_selected':'False',
														   'error score':each_annoted_qa['error score'],
														   'wid':each_annoted_qa['wid']}]

	json.dump(qa_augmented, open('amazon_qa_summary_all.json', 'w', encoding='utf-8'), indent=2)
	# exit()

def simple_format(qa_summary):

	qa_summary_list = []
	# qa_summary_simple={}
	for each_item in tqdm(qa_summary):
		gold_sum = ''
		sample_text = ''
		for each_qa in each_item['qa_pair']:
			# extract gold summary
			if 'annotation' in each_qa:
				for each_edit in each_qa['annotation']['rewrite']:
					if each_edit['is_selected']=='True':
						gold_sum += each_edit['edit'] if each_edit['edit'][-1]==' ' else each_edit['edit']+' '
			sample_text += (each_qa['question']+' '+each_qa['answer'])
		
		qa_summary_list.append({'asin':each_item['asin'],'text':sample_text,'summary':gold_sum})
		# print('gold_sum:',gold_sum)
		# print('sample_text:',sample_text)
	print(len(qa_summary_list))	
	return qa_summary_list
	# exit()

def create_qa_dataset():

	path = '../qa-summarization/'

	seed(7)
	with open(path+'amazon_qa_summary_all.json', 'r', encoding='utf-8') as f:
		qa_summary = json.load(f)
		print(len(qa_summary))

	random.shuffle(qa_summary)
	train_data = qa_summary[:int(len(qa_summary)*0.8)]
	val_data = qa_summary[int(len(qa_summary)*0.8):int(len(qa_summary)*0.9)]
	test_data = qa_summary[int(len(qa_summary)*0.9):]

	# train_data = simple_format(train_data)
	# val_data = simple_format(val_data)
	# test_data = simple_format(test_data)

	# calculate stastic
	# data_statistic(train_data+val_data)
	# data_statistic(qa_summary)
	# data_statistic(train_data)
	# data_statistic(val_data)
	# data_statistic(test_data)

	# extract simple info ['id','text','summary']
	# with open(path+'amz_qa_sum_train_simple.json', 'w', encoding='utf-8') as outfile:
	# 	for item in train_data:
	# 		print(json.dumps(item), file=outfile)
	# with open(path+'amz_qa_sum_val_simple.json', 'w', encoding='utf-8') as outfile:
	# 	for item in val_data:
	# 		print(json.dumps(item), file=outfile)
	# with open(path+'amz_qa_sum_test_simple.json', 'w', encoding='utf-8') as outfile:
	# 	for item in test_data:
	# 		print(json.dumps(item), file=outfile)
	
	json.dump(train_data, open(path+'amz_qa_sum_train_ori.json', 'w', encoding='utf-8'))
	json.dump(val_data, open(path+'amz_qa_sum_val.json_ori', 'w', encoding='utf-8'))
	json.dump(test_data, open(path+'amz_qa_sum_test_ori.json', 'w', encoding='utf-8'))


def data_statistic(qa_summary):

	qa_summary_simple={}
	sam_txt, g_sum = [], []
	for each_item in tqdm(qa_summary):
		gold_sum = ''
		sample_text = ''
		for each_qa in each_item['qa_pair']:
			# extract gold summary
			if 'annotation' in each_qa:
				for each_edit in each_qa['annotation']['rewrite']:
					if each_edit['is_selected']=='True':
						gold_sum += each_edit['edit'] if each_edit['edit'][-1]==' ' else each_edit['edit']+' '
			sample_text += (each_qa['question']+' '+each_qa['answer'])

		# sam_txt.append(len(word_tokenize(sample_text)))
		# g_sum.append(len(word_tokenize(gold_sum)))
		sam_txt.append(len(sent_tokenize(sample_text)))
		g_sum.append(len(sent_tokenize(gold_sum)))
		# qa_summary_simple[each_item['asin']]=

	# print('avg. all QA pairs length (words):',round(sum(sam_txt)/len(sam_txt),2))
	# print('avg. gold summary length (words):',round(sum(g_sum)/len(g_sum),2)) 
	print('avg. all QA pairs length (sents):',round(sum(sam_txt)/len(sam_txt),2))
	print('avg. gold summary length (sents):',round(sum(g_sum)/len(g_sum),2))            

def load_data():
    with open('amazon_qa_dataset/amazon_qa_summary_filtered.json','r',encoding='utf-8') as infile:
        data = json.load(infile)
        print(len(data))
        total_summary = []
        word_cnt = 0
        max_len = 0
        for each_item in data:
           # qa_pairs = each_item['qa_pair']
           # print(len(qa_pairs))
           # for qa_pair in qa_pairs:
           #     print('Q:',qa_pair['question'])
           #     print('A:',qa_pair['answer'])
            #print(len(each_item['summary']))
            total_summary += each_item['summary']
            for each_sum in each_item['summary']:
                word_cnt += len(each_sum)
                max_len = max(len(each_sum), max_len)
                print(max_len)
                exit()
        print(len(total_summary))
        print(word_cnt)
        print(word_cnt/len(total_summary))
        print(max_len)
        exit()

def split_data():
    with open('amazon_qa_dataset/amazon_qa_summary_filtered.json','r',encoding='utf-8') as infile:
        data = json.load(infile)
        print(len(data))
        nums_of_data = len(data)
        random.shuffle(data)
        train_data = data[:int(nums_of_data*0.8)]
        val_data = data[int(nums_of_data*0.8):int(nums_of_data*0.9)]
        test_data = data[int(nums_of_data*0.9):]
        #print(len(train_data),train_data[:3])
        #print(len(val_data),val_data[:3])
        #print(len(test_data),test_data[:3])
        data_list = [train_data, val_data, test_data]
    
    with open('qa_summary_filtered_train.json', 'w', encoding='utf-8') as outfile:
        json.dump(train_data, outfile, indent=2)
    
    with open('qa_summary_filtered_val.json', 'w', encoding='utf-8') as outfile:
        json.dump(val_data, outfile, indent=2)
    
    with open('qa_summary_filtered_test.json', 'w', encoding='utf-8') as outfile:
        json.dump(test_data, outfile, indent=2)
    

if __name__ == "__main__":

    load_data()
    exit()
    split_data()
	#exit()
    #create_qa_dataset()
    # merge_qa_summary()
	# best_rewrite()
	# eval_4_standard()
	# mturk_annotation_to_json()
