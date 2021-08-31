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
from sumeval.metrics.rouge import RougeCalculator
from tqdm import tqdm

from lexrank import LexRank
from lexrank.mappings.stopwords import STOPWORDS
from path import Path


# seed(1)
seed(20)

path = 'QA_Office_Products.json'

Amazon_prod_dict = {}

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


def levenshteinDistance(s1, s2):
    if len(s1) > len(s2):
        s1, s2 = s2, s1

    distances = range(len(s1) + 1)
    for i2, c2 in enumerate(s2):
        distances_ = [i2+1]
        for i1, c1 in enumerate(s1):
            if c1 == c2:
                distances_.append(distances[i1])
            else:
                distances_.append(1 + min((distances[i1], distances[i1 + 1], distances_[-1])))
        distances = distances_
    return distances[-1]

def preprocess():
	with open(path, 'r', encoding='utf-8') as f:
		
		data = f.readlines()
		print('total items:',len(data))
		# print(data[0])
		# exit()

		total_questions, total_answers = 0, 0
		
		for each_item in data:

			qa = eval(each_item)
			pprint(qa)
			exit()
			# print(len(qa))

			prod_id = qa['asin']
			# print(prod_id)
			
			questions = qa['questions']
			total_questions += len(questions)
			# print('number of questions (per item):',len(questions))

			total_ans_each_q = 0
			for each_q in questions:
				answers = each_q['answers']
				total_ans_each_q += len(answers)
				# print('number of answer:',len(answers))
			total_answers += total_ans_each_q
			# print('number of answers (per question):',total_ans_each_q)

		print('total_questions:',total_questions)
		print('total_answers:',total_answers)

def parse_sample_info():

	# csv_file = open('sample_product_info_8q.csv', 'w')
	# writer = csv.writer(csv_file)
	# writer.writerow(['Product', 'Product Id','Question', 'Answer (min)', 'Answer (max)'])

	# with open('Amazon_product_sample_8_questions.json', 'r', encoding='utf-8') as f:
	# 	data = json.load(f)
	# 	for prod, qa_info in data.items():
	# 		print(prod)
	# 		# print(len(qa_info))
	# 		for each_item in qa_info:
	# 			prod_id = each_item['asin']
	# 			qa_pair = each_item['questions'][randint(0,8)-1] #random pick one of eight
	# 			ques = qa_pair['questionText']
	# 			defult_ans = nltk.word_tokenize(qa_pair['answers'][0]['answerText'])
	# 			min_ans, max_ans = defult_ans, defult_ans
	# 			for each_ans in qa_pair['answers']:
	# 				ans = each_ans['answerText']
	# 				token = nltk.word_tokenize(ans)
	# 				min_ans = ans if len(token) <= len(min_ans) else min_ans
	# 				max_ans = ans if len(token) >= len(max_ans) else max_ans
	# 			# ans = qa_pair['answers'][0]['answerText']				
	# 			writer.writerow([prod, prod_id, ques, ''.join(min_ans), ''.join(max_ans)])
	
	# csv_file.close()
	
	Amazon_prod_dict = {}
	with open('Amazon_product_sample_8_questions.json', 'r', encoding='utf-8') as f:
		
		data = json.load(f)

		for prod, qa_info in data.items():
			print(prod)
			print(len(qa_info))
			item_dict = {}
			for each_item in qa_info:
				prod_id = each_item['asin']
				qa_pair = each_item['questions']
				for each_qa in qa_pair:
					del each_qa['askerID']
					del each_qa['questionTime']
					del each_qa['answers']

				item_dict[prod_id] = qa_pair
			Amazon_prod_dict[prod] = item_dict

	json.dump(Amazon_prod_dict, open("Amazon_product_info_ref.json", 'w'))
	
def sample_100_item_qa_pair():

	# with open('Amazon_product_sample_over_8_questions.json', 'r', encoding='utf-8') as f:
	# 	data = json.load(f)
	# 	sample_qa_pair = {}
	# 	for category, qa_info in data.items():
	# 		print(category)
	# 		qa_sample_list = random.sample(qa_info, 200)
	# 		qa_pair_per_item, items_list = [], []
	# 		for each_item in qa_sample_list:
	# 			qa_pair_list = []
	# 			prod_id = each_item['asin']
	# 			qa_pair_per_item = random.sample(each_item['questions'], 8)	
	# 			for qa_pair in qa_pair_per_item:
	# 				ans = []
	# 				for each_ans in qa_pair['answers']:
	# 					tmp_ans = nltk.word_tokenize(each_ans['answerText'])
	# 					if len(tmp_ans) > 3:
	# 						ans.append(each_ans['answerText'])
	# 				if ans:
	# 					ans = random.sample(ans,1)[0]
	# 					qa_pair_list.append((qa_pair['questionText'], ans))
	# 			if len(qa_pair_list) == 8:
	# 				items_list.append({'prod_id':prod_id, 'Q-A_pair':qa_pair_list})
	# 		print(len(items_list))
	# 		sample_qa_pair[category]=items_list[:100]

	# json.dump(sample_qa_pair, open("Amazon_sample_100_item_qa_info.json", 'w'))
	# exit()

	csv_file = open('sample_100_item_qa_pair.csv', 'w', encoding='utf-8')
	writer = csv.writer(csv_file)
	writer.writerow(['Product', 'Product Id', 'Question', 'Answer'])

	with open('Amazon_sample_120_item_qa_info.json', 'r', encoding='utf-8') as f:
		data = json.load(f)
		for cat, prod_list in data.items():
			print(cat)
			print(len(prod_list))
			for each_item in prod_list:
				prod_id = each_item['prod_id']
				qa_pairs = each_item['Q-A_pair']
				for each_qa in qa_pairs:
					writer.writerow([cat, prod_id, each_qa[0], each_qa[1]])
					

def parse_cluster_sample():

	with open('Amazon_product_info_ref.json', 'r', encoding='utf-8') as f:
		ref = json.load(f)


	csv_file = open('sample_cluster6_8q_label_info.csv', 'w')
	writer = csv.writer(csv_file)

	writer.writerow(['Product', 'Product Id', 'Cluster 1 (Q of Q-A)', 'Q types label (1: open-ended, 0: yes/no)','Ratio (of total)', 
											'Cluster 2 (Q of Q-A)', 'Q types label (1: open-ended, 0: yes/no)','Ratio (of total)',
					 						'Cluster 3 (Q of Q-A)', 'Q types label (1: open-ended, 0: yes/no)','Ratio (of total)',
					 						'Cluster 4 (Q of Q-A)', 'Q types label (1: open-ended, 0: yes/no)','Ratio (of total)',
					 						'Cluster 5 (Q of Q-A)', 'Q types label (1: open-ended, 0: yes/no)','Ratio (of total)',
					 						'Cluster 6 (Q of Q-A)', 'Q types label (1: open-ended, 0: yes/no)','Ratio (of total)',
					 						])

	# writer.writerow(['Product', 'Product Id', 'Cluster Type', 'Cluster 1', 'Cluster 2', 
	# 				 'Cluster 3', 'Cluster 4', 'Cluster 5', 'Cluster 6'])

	with open('Amazon_product_sample_8q_cluster6_info.json', 'r', encoding='utf-8') as f:
		data = json.load(f)
		for cat, prod_list in data.items():
			print(cat)
			print(len(prod_list))
			for each_item in prod_list:
				prod_id = each_item['product_id']
				# ques_cluster = each_item['questions_cluster']
				# ans_cluster = each_item['answers_cluster']
				qa_cluster = each_item['qa_cluster']
				qa_list = []
				for c, qa in qa_cluster.items():
					q_type_list = []
					qa_q_only = list(set([each_qa.split('[SEP]')[0].strip() for each_qa in qa]))
					for each_q in qa_q_only:
						for q_index in ref[cat][prod_id]:
							if each_q == q_index['questionText']:
								q_type_list.append(q_index['questionType'])

					qa_list.append(qa_q_only)
					qa_list.append(q_type_list)
					
					cnt = 0
					for q_type in q_type_list:
						if q_type == 'open-ended':
							cnt += 1
					qa_list.append(cnt/len(q_type_list))
					# print(qa_list)

				# writer.writerow([cat, prod_id, 'Questions']+[v for k, v in ques_cluster.items()])
				# writer.writerow([cat, prod_id, 'Answers']+[v for k, v in ans_cluster.items()])
				# writer.writerow([cat, prod_id]+[v for k, v in qa_cluster.items()])
				writer.writerow([cat, prod_id]+ qa_list)
				
	csv_file.close()

def qa_cluster_diff_sample():

	csv_file = open('overlap_cluster6_8q.csv', 'w')
	writer = csv.writer(csv_file)
	writer.writerow(['Product Id', 'Questions in diff Cluster'])

	with open('Amazon_product_sample_8q_cluster6_info.json', 'r', encoding='utf-8') as f:
		data = json.load(f)
		q_cross_item = []
		for cat, prod_list in data.items():
			
			for each_item in prod_list:
				prod_id = each_item['product_id']
				ques_cluster = each_item['questions_cluster']
				ans_cluster = each_item['answers_cluster']
				qa_cluster = each_item['qa_cluster']
				qa_q_list = []

				for c, qa in qa_cluster.items():
					qa_q_only = list(set([each_qa.split('[SEP]')[0] for each_qa in qa]))
					qa_q_list.append((c,qa_q_only))

				qa_q_list = sorted(qa_q_list, key=lambda x: x[0])
				for first, second in zip(qa_q_list, qa_q_list[1:]):
					tmp = list(set(first[1]) & set(second[1]))
					if tmp:
						q_cross_item.append((prod_id,tmp))
	
	for row in q_cross_item:
		writer.writerow([row[0], row[1]])

	csv_file.close()


def qa_statistic():

	'''
	count the questions/per product
	'''
	with open('Amazon_product_info.json', 'r', encoding='utf-8') as f:
		data = json.load(f)
		items_list = []
		for prod, prod_info in data.items():
			print(prod)
			# pprint(prod_info['items info'][:5])
			print(len(prod_info['items info']))
			count_ques = {}
			items = prod_info['items info']
			for each_item in items:
				if each_item['number of questions'] not in count_ques:
					count_ques[each_item['number of questions']] = 1
				else:
					count_ques[each_item['number of questions']] += 1
		
			items_list.append({prod:count_ques})
		
		print(items_list)
		# plot_fig(items_list)

	'''
	count the question/per answer
	'''
	# with open('Amazon_product_info.json', 'r', encoding='utf-8') as f:
	# 	data = json.load(f)
	# 	# items_list = []
	# 	count_ques, total_ans = {}, {}
	# 	for prod, prod_info in data.items():
	# 		# print(prod)
	# 		# pprint(prod_info['items info'][:5])
	# 		# print(len(prod_info['items info']))
			
	# 		items = prod_info['items info']
	# 		for each_item in items:
	# 			if each_item['number of questions'] not in count_ques:
	# 				count_ques[each_item['number of questions']] = 1
	# 				total_ans[each_item['number of questions']] = sum(each_item['number of answers'])
	# 			else:
	# 				count_ques[each_item['number of questions']] += 1
	# 				total_ans[each_item['number of questions']] += sum(each_item['number of answers'])
		
	# 	ans = []
	# 	for k, v in total_ans.items():
	# 		ans.append((k,'{:.2f}'.format(total_ans[k]/count_ques[k]/int(k))))
	# 	print(ans)

def plot_fig(count_dict):


	df = pd.DataFrame()
	for prod in count_dict:
		# print(prod)
		for cat, ques in prod.items():
			each_df = pd.DataFrame(ques.items(), columns=['num of ques', 'items'])
			each_df['Product'] = [cat for i in range(len(ques))]
			# print(new_df)
			df = df.append(each_df)
	
	# df = df.sort_values('a')
	print(df.head(50))
	# exit()

	sns.relplot(data=df, x='num of ques', y='items', hue='Product', kind='line')
	plt.show()
	
	exit()


def cluster_represent():

	csv_file = open('sample_cluster6_8q_rouge.csv', 'w')
	writer = csv.writer(csv_file)
	writer.writerow(['Product', 'Product Id', 'Cluster 1', 'Cluster 2',
					 'Cluster 3', 'Cluster 4', 'Cluster 5', 'Cluster 6', 
					 'Removed Ans'])

	rouge = RougeCalculator(stopwords=True, lang="en")
	with open('Amazon_product_sample_8q_cluster6_info.json', 'r', encoding='utf-8') as f:

		data = json.load(f)
		for cat, prod_list in data.items():
			
			print(cat, len(prod_list))

			for each_item in prod_list:
				prod_id = each_item['product_id']
				ques_cluster = each_item['questions_cluster']
				ans_cluster = each_item['answers_cluster']
				qa_cluster = each_item['qa_cluster']
				max_rouge_list, second_rouge_list, third_rouge_list = [], [], []
				short_ans = [] 
				for clu, qa_pair in qa_cluster.items():

					#remove short length
					remove_list = []
					qa_a_only = [(idx, each_qa.split('[SEP]')[1]) for idx, each_qa in enumerate(qa_pair)] 
					# print(qa_pair)

					for idx, each_ans in qa_a_only:
						tmp_ans = nltk.word_tokenize(each_ans)
						if len(tmp_ans) < 3:
							short_ans.append(' '.join(tmp_ans))
							remove_list.append(qa_pair[idx])

					for each in remove_list:
						qa_pair.remove(each) 

					# print(short_ans, remove_list)
					# print(qa_pair)

					rouge_list = []
					for i in range(len(qa_pair)):
						# print(len(qa_pair))
						rouge_l = rouge.rouge_l(summary=qa_pair[i],references=qa_pair[i:])
						# print(qa_pair[i], rouge_l)
						rouge_list.append((round(rouge_l,3), qa_pair[i]))

					rouge_list = sorted(rouge_list, key=lambda x: x[0], reverse=True)
					# print(rouge_list)
					if rouge_list:
						max_rouge_list.append(rouge_list[0])
					else:
						max_rouge_list.append([])

				writer.writerow([cat, prod_id] + max_rouge_list + [short_ans])
				
	csv_file.close()

def sample_annotated_data():

	# qa_sample_50 = open("qa_sample_50.txt","w+")
	ques_cluster, ans_cluster, ids_list, category = [], [], [], []
	
	with open('Amazon_product_sample_8q_cluster6_info.json', 'r', encoding='utf-8') as f:

		data = json.load(f)
		for cat, prod_list in data.items():
			
			print(cat, len(prod_list))
			
			tmp_qa_list = []
			for each_item in prod_list:
				category.append(cat)

				prod_id = each_item['product_id']
				qa_cluster = each_item['qa_cluster']
				

				for clu, qa_pair in qa_cluster.items():

					#remove short length
					remove_list = []
					qa_a_only = [(idx, each_qa.split('[SEP]')[1]) for idx, each_qa in enumerate(qa_pair)] 

					for idx, each_ans in qa_a_only:
						tmp_ans = nltk.word_tokenize(each_ans)
						if len(tmp_ans) < 3:
							remove_list.append(qa_pair[idx])

					for each in remove_list:
						qa_pair.remove(each) 

					for each_qa in qa_pair:
						qa = each_qa.split('[SEP]')
						ques, ans = qa[0], qa[1]
						tmp_qa_list.append((prod_id, ques, ans))
						# tmp_ans_list.append(ans)
	

			tmp_qa_list = random.sample(tmp_qa_list, 5)
			for ids, q, a in tmp_qa_list:
				ids_list.append(ids)
				ques_cluster.append(q)
				ans_cluster.append(a)

		json.dump(category, open("cats_sample_5.txt", 'w'))
		json.dump(ids_list, open("ids_sample_5.txt", 'w'))
		json.dump(ques_cluster, open("qa_q_sample_5.txt", 'w'))
		json.dump(ans_cluster, open("qa_a_sample_5.txt", 'w'))


def parse_annotated_data():

	# # with open('annotated_result.json', 'r', encoding='utf-8') as f:
	# with open('annotated_testset.json', 'r', encoding='utf-8') as f:
	# 	data = json.load(f)
	# 	# pprint(data)

	# csv_file = open('sample_85_annotarted_result.csv', 'w')
	# # csv_file = open('sample_850_annotarted_result.csv', 'w')
	# writer = csv.writer(csv_file)
	# writer.writerow(['Product', 'Product Id', 'Question', 'Answer',
	# 				 'Label(good:1/bad:0)'])

	# for each_qa in data:
	# 	label = 1 if each_qa['qa_quality']== 'yes' else 0
	# 	writer.writerow([each_qa['prod_cat'], each_qa['prod_idx'], each_qa['ques'], each_qa['ans'], label])
	# 	# exit()

	# sample small testset
	with open('annotated_testset.json', 'r', encoding='utf-8') as f:
		data = json.load(f)
		# pprint(data)

	good_qa, bad_qa = [], []
	for each_qa in data:
		if each_qa['qa_quality']== 'yes':
			good_qa.append(each_qa['ques']+'[SEP]'+each_qa['ans'])
		else:
			bad_qa.append(each_qa['ques']+'[SEP]'+each_qa['ans'])

	# print(good_qa[:5])#+bad_qa[:5])
	# print(bad_qa[:5])
	print(bad_qa[0])
	print(bad_qa[2])
	print(bad_qa[3])
	exit()
	with open('testset_sample.txt','w+', encoding='utf-8') as outfile:
		outfile.writelines(good_qa)
		outfile.writelines(bad_qa)

		# writer.writerow([each_qa['prod_cat'], each_qa['prod_idx'], each_qa['ques'], each_qa['ans'], label])
		# exit()


def check():

	# with open('Amazon_product_sample.json', 'r', encoding='utf-8') as f:
	# 	data = json.load(f)
	# 	for each in data['Video_Games']:
	# 		if each['asin'] == 'B00006B84X':
	# 			print(each['questions'][0]['questionText'])

	csv_file = open('qa_statistic.csv', 'w')
	writer = csv.writer(csv_file)
	writer.writerow(['Category', 'total items', 'total questions', 'total answers',
					 'avg questions (per item)', 'avg answers (per item)', 
					 'avg words (per question)','avg words (per answer)'])

	with open('Amazon_product_info.json', 'r', encoding='utf-8') as f:
		data = json.load(f)
		for cat, data_info in data.items():
			writer.writerow([cat, data_info['total items'], data_info['total questions'], 
							data_info['total answers'], data_info['avg questions (per item)'], 
							data_info['avg answers (per item)'], data_info['avg words (per question)'],
							data_info['avg words (per answer)']])

def parse_filtered_data():

	path='../amazon_qa_filtered_5_150/'
	item_8qa_list = []

	qa_dict = {}

	csv_file = open('amazon_qa_filtered_5_150_8q.csv', 'w', encoding='UTF-8')
	writer = csv.writer(csv_file)
	writer.writerow(['Product', 'Product Id','Question', 'Answer'])
	cnt = start = 0
	for filename in glob(os.path.join(path, '*.csv')):
		print(filename, (cnt-start)/8)
		# exit()
		start = cnt
		with open(path+filename, newline='', encoding='UTF-8') as f:
			reader = csv.reader(f)
			head = next(reader)
			print(head)
			cat, prod_id, ques, ans =['default'], ['default'], ['default'],['default']
			for row in reader:
				print(row)
				if not row[1] in prod_id:
					if len(set(ques))>=8:
						for i in range(8): 
							writer.writerow([cat[i], prod_id[i], ques[i], ans[i]])
							cnt+=1
					cat, prod_id, ques, ans = [], [], [], []
				cat.append(row[0])
				prod_id.append(row[1])
				if not row[2] in ques:
					ques.append(row[2])
					ans.append(row[3])


	# path='../amazon_qa_filtered_5_150/'
	# item_8qa_list = []

	# qa_dict = {}

	# csv_file = open('amazon_qa_filtered_5_150_multi-QA.csv', 'w', encoding='utf-8')
	# writer = csv.writer(csv_file)
	# writer.writerow(['Product', 'Product Id','Question', 'Answer'])

	# for filename in glob(os.path.join(path, '*.csv')):
	# 	with open(path+filename, newline='', encoding='utf-8') as f:
	# 		reader = csv.reader(f)
	# 		head = next(reader)
	# 		print(head)
	# 		for row in reader:
	# 			writer.writerow([row[0], row[1], row[2], row[3]])
						

def get_mturk_annotation():

	# path = 'mturk_results/filtered/'

	# csv_file = open(path+'mturk_30_items_results_filtered.csv', 'w')
	# writer = csv.writer(csv_file)
	# writer.writerow(['Assign Id', 'Worker Id', 'Product', 'Product Id',
	# 				 'Question', 'Answer', 'Label(pos:yes/neg:no)'])

	# for filename in glob(os.path.join(path, '*.json')):
	# 	# print(filename)
	# 	hit_id = filename.split('/')[1].replace('.json','')
	# 	pprint(hit_id)
	# 	with open(filename, 'r', encoding='utf-8') as f:
	# 		data = json.load(f)
	# 		# pprint(data)
	# 		for assign_id, data_info in data.items():
	# 			print(assign_id)
	# 			print(data_info[0])
	# 			worker_id = data_info[0]
	# 			qa_info = json.loads(data_info[1])
	# 			# pprint(qa_info)
	# 			for each_item in qa_info:
	# 				for i in range(8): 
	# 					writer.writerow([assign_id, worker_id, each_item['prod_cat'], each_item['prod_idx'],
	# 								 each_item['ques'][i], each_item['ans'][i], 
	# 								 1 if each_item['qa_quality'][i]=='yes' else 0])

	path = 'mturk_results/post-edit_all/'
	# path = 'mturk_results/post-edit_v4_batch_6'

	csv_file = open('mturk_results/mturk_results_filtered_post-edit_all(stat).csv', 'w', encoding='utf-8')
	# csv_file = open('mturk_results/mturk_results_filtered_post-edit_v4(stat).csv', 'w', encoding='utf-8')
	writer = csv.writer(csv_file)
	writer.writerow(['Assign Id', 'Worker Id', 'Product', 'Product Id',
					 'Question', 'Answer', 'Q-A post-edit', 'LCS(Q)',
					 'LCS(A)','LCS(Q-A)','Time(s)'])
	items_list = []
	for filename in tqdm(glob(os.path.join(path, '*.json'))):
		hit_id = filename.split('/')[2].replace('.json','')
		# print('hit id:',hit_id)
		with open(filename, 'r', encoding='utf-8') as f:
			data = json.load(f)
			for assign_id, data_info in data.items():
				worker_id = data_info[0]
				qa_info = json.loads(data_info[1])
				for each_item in qa_info:
					# duplicted product id
					leng = int(len(each_item['prod_idx'])/2)
					if len(str(each_item['prod_idx']))>10:
						each_item['prod_idx']=each_item['prod_idx'][:leng]
						if leng < 10:
							each_item['prod_idx'] = '0'+each_item['prod_idx']
						# if '792283' in str(each_item['prod_idx']):
						# 	print(each_item['prod_idx'])
					items_list.append(each_item['prod_idx'])

					# correct QA decoding
					for i in range(len(each_item['ques'])):
						# lcs_q = longestCommonSubsequence(each_item['ques'][i], each_item['qa_summary'][i])
						# lcs_a = longestCommonSubsequence(each_item['ans'][i], each_item['qa_summary'][i])
						# lcs_qa = longestCommonSubsequence(each_item['ques'][i]+' '+each_item['ans'][i], each_item['qa_summary'][i])
						lcs_q = 0
						lcs_a = 0
						lcs_qa = 0
						if i == 0:
							if 'time' in each_item:
								writer.writerow([assign_id, worker_id, each_item['prod_cat'], each_item['prod_idx'],
									 each_item['ques'][i], each_item['ans'][i], each_item['qa_summary'][i].replace('\n',' ').strip(), 
									 lcs_q, lcs_a, lcs_qa, each_item['time']])
							else:
								writer.writerow([assign_id, worker_id, each_item['prod_cat'], each_item['prod_idx'],
									 each_item['ques'][i], each_item['ans'][i], each_item['qa_summary'][i].replace('\n',' ').strip(), 
									 lcs_q, lcs_a, lcs_qa, 'Not record'])
						else:
							writer.writerow([assign_id, worker_id, each_item['prod_cat'], each_item['prod_idx'],
									 each_item['ques'][i], each_item['ans'][i], each_item['qa_summary'][i].replace('\n',' ').strip(), 
									 lcs_q, lcs_a, lcs_qa])


	print(len(set(items_list)))


def agreement():

	df = pd.read_csv("mturk_results/filtered/mturk_30_items_results_filtered.csv")
	df["QA"] = df["Question"] + "=====" + df["Answer"]
	print(df.groupby("QA").mean("Label(pos:1/neg:0)").value_counts())


def eval_lcs():

	'''
	evaluate LCS 
	'''

	df = pd.read_csv('mturk_results/post-edit/mturk_30_items_results_filtered_post-edit(stat).csv',encoding='utf-8')
	# print(df.head())
	
	lsc_q_min = df.groupby(['Question']).agg({'Product Id':'first','Question':'first','Answer':'first',
											  'LCS(Q)':'min','Worker Id':'first'})
	print(lsc_q_min)
	# print(list(lsc_q_min))
	# print(Counter(lsc_q_min))

	lsc_q_max = df.groupby(['Question']).agg({'Product Id':'first','Question':'first','Answer':'first',
											  'LCS(Q)':'max','Worker Id':'last'})
	print(lsc_q_max)
	# print(list(lsc_q_max))
	# print(Counter(lsc_q_max))

	lsc_a_min = df.groupby(['Question']).agg({'Product Id':'first','Question':'first','Answer':'first',
											  'LCS(Q)':'min','Worker Id':'first'})
	print(lsc_a_min)
	# print(list(lsc_q_min))
	# print(Counter(lsc_q_min))

	lsc_a_max = df.groupby(['Question']).agg({'Product Id':'first','Question':'first','Answer':'first',
											  'LCS(Q)':'max','Worker Id':'last'})
	print(lsc_a_max)
	# print(list(lsc_q_max))
	# print(Counter(lsc_q_max))

	lsc_qa_min = df.groupby(['Question']).agg({'Prodct Id':'first','Question':'first','Answer':'first',
											  'LCS(Q)':'min','Worker Id':'first'})
	print(lsc_qa_min)
	# print(list(lsc_q_min))
	# print(Counter(lsc_q_min))

	lsc_qa_max = df.groupby(['Question']).agg({'Product Id':'first','Question':'first','Answer':'first',
											  'LCS(Q)':'max','Worker Id':'last'})
	print(lsc_qa_max)
	# print(list(lsc_q_max))
	# print(Counter(lsc_q_max))

	csv_file = open('mturk_results/post-edit/post-edit_eval.csv', 'w')
	writer = csv.writer(csv_file)
	writer.writerow(['Product Id','Question', 'Answer',
					 'LCS(Q) Min','LCS(Q) Max','LCS(A) Min','LCS(A) Max','LCS(Q-A) Min','LCS(Q-A) Max',])
	
	for i in range(len(list(lsc_q_min['Worker Id']))):
		writer.writerow([lsc_q_min['Product Id'][i],lsc_q_min['Question'][i],lsc_q_min['Answer'][i],
						 lsc_q_min['Worker Id'][i], lsc_q_max['Worker Id'][i], lsc_a_min['Worker Id'][i], 
						 lsc_a_max['Worker Id'][i],lsc_qa_min['Worker Id'][i], lsc_qa_max['Worker Id'][i]])


def eval_4_standard():

	df = pd.read_csv('mturk_results/mturk_results_filtered_post-edit_all(stat).csv',encoding='utf-8')
	# df = pd.read_csv('mturk_results/mturk_results_filtered_post-edit_v4(stat).csv',encoding='utf-8')
	# print(df.head())

	csv_file = open('mturk_results/best_worker_post-edit_data.csv', 'w', encoding='utf-8')
	writer = csv.writer(csv_file)
	writer.writerow(['PRODUCT','PRODUCT ID','QA1','QA2','QA3','QA4','QA5','QA6','QA7','QA8'])
	
	worker_eval = {}
	# item_error = {}
	item_worker_eval = {}
	# valid_worker_data = {}
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
	item_summary_file = 'item_summary.json'
	item_sum = {}
	worker_list, item_list = [], []
	# check_list = []
	print(len(df))
	for index, row in df.iterrows():
		# print('============')

		lcs_sim = row['LCS(Q-A)']/len(row['Question']+' '+row['Answer'])
		clean_q = re.sub(r'[^\w\s]','',row['Question'])
		# first iitem
		if index == 0:
			# cur_worker = row['Worker Id']
			# cur_item = row['Product Id']
			worker_list.append(row['Worker Id'])
			items_list.append(row['Product Id'])
			print(worker_list[-1], items_list[-1])

		if row['Worker Id'] != worker_list[-1]:
			all_worker_data.append((error/qa_count,worker_data))
			error, qa_count = 0, 0
			worker_data = []
			cur_worker = row['Worker Id']
			# check current product
			if row['Product Id'] != cur_item:
				# print(cur_worker, cur_item)
				# print(all_worker_data)
				all_worker = sorted(all_worker_data, key=lambda x:x[0])
				test = [k for k, v in all_worker]
				# print(cur_item, test)
				best_worker_data = sorted(all_worker_data, key=lambda x:x[0])[0][1]
				# print(best_worker_data)
				# select the best one worker per item and output
				prod,prod_id,worker_id,qa_edit = '', '', '', []
				for p,p_id,w_id,ques,ans,qa_pair,q_edit in best_worker_data:
					prod = p
					prod_id = p_id
					worker_id = w_id
					qa_edit.append(q_edit)
				writer.writerow([prod,prod_id]+qa_edit)
				item_sum[cur_item] = ' '.join(qa_edit)
				cur_item = row['Product Id']
				all_worker_data = []
		else:
			cur_worker = row['Worker Id']
			error = 0

		# 1. LCS
		if (lcs_sim > 0.7 or lcs_sim < 0.3):
			error += 2
			# print(lcs_sim)
			# print('error1')
			# print(row['Q-A post-edit'])
		# 2. Yes/No
		if (set(row['Q-A post-edit'].split()) & set(yn_list)):
			error += 2
			# print('error2')
			# print(row['Q-A post-edit'])
		# 3. QA format
		if (row['Q-A post-edit'].split()[0].lower() in q_term) or ('?' in row['Q-A post-edit']):
			error += 2
			# print('error3')
			# print(row['Q-A post-edit'])
		# 4. It/it
		if (row['Q-A post-edit'].split()[0] in third_prons):
			error += 1
			# print('error4')
			# print(row['Q-A post-edit'])
		# 5. First pronouns
		if (set(row['Q-A post-edit'].split()) & set(first_prons)):
			error += 2
			# print('error5')
			# print(row['Q-A post-edit'])
		# 6. ignore Q (maybe less error weight)
		if not any(ext in row['Q-A post-edit'].split() for ext in clean_q.split()):
			error += 1
			# print('error6')
			# print(row['Q-A post-edit'])
		

		qa_count += 1
		worker_data.append((row['Product'],row['Product Id'],row['Worker Id'],row['Question'],
								 row['Answer'],row['Question']+' '+row['Answer'],row['Q-A post-edit']))

		# worker total eval
		if row['Worker Id'] not in worker_eval:
			worker_eval[row['Worker Id']] = [error]
		else:
			worker_eval[row['Worker Id']].append(error)

	# exit()
	for key, val in worker_eval.items():
		worker_eval[key] = float("{:.3f}".format(sum(val)/len(val)))

	dictlist = []
	for key, value in worker_eval.items():
		temp = (key,value)
		dictlist.append(temp)

	# pprint(dictlist)

	dictlist = []
	for k, v in sorted(worker_eval.items(), key=lambda kv: kv[1], reverse=True):
		dictlist.append(k)

	# print(dictlist)
	# json.dump(item_worker_eval, open("worker_error_eval.json", 'w'))

	json.dump(item_sum, open('../item_summary.json', 'w', encoding='utf-8')) 


def hit_id():

	a = ['32L724R85LIVR38TDR6JM8YZKJNIPT', '38LRF35D5LUTT5Y69AYQS8J99II3UA', '3UYRNV2KITX2ZCK3OQH05UZQH8H8NF', '38VTL6WC4ABDOT5FXUJ8AQN6J2K5Y0', '3GKAWYFRAPREJSS7LD58VBIMSEFDPM', '3MZ3TAMYTLLG3GO8QJA8R3YEL1BIRL', '3RWB1RTQDJL22XWG45US9J4LHNI8PB', '3QGTX7BCHP0DTJFKTGLROXW8PWO5ZR', '3S8APUMBJXH9DI5TTEIXORGINNGBFA', '3UQ1LLR26A6QU0AX9BLZOT1W21HLAU', '3MYASTQBG79ZJ4TMLQKXVG8WXA1DQD', '3MVY4USGB6LS4VOL58ADHD3A73UIS9', '388FBO7JZRRHEIBL1UFU47NSANJYNI', '3OREP8RUT29FRI1O4YOF6CGUO2GBG9', '3UXQ63NLAAKVDAFPFPLVJ4L2KJ2LBP', '3J5XXLQDHM9T6KN6E08VOULSNPR3VZ', '3IQ9O0AYW6XTJV10U8F3RIZW9M8ITY', '3XABXM4AJ13N29XV4I9H9OUVMJ48Q2', '3M4KL7H8KVLCHRUQQOM4O9QW8CM61J', '3D0LPO3EABXZW3BX29F7JWNX0XOYOX', '3BVS8WK9Q0TTN97JUMOIROTA3F8BI1', '30Y6N4AHYPUZV58MY59X06I1TW3DRE', '33BFF6QPI196L9NB4ADRAW46UYT3W9', '38RHULDV9YDLBC5UPDKEE26VDGQIWO', '3K1H3NEY7LX86FZ8IUFL124URNSGDL', '30U1YOGZGAUBWK8ZDU92QGNXFYMDS2', '3RBI0I35XE1AAEIBJBVFVTPM56T3Y8', '3SD15I2WD2S8RU85DS0NC25T6W063B', '3E22YV8GG14N54JKA7JBRGUG6IHNPE', '3WPCIUYH1A6CBKLE2UES0LJJHH0DTR', '39WICJI5ATQAF4SGFXXY90YOB0X3ZZ', '3H5TOKO3D9HT9QR9D656M7XZKWW64W', '3KG2UQJ0MJM85GKZAKY0RLKQBE3NQ5', '3Q7TKIAPOT8OS7D9TLC1EU92N4ELDW', '3B623HUYJ4OZFPC8WMYM4O9W47P8SR', '3WRKFXQBOB5P0H0U4E22ZZRBOOQIYN', '39XCQ6V3KY2B59V9RBOGRZ3DKSR651', '337F8MIIMZBYVF5UBHPPD6N87TA40T', '3ECKRY5B1QUP57AZ004LD60DUIUIZE', '3PEG1BH7AEPKT4X7UP9U448NVJXBKC', '3MQKOF1EE2M431P8XZJ3N5QILBIDWH', '3BPP3MA3TCITKDR9PGNLKCRNNGFLEP', '3J9UN9O9J3QOYXOFEOCTJ8CRPQ0J0Z', '31GECDVA9JK7ODVNWUYLJ5HDGN266H', '3AQN9REUTFE8S6K8C01R82BYWJIDYG', '3TL87MO8CMNLQRR5KR7R2UCNZHULF7', '3IVEC1GSLPXO5M9XA7GTXDV1UI6J12', '3NCN4N1H1GFL1AIAR21AWWPRPE2BNH', '3JY0Q5X05J4BIZ9ZKKGT7N7R4F8GGV', '3KI0JD2ZU1GQSD9SF43DPJH9BYM67G', '382GHPVPHSPUC74RU478F4LFA5U43O', '3P4C70TRMRFR4RN0VBD9KFCZ0WULG6', '3YCT0L9OMM7KIWZC2ON5MLZRT2ONSU', '32CXT5U14G1FU24CLRTMJPRKP4A8U7', '3SBNLSTU6U38XWUD8M3AM9K02DMDZ7', '362E9TQF2HOPEQFUA8GWSZK7JS0GIN', '3C8QQOM6JPZ50ITVLZDC5RPFF9MLIY', '3FI30CQHVKHDUPCISLFTUL35UOD6B2', '3ICOHX7ENC9GNH2N0N12MSZZKI4E09', '3U74KRR67MJLXWCRROSVWQVDVL2NTJ','3TKXBROM5T8Z5S5VIE1ZPCCNBL6IJK', '3UDTAB6HH6XZSLB6SCLGUEYOTMD90L', '3NSM4HLQNRST8DXRQCQHVFFVGQFQQM', '31KPKEKW4ABIGEQ3QWZCLKTEHMGB0X', '3HKIF5DF6YVW3PMR6EAJF6L4032G9V', '3ZQA3IO31BP26X0UQK8Y59CNTD5O1H', '3HO4MYYR12MSQOLOSCWOZBVTJA96UA', '306996CF6WIIN3BLMF3CZPCOMEMB10', '3Q7TKIAPOT8OS7D9TLC1EU92N53LDN', '37PGLWGSJT4UGCBD0Z1BQRC0QA6IKJ', '3TZDZ3Y0JS4ZZAWOOVPG8JHYYEJ91O', '31SIZS5W59DTKECR3RFH05P0CCHQRN', '3INZSNUD80OTR5C04O9B9LI7NBMD9I', '3OKP4QVBP2VA88WWYKOJB1WO6LKGAA', '3S8A4GJRD31S33AOX26TVXXCXHI6VZ', '3C8QQOM6JPZ50ITVLZDC5RPFFABILM', '3MQY1YVHS3IPCOTH5J3Q48MTEK0B2H', '31ODACBENUD5LTC4IGFMQFUWYE0QSB', '3OREP8RUT29FRI1O4YOF6CGUO35GB5', '3CESM1J3EI15ISATNCBPHZGQ4QK6W9', '3FJ2RVH25Z46OVEK7ZPUD2R3QKX925', '3XJOUITW8UP60TPJ7GKC0KQI0XEQT0', '3BPP3MA3TCITKDR9PGNLKCRNNH4LEG', '3IWA71V4TIEK7SFJXH7BJIUXDSQ6X2', '3PN6H8C9R4O7WOM5WUNB5GTRTT4DAX', '3LAZVA75NIP9VIIQ9O8CASMSLJJO2Y', '30EV7DWJTVT97X5T2DTD2W16FYK6Y8', '3VIVIU06FKAP60BGLBER5444FF1IMO', '3Z8UJEJOCZBV9DA3BZ3ZWCWVWYX93G', '3PUV2Q8SV42CFYRX28N70RDXBBPDBS', '3SU800BH86QL06487LAM5GCKJBLQUR', '35ZRNT9RUIWYG0E9DOMHT2RKRXJO39', '3IZVJEBJ6AI9CNFYYZVWG3A8LSO6ZZ', '3O0M2G5VC60GR9T7BD8I6HO1AYT941', '33EEIIWHK75LDLT8CBKR12E3XIUQVG', '3TL87MO8CMNLQRR5KR7R2UCNZIJLFY', '3VGZ74AYTGEY1NTUIYKAHWMC6O5GCC', '3WJGKMRWVI7VP3J3G8J2BBJFTWPDCZ', '3URJ6VVYUPLJYWXDD2R037JQ5XFO4U', '3P4C70TRMRFR4RN0VBD9KFCZ0XJLGX', '36KM3FWE3RAVE2NDDFNWGI7IYWP707', '3GS542CVJVLYNSX7PIRSB9UFAUO956', '3MJ28H2Y1E61UF1DR7AA8ZP45TAO5Z', '3K1H3NEY7LX86FZ8IUFL124UROHGDC', '36FQTHX3Z3PEX6P09JHVNIRLKY0B3S', '3T8DUCXY0N408U8XQE9QEP0TQXAT9H', '38B7Q9C28G3U6WXLU11X3F8F6PZ96M', '3E24UO25QZOXDJ1RWQKF05341OLO6F', '35NNO802AVUVOIWGXCTRIJT4K5BINO', '308KJXFUJR4EDDI2IKNQAKBDWFSTAW', '3RDTX9JRTYZDAWBQDB6P9T8B10J97L', '3X4Q1O9UBHKGHJFWF0P76J30WZ5O7E', '388CL5C1RJL54NIBFHVR5FO0FXWLHW', '3AFT28WXLF0D63WSYRT4X8T9AFGIO3', '3NKW03WTLM5YSATD2LPNN4XH4RWQWQ', '3L4YG5VW9NQ5UVPHG4EDVH1XEW1DDZ', '3C8QQOM6JPZ50ITVLZDC5RPFFABLIP', '35JDMRECC47DYLD8EPQ57KMFR0IGE5', '32L724R85LIVR38TDR6JM8YZKKCIPK', '3T2EL38U0MIDHAY3CQL9PNBODT2QXJ', '3TZ0XG8CBUIHAAG9NGYVNT46CKP988', '36MUZ9VAE60AM13HCZPX1ZJIE82DES', '3ZQX1VYFTD3KHXKFP5HDKJZV7JBO81', '3V7ICJJAZAEZF2849XMEXNJRYYWB4D', '38Z7YZ2SB30GRF98D4L8MDO9PGYIQB', '3NFWQRSHVEC54ZD490ABP27F31XGFN', '3OID399FXG52SF3D7A93JH4IQ9HDFA', '3NZ1E5QA6ZZHBLC4N25O2FP5YURB5I', '3K1H3NEY7LX86FZ8IUFL124UROHDG9', '3BKZLF990ZX2HFODHM7B81IXFZWQYP', '3LXX8KJXPW7KV23PUXNRKFU6RPTO9Q', '3JY0Q5X05J4BIZ9ZKKGT7N7R4GXGGM', '3FI30CQHVKHDUPCISLFTUL35UP2B6Y', '3F6045TU7DMHOFZJS849NPZHWQ799X', '3MZ3TAMYTLLG3GO8QJA8R3YEL20IRC', '3CMV9YRYP3Z1VTD13HYCHS4FSP0LJ1', '3SX4X51T807Y0LDUM31RGA5QX7BOA5', '335HHSX8CD3M6BUJ20X3M2GV6OUDH8', '32204AGAABAPIV4A4QYBSNJSJGAGHL', '3T5ZXGO9DEM2M5YID89UM8RZLT0QZG', '3YKP7CX6G2DWYPQNBVKL0Z31P0MB7X', '3M67TQBQQHMVTY9OKEI9JKA128P9AC', '30F94FBDNRIJWNNUOYNM5VVJEXDTBR', '3QREJ3J433VW6DJJL2YOI74S7E0LK0', '375VMB7D4JHM26538IFOMEHA619DI1', '362E9TQF2HOPEQFUA8GWSZK7JTPGIE', '3MVY4USGB6LS4VOL58ADHD3A74JIS0', '3S4TINXCC0L3JVIMSH1NBLPWFPWOB0', '37J05LC5AXHIXHP9Q00OYFWAJGYDJD', '3MDWE879UH00C8EGQSI5EVU7KQA9B7', '374UMBUHN5N26SF02YJHGF11WIDTCY', '3VAOOVPI3ZQ7QJ1162APX7H7WE5LL3', '3LOJFQ4BOXDD81VR8L00ZUWNY5YDKC', '3T2HW4QDUV5JMD6M4SE0PF0P2BA9CE', '3IQ9O0AYW6XTJV10U8F3RIZW9NXITP', '3EKTG13IZU1RW1JM6EB4XKWWWJVLM5', '3Q7TKIAPOT8OS7D9TLC1EU92N53DLF', '36GJS3V78VOL91Z0SQ1W40Z7W8EGJQ', '3INZSNUD80OTR5C04O9B9LI7NBM9DE', '3D1TUISJWIY8J8GPUD5DWELYS14IUG', '3CO05SML7V35WL7SMTL2LZYIZ8JR0Q', '3OPLMF3EU5LXEJ4MIFQ4AZLW195LN5', '3KL228NDMVKGKL5IAB185FZKBXEGKP', '335VBRURDJYYJBQ00JKVF30SNNN9E7', '3WPCIUYH1A6CBKLE2UES0LJJHIPTDY', '3ZTE0JGGCEQMT0AS6HXIM5VEXAWOC7', '3P4C70TRMRFR4RN0VBD9KFCZ0XJGLS', '3L1EFR8WWT3QPPQWVU41XLLSZO29FP', '3OLZC0DJ8JD8WN5PZ3FIS0NH68DIV5', '3BAKUKE49HA53ASAICP2Z4HS40PR1T']
	b = ['3OQQD2WO8I4OKE36FT9C95VYFO53IE', '3H5TOKO3D9HT9QR9D656M7XZKX6648', '3NZ1E5QA6ZZHBLC4N25O2FP5YUCB53', '3QE4DGPGBR9V6JGFPJESYYAOEB9G4K', '39XCQ6V3KY2B59V9RBOGRZ3DKT165D', '3NCN4N1H1GFL1AIAR21AWWPRPFCNB5', '3FI30CQHVKHDUPCISLFTUL35UPNB6J', '3ZUE82NE0AZQBK7MQ2YNXPQHFI58FK', '3MYASTQBG79ZJ4TMLQKXVG8WXBBDQP', '3DIIW4IV8PT92AMAF7EVJAN4TO1I4E', '3I6NF2WGIGUD22KF3OX23QG2E74G5P', '3O4VWC1GEW4KFPNCXBUCL6AYS3U3JQ', '3SV8KD29L4QGB20N03PXQMFQMNTKZ3', '3VDI8GSXAFRWPK3H1M45FAQTGX58GJ', '30Y6N4AHYPUZV58MY59X06I1TXDDRQ', '35A1YQPVFEERYTQATCX5O2TITKWI5J', '31GECDVA9JK7ODVNWUYLJ5HDGOC66T', '32XN26MTXZHUTNCFH9O6ZNPJTGFL0F', '3U18MJKL1UK4BFAG52X57GV970CNCC', '3EHIMLB7F7XAPGYSLSMN0A2UVXI8HI', '3XT3KXP24ZWNHXQOYV7AG87IPF7I6Z', '329E6HTMSW0FQ9TUFWUOMLAB7SU3KP', '3APP19WN71C9L6KT8777VWU2A2FG65', '3KI0JD2ZU1GQSD9SF43DPJH9BZW67S', '3YKP7CX6G2DWYPQNBVKL0Z31P07B7I', '31J7RYECZLOU0CXXDSS6DS8TY8LL1I', '30U1YOGZGAUBWK8ZDU92QGNXFZWDSE', '3GVPRXWRPHS6LX4TH5C2MM7EKQRI7Y', '37SOB9Z0SSVQAFBC0W6P1LNQWSZ3LS', '3JMQI2OLFZ3EG7GU5YSGRMDRS0ONDC', '3WPCIUYH1A6CBKLE2UES0LJJHIADT3', '3Q2T3FD0ON6AGXTX08741Y2FWXP3MU', '3M47JKRKCXZUSREBP9VJ3JD4MJ268F', '306W7JMRYYW0Y3V6L0CREZZW0KDB85', '3R0WOCG21M7RBX032Z425H5L0WHDUU', '3TRB893CSJ8SP6YYRHCZ1AUY5DZG74', '38B7Q9C28G3U6WXLU11X3F8F6PK694', '3N3WJQXELSO1PXFTWWSKIBIYQEZL2Z', '3MDWE879UH00C8EGQSI5EVU7KQVB9U', '344M16OZKIDJ8DUU1T30X4VCSCPNE5', '32K26U12DNMROCP37PE71374E3QDVJ', '3VDI8GSXAFRWPK3H1M45FAQTGX5G8R', '307L9TDWJYQGYFEXC9M4EDRF1NZ3NU', '37SOB9Z0SSVQAFBC0W6P1LNQWSZL3A', '3TD33TP5DL0EHROLIYW5AQ5RQ8DBA9', '3M0556243SIBERUQW4N6FMGC4D4NFN', '3HKIF5DF6YVW3PMR6EAJF6L403NG9G', '35ZRNT9RUIWYG0E9DOMHT2RKRX43O9', '3IHWR4LC7DBALB9CRA480M39VAX8IB', '3WKGUBL7SZKBSBUG0AB8BQFWASVL4V', '3TKSOBLOHLEJ01TDOCW151PX8QYBB4', '3IJ95K7NDXAHSRQL7OTOX7GO5S4NGM', '3IHWR4LC7DBALB9CRA480M39VAXI8L', '3X55NP42EOEG10QASOZWI2WA1203PQ', '3OCZWXS7ZO5TOUYGEFUIGILAAOQL50', '3MQKOF1EE2M431P8XZJ3N5QILCSDWT', '3FBEFUUYRK38BF7QM7FXZAJZC726AJ', '31N9JPQXIPGVSNLWRUB6I7SPKSHNHL', '3OKP4QVBP2VA88WWYKOJB1WO6L5GAV', '3IV1AEQ4DRB6GMTI9SP8CNI98PM8JN', '3GV1I4SEO9NP7YYUJY4N8OZA6J1L6G', '309D674SHZJ2A6LJ2CSWGLVFQBYBCB', '3FI30CQHVKHDUPCISLFTUL35UPN6BE', '34OWYT6U3WFAZGSMW2AM0IYKFGFI9A', '33J5JKFMK6W11CRPS1ELI7MK6YM3QH', '3OREP8RUT29FRI1O4YOF6CGUO3QGBQ', '3W0KKJIARR71R6Z0RDPKD2IMNEM8KM', '3ZXNP4Z39RJ8BYCZ289FE2Z61ULL7F', '35NNO802AVUVOIWGXCTRIJT4K5WNIE', '3VGZ74AYTGEY1NTUIYKAHWMC6OQGCX', '3PUV2Q8SV42CFYRX28N70RDXBBABDB', '3M7OI89LVYMW4U4O6LBO559NCAN6CL', '351S7I5UG9URJTGMFUERUK84XKLNJQ', '3BO3NEOQM0FO4Z2RO8OMWD94LYXIAP', '3K1H3NEY7LX86FZ8IUFL124URO2GDX', '31JUPBOORN2CBCHICD1LS2V1CER8LP', '31JUPBOORN2CBCHICD1LS2V1CERL82', '3S829FDFT2ZJS1UY74FPPO4PUEYDXM', '3BS6ERDL93569MA26H6ZPBR5XAZ6DL', '3J6BHNX0U9QMUDM4XFE3VZ8HC9LNKP', '3HJ1EVZS2OH1DD6P5G3LNXWP2KO3RI', '35JDMRECC47DYLD8EPQ57KMFR03GEQ', '3AQN9REUTFE8S6K8C01R82BYWKSDYS', '3WA2XVDZEMFB1SO22CHJVT9QXM06EE', '3OPLMF3EU5LXEJ4MIFQ4AZLW19QNLS', '3HFWPF5AK9HDES62K53QD71LOM73S6', '3NFWQRSHVEC54ZD490ABP27F31IGF8', '3ACRLU860NCH745XY3YR69VIBNBBE4', '3SBNLSTU6U38XWUD8M3AM9K02EWDZJ', '3XAOZ9UYRZP5R9DA5X02UE7N8E8Q1C', '3BVS8WK9Q0TTN97JUMOIROTA3GIIBK', '3JY0Q5X05J4BIZ9ZKKGT7N7R4GIGG7', '3S8APUMBJXH9DI5TTEIXORGINOQBFM', '3NQUW096N66CPH0SH57ZSYQCWK9L9R', '3E6L1VR4XWK376OYXN1PDBUQ9NF6FW', '37ZQELHEQ0WHK1M7IRRJAC0L1EGNMU', '3IKDQS3DQEYCXEZP8MKD28ZSL1IICR', '32204AGAABAPIV4A4QYBSNJSJGVGH6', '3OREP8RUT29FRI1O4YOF6CGUO3QBGL', '3H4IKZHALBGN2J77US6JNRPL64QNNU', '3APP19WN71C9L6KT8777VWU2A2F6GV', '362E9TQF2HOPEQFUA8GWSZK7JTAGIZ', '3KTZHH2ONIDWHUZ3CP20SFAQCJH8MR', '3ICOHX7ENC9GNH2N0N12MSZZKJEE0L', '3UQ1LLR26A6QU0AX9BLZOT1W22RLA6', '3JUDR1D0D6PCGUV6O10GZXHS0KMQ2T', '3UYRNV2KITX2ZCK3OQH05UZQH9R8NR', '3HY86PZXPYGGU6N5W6520XI9PBKE1O', '3MWOYZD5WVM5K47JV76W2GPQWEVNO9', '3UXQ63NLAAKVDAFPFPLVJ4L2KKCLB1', '33J5JKFMK6W11CRPS1ELI7MK6YMQ34', '36GJS3V78VOL91Z0SQ1W40Z7W8ZGJB', '3ZQX1VYFTD3KHXKFP5HDKJZV7JW8O6', '33IXYHIZB5GNJR51FA5G5GSEHHYE25', '3E22YV8GG14N54JKA7JBRGUG6JRNPQ', '375VMB7D4JHM26538IFOMEHA61UIDR', '3SBX2M1TKDLMJ8ATSFJ4SCEQKYIQ4P', '3RWB1RTQDJL22XWG45US9J4LHOS8PN', '31MBOZ6PAOPENF7VTPHQUORK25CLC8', '3KL228NDMVKGKL5IAB185FZKBXZGKA', '3TTPFEFXCTINL2F4SDPPGW63P2S6HU', '3SNR5F7R92RRUCJ34DQ8SWZV6DVIEK', '3XABXM4AJ13N29XV4I9H9OUVMKE8QE', '3N7PQ0KLI5NC491KJAJLOQX6NVYE3G', '3KG2UQJ0MJM85GKZAKY0RLKQBFDNQH', '3Q7TKIAPOT8OS7D9TLC1EU92N5OLD8', '37VE3DA4YUFTREWZO46XRCSV333BHK', '3P4C70TRMRFR4RN0VBD9KFCZ0X4GLD', '3XT3KXP24ZWNHXQOYV7AG87IPF76IN', '3K3G488TR264FRET6K2EX4K4KUDQ5U', '3BA7SXOG1JONEACVHXYHEE40I6G8RF', '3CZH926SICCXM5KOJOO4YVPC1VUE41', '3BPP3MA3TCITKDR9PGNLKCRNNHPLE1', '38EHZ67RIMQBA95LVNEOKSRO02UGMF', '3X7837UUADWJC8AUGDSAS9MI2UW6JZ', '3B623HUYJ4OZFPC8WMYM4O9W48Z8S3', '34R0BODSP1XFIOOOXT7E3NVQ1RPE56', '3YGYP1364178HHZZNZN0WBUV71FNRI', '3TL87MO8CMNLQRR5KR7R2UCNZI4LFJ', '3IJ95K7NDXAHSRQL7OTOX7GO5S4GNF', '3AJA9FLWSCWJ0QJZZOAEAEKVIEAIF2', '3BCRDCM0ODSENSGCYYSMTOMVHJW6KY', '371DNNCG4400UPPNLM3CET5I6RD8TS', '3BVS8WK9Q0TTN97JUMOIROTA3GIBID', '362E9TQF2HOPEQFUA8GWSZK7JTAIG1', '3BFF0DJK8XAID94WZK9HVK5JILKSTC', '32CXT5U14G1FU24CLRTMJPRKP5K8UJ', '3P4C70TRMRFR4RN0VBD9KFCZ0X4LGI', '3P6ENY9P79U3EMA5UEYEDZW8YTNIH0', '3WA2XVDZEMFB1SO22CHJVT9QXM0E6M', '3NBFJK3IOHGZACQX83T1CWGTV29GOU', '388CL5C1RJL54NIBFHVR5FO0FXHLHH', '3YCT0L9OMM7KIWZC2ON5MLZRT3YNS6', '3T6SSHJUZF83AHLP0WGZDBXNY62IIT', '36QZ6V1589BXDMJLZPZR0GRL1ZRSU3', '3DW3BNF1GHGF7HTCQH3RFBT33CT8V8', '3C8QQOM6JPZ50ITVLZDC5RPFFAWLIA', '3TKXBROM5T8Z5S5VIE1ZPCCNBLRIJ5', '3HA5ODM5KAQXQ18L4F9WW2T4F60SVS', '3FCO4VKOZ4BU5S27LMMB179MSXKE7L', '3X2LT8FDHWGSM6THGR8N1DCHALV8WI', '3B9XR6P1WETPIKRPC49I3P8AGV7BJP', '3CMIQF80GNO0YVE7B3CJPAY4GPOQ6A', '31GN6YMHLPQA5Q8QUPESI4CIMF2SW2', '3HYV4299H0UY567QVREHF75H3HQE88', '3FHTJGYT8NYHVC2YN36G1WLJ575GPB', '3PEG1BH7AEPKT4X7UP9U448NVK7BKO', '3CMV9YRYP3Z1VTD13HYCHS4FSPLLJM', '3VO4XFFP15KJ2VSCUDHBVOY0B08Q79', '37PGLWGSJT4UGCBD0Z1BQRC0QARIK4', '335VBRURDJYYJBQ00JKVF30SNN8E9X', '3UXQ63NLAAKVDAFPFPLVJ4L2KKCBLR', '3XABXM4AJ13N29XV4I9H9OUVMKEQ8W', '3A520CCNWNYCOU05SPYVBYBCT5QEAC', '33K3E8REWWT7B6Y7QW493WQOJN18XB', '3LVTFB9DE5G2VO3DNGL511BTA3RGQ2', '3QREJ3J433VW6DJJL2YOI74S7ELLKL', '3U74KRR67MJLXWCRROSVWQVDVMCNTV', '3D7VY91L65VFJSXAF1MAJH0RKP2BMT', '3JHB4BPSFK7NGEG59AFV9KP66QWQ9L', '37Y5RYYI0P3PUQDG4UAEKNQPVH8SXV', '3ACRLU860NCH745XY3YR69VIBNBEB7', '3L2OEKSTW98WBBOHVSQBMAXXLTV8YH', '3ZVPAMTJWN127PID0VA56RLY6PTGR3', '3PIOQ99R7YK0X9RGRTI51MHFE0JNUM', '3C8QQOM6JPZ50ITVLZDC5RPFFAWIL7', '3PGQRAZX02IEUV3Q9QWG31XYXN2SY1', '3H1C3QRA01H0H9X3C3UMHT10T8BECE', '33NOQL7T9OXWG1YMRESU0H6ZRNZ8Z8', '3VAOOVPI3ZQ7QJ1162APX7H7WEQLLO', '302U8RURJZZ0AOGGWJSAX8JYS7SNVB', '3VIVIU06FKAP60BGLBER5444FFMIM9', '36MUZ9VAE60AM13HCZPX1ZJIE8NEDE', '3UDTAB6HH6XZSLB6SCLGUEYOTMY906', '3QHITW7OYO71LXQA1GTV5F0QC8EQA0', '3EKTG13IZU1RW1JM6EB4XKWWWJGLMQ', '3ZRKL6Z1E81E84IQFKAAW1QUSRCGSR', '35NNO802AVUVOIWGXCTRIJT4K5WIN9', '371Q3BEXDH7EZLDV5CYZH8603H6SZS', '3QO7EE372OL647V27UTR0QKWUQZQBV']
	c = ['3R16PJFTS3P085CYFAZ7WQ2HLTAK4D', '3MYASTQBG79ZJ4TMLQKXVG8WXCVDQB', '3VCK0Q0PO5CKHNF0TM4LCFT8YCZN05', '3MNJFORX8B27O2OKS4RKL5GQDTO5F1', '3TLFH2L6Y9MP8ANY55UB42SFKS52T5', '3QHITW7OYO71LXQA1GTV5F0QC9YAQ6', '3OND0WXMHWDRYH8JYV8NSKYG610HEC', '30Y6N4AHYPUZV58MY59X06I1TYXDRC', '3I6NF2WGIGUD22KF3OX23QG2E8O5G0', '3OWZNK3RYLN48N2N5AKL9YEH36C2UW', '36JW4WBR06IJ4V8FT6STA2JGI2FHFU', '3UY4PIS8QRJKOC0IP58LQKCI345N18', '3G9UA71JVVS2G8OZZNX2YNMEX607JL', '31ANT7FQN80R2YFQNUFKOQS3T815HZ', '3ZG552ORAM24L2RNA0UQ5KG0HDL2VL', '32204AGAABAPIV4A4QYBSNJSJHFHGT', '3JTPR5MTZSAI4OGYTFIH1I8VLP5K5I', '3GITHABACYJRDXIE898ZV3MNVAJN2P', '3ZURAPD288L80KN1RHP8IF391DJF1T', '35A1YQPVFEERYTQATCX5O2TITLG5IS', '3JMNNNO3B12H0RRS0AZMRMZEOMN2WV', '3BCRDCM0ODSENSGCYYSMTOMVHKGK6Y', '30U1YOGZGAUBWK8ZDU92QGNXF0GDS0', '307L9TDWJYQGYFEXC9M4EDRF1OJN30', '35O6H0UNLSENT4AGBUI5038I6055J4', '34HEO7RUG6S1XY5AEVIVA5AV8V0AR7', '3L60IFZKF3G3IRZLOWGTDNVTYHSHHS', '3UEDKCTP9VOXRSUHH8XEZ2MRCV0K7X', '3UEDKCTP9VOXRSUHH8XEZ2MRCV07KK', '3WPCIUYH1A6CBKLE2UES0LJJHJUDTP', '3PZDSVZ3J5F1GBX1CNRNOIJLFOFN4L', '34D9ZRXCYRSDYD5NTKI00FFRUXJASV', '3P458N04Q1FWPRWIAFV8T5DLXOT2XO', '3P6ENY9P79U3EMA5UEYEDZW8YU7HIL', '3W0KKJIARR71R6Z0RDPKD2IMNF6K8K', '3LEG2HW4UFLFP55XALPMNYDETJXF2A', '3R0WOCG21M7RBX032Z425H5L0X1DUG', '308KJXFUJR4EDDI2IKNQAKBDWGXATK', '37MQ8Z1JQEULPWMSFBHACJKUZUN2YU', '3ZXNP4Z39RJ8BYCZ289FE2Z61V57LN', '3I7KR83SNAB15BIAW5VYDYDX7LOK99', '3538U0YQ1FS4AN1GEL3R68I6ZXXF3L', '3JTPR5MTZSAI4OGYTFIH1I8VLP55K3', '3HRWUH63QU0JCU11QSAXTAPZFKAN5Q', '3VJ4PFXFJ35TDQXRIPD0FGXFFU4AUB', '3P7QK0GJ3TJLUMWXBXJTQQTW5OR2ZL', '3PKJ68EHDNUZ9XUBCWJEP0B8B9WHJX', '3P7RGTLO6EBFAUSFOB9Y9TOHD36KAO', '3OCZWXS7ZO5TOUYGEFUIGILAAPA5L6', '39AYGO6AFFIFVY1FVBK2LG3ZBFLN66', '3EFNPKWBMSMDDJMQB3K631V2TV8305', '33P2GD6NRNQUKH0TUHJQQFBLQYWHKW', '3UV0D2KX1MHPSJKKEZ8AGDACDXTF46', '37M4O367VJGDUCG1ERVXGV0ZAU05M8', '3SCKNODZ0XEYZYFKELPURU3V6Q5N75', '363A7XIFV4KTQ5MRNFN5B2ZYT1DAV0', '3D17ECOUOETDK8787MO6H6ECYNE318', '3MNJFORX8B27O2OKS4RKL5GQDTOF5B', '3HRWUH63QU0JCU11QSAXTAPZFKA5N8', '3UYRNV2KITX2ZCK3OQH05UZQHABN8S', '3I7SHAD35MUSHGUK2KAUEFEV10V7MP', '388CL5C1RJL54NIBFHVR5FO0FY1HLZ', '3E6L1VR4XWK376OYXN1PDBUQ9OZF6R', '3MJ28H2Y1E61UF1DR7AA8ZP45UF5ON', '3PEG1BH7AEPKT4X7UP9U448NVLRKBJ', '32K26U12DNMROCP37PE71374E4ADV5', '3SCKNODZ0XEYZYFKELPURU3V6Q57NP', '3ZLW647WALTK9TP4QQOKMPOHQTS32P', '3RIHDBQ1NEWPA50WFTW65S3PF3RHM1', '3X878VYTIEGMB623GX6HJPUM4ZJF7Q', '3EPG8DX9LKOJFFDE67NPXZUUFZB5P4', '3G5RUKN2EC12DH3DTINE5QU11GTN9H', '3Q9SPIIRWJK65UMWDPS1X4IC0AFAWA', '3JAOYN9IHL09UBLNUQ2P5ZT9W7S330', '3ZUE82NE0AZQBK7MQ2YNXPQHFJPF8D', '3N5YJ55YXG1GI0DILO1E1L5L7YBNAW', '3X4Q1O9UBHKGHJFWF0P76J30W0A7O4', '3MQKOF1EE2M431P8XZJ3N5QILDCDWF', '3WRAAIUSBJXLUURMNUONZNWJ9CLAX3', '3K3G488TR264FRET6K2EX4K4KVX5QV', '3L1EFR8WWT3QPPQWVU41XLLSZP7F92', '3W31J70BASU339PD8P5PFOE5D6RKCQ', '3E9VAUV7BWCAUZHWSQAPI13SBIFAY9', '3Y3CZJSZ9KR4RSTTJZRE2UU9GHZ5RW', '3NCN4N1H1GFL1AIAR21AWWPRPGWNBR', '3LOJFQ4BOXDD81VR8L00ZUWNY63KDQ', '3WUVMVA7OB1AZPR1OCC8W8CUHCJAZ0', '382GHPVPHSPUC74RU478F4LFA7O34L', '3YZ7A3YHR5RGS7T6YORJS4Z52JI5SK', '31N9JPQXIPGVSNLWRUB6I7SPKT1HN1', '366FYU4PTGNI079R4GBK5CE8YI4KEJ', '31KPKEKW4ABIGEQ3QWZCLKTEHNLB04', '30UZJB2POHAC8Q8R89QIKWRTA3J35Q', '3S1L4CQSFX34U801N0I1TGWC57PFAH', '3UUIU9GZC53H776LNOW929VR42W5T9', '36FFXPMST9MDA8L8S9BJXWSUA36HOG', '3O2Y2UIUCQSA6L9NZRVQNUZ8AJJKF1', '3S829FDFT2ZJS1UY74FPPO4PUFIDX8', '306996CF6WIIN3BLMF3CZPCOMFRB17', '3AQN9REUTFE8S6K8C01R82BYWLCDYE', '3YLTXLH3DF4VV8X979OYMWXKK82HPX', '3MQY1YVHS3IPCOTH5J3Q48MTEL5B2O', '3P520RYKCH4W7KLANTMJ75HTNG35U0', '3SD15I2WD2S8RU85DS0NC25T6YU366', '36FQTHX3Z3PEX6P09JHVNIRLKZ5B3Z', '3PA41K45VN2Y2JRXU02MVJ8Q6567PL', '3U18MJKL1UK4BFAG52X57GV971WNCY', '3V7ICJJAZAEZF2849XMEXNJRYZ1B4K', '3S8APUMBJXH9DI5TTEIXORGINPAFBC', '3SBNLSTU6U38XWUD8M3AM9K02FGDZ5', '3JMQI2OLFZ3EG7GU5YSGRMDRS18NDY', '3BFNCI9LYKORVUMAW25FIG5P19E375', '3KL228NDMVKGKL5IAB185FZKBYJKG0', '30P8I9JKOIJWKZAASJWO3RJC1NC5VP', '3ICOHX7ENC9GNH2N0N12MSZZKKYE07', '34ZTTGSNJXMGVKYO7M3NM1NUP4OHQO', '3NZ1E5QA6ZZHBLC4N25O2FP5YVWB5P', '33P2GD6NRNQUKH0TUHJQQFBLQYWKHZ', '344M16OZKIDJ8DUU1T30X4VCSD9NER', '3HY86PZXPYGGU6N5W6520XI9PC4E1A', '3D1UCPY6GG7VV8RT67XLWG1KCTK38S', '3ZXV7Q5FJBMSNNXZ7EESZBM05AAFCJ', '3KVQ0UJWPXJ9ZOAFIT1KPT2Q8WE5WZ', '33IXYHIZB5GNJR51FA5G5GSEHIIE2R', '3IZPORCT1F7G7LDOK1SNRRXZLQQHRP', '3VO4XFFP15KJ2VSCUDHBVOY0B1S7QC', '3QD8LUVX4XWOOOF5SYX6RCGXHYK5XS', '3IVKZBIBJ07S80D1ZQSSH12V7S9HSD', '3OID399FXG52SF3D7A93JH4IQAMFDJ', '37PGLWGSJT4UGCBD0Z1BQRC0QBBKIS', '3M0556243SIBERUQW4N6FMGC4EONF9', '3YLTXLH3DF4VV8X979OYMWXKK82PH5', '373L46LKP74QBNVJIHMB2SR03Q0KJ4', '3FI30CQHVKHDUPCISLFTUL35UQ7B65', '38VTL6WC4ABDOT5FXUJ8AQN6J4E5YY', '3909MD9T2ZF7KLHD35KNPZM3QMNFEC', '3N7PQ0KLI5NC491KJAJLOQX6NWIE32', '3YKP7CX6G2DWYPQNBVKL0Z31P1RB74', '39O0SQZVJN5JEW7C7S6B0E857NU7RD', '32L724R85LIVR38TDR6JM8YZKLHPIY', '3CZH926SICCXM5KOJOO4YVPC1WEE4N', '3L84EBDQ370LM71102MN37RDIF0KK3', '3EQVJH0T40JTN0QGOQXIR6YH9BNHT2', '32ZCLEW0BZIRMESZV9RJY9DZX06PJA', '3QGTX7BCHP0DTJFKTGLROXW8PYI5ZP', '34R0BODSP1XFIOOOXT7E3NVQ1S9E5S', '3IJ95K7NDXAHSRQL7OTOX7GO5TONG8', '391FPZIE4CK8ND55OVNSW2KJSPUHUT', '39KV3A5D185VFB7PMH6GQOD1TPD7S1', '3WA2XVDZEMFB1SO22CHJVT9QXNKE68', '3RWSQDNYL9KZQZH9YG4T7H732N2FFU', '306W7JMRYYW0Y3V6L0CREZZW0LXB8R', '3KLL7H3EGDZ80SU5TLXXSOM26W3HVI', '35F6NGNVM8HWUBK4BHB60T9NV8R7TQ', '3FCO4VKOZ4BU5S27LMMB179MSY4E77', '3MDWE879UH00C8EGQSI5EVU7KRFB9G', '34R3P23QHSZLFHUAJV2TEQ5GD55HWS', '3NI0WFPPI9ECA298U5I4A47M3MA60E', '3NFWQRSHVEC54ZD490ABP27F322FGT', '3HYV4299H0UY567QVREHF75H3IAE8U', '31N9JPQXIPGVSNLWRUB6I7SPKT1NH7', '3A9LA2FRWSC04HZ0T0YFG9JNM7BHXL', '3G4VVJO6PZEMXYYHDURVZODCCP6PK9', '3XD2A6FGFNSZTUVI1E404K47OF1S9Y', '3QREJ3J433VW6DJJL2YOI74S7F5KL6', '35NNO802AVUVOIWGXCTRIJT4K6GNI0', '30QQTY5GMKIBUOZTBM1G5PVPEMY7UH', '34D9ZRXCYRSDYD5NTKI00FFRUXJSAD', '3TD33TP5DL0EHROLIYW5AQ5RQ9XBAV', '3LN50BUKPV9XH4GZYU3WEOQR1PBPLC', '3BAWBGQGYLXB73OTGCBL1BX8ST77V6', '36JW4WBR06IJ4V8FT6STA2JGI2FFHS', '335VBRURDJYYJBQ00JKVF30SNOSE9J', '3TKSOBLOHLEJ01TDOCW151PX8RIBBQ', '3SR6AEG6W5RP4MPAYWKHZNQWOD5HYR', '34XASH8KLQKHNMYKY64BE15G1U1PME', '34KYK9TV2R6IHNAFZYIWVQZXCF4SB8', '391JB9X4ZY6GCV14LEZ3IKJH7KVKM8', '309D674SHZJ2A6LJ2CSWGLVFQCIBCX', '3E22YV8GG14N54JKA7JBRGUG6KBPNE', '3B9J25CZ25B1RS2LDYER6A5FU04SCF', '3J6BHNX0U9QMUDM4XFE3VZ8HCA5KN8', '3PUV2Q8SV42CFYRX28N70RDXBCUBDX', '3JU8CV4BRLA5NPJWBMJO65ULWUGPOT', '3AC6MFV69KGP9CZFUIM0DUZYU79HZI', '351S7I5UG9URJTGMFUERUK84XL5NJC', '3ACRLU860NCH745XY3YR69VIBOVBEQ', '3JVP4ZJHDPQS3M49W6RT77XRCD5I0K', '3OYHVNTV5TW4CYMGYUEGAO8M2KAKON', '3A520CCNWNYCOU05SPYVBYBCT6AEAY', '3IH9TRB0FBXSABPRSPVTLCG1H5BI1N', '3J6BHNX0U9QMUDM4XFE3VZ8HCA5NKB', '30U1YOGZGAUBWK8ZDU92QGNXF0GSDF', '341YLJU21IXZZW7NBTV7QVQ69BPI24', '3VGET1QSZ0XOMSOY6MGHNDGMZ297WG', '3B0MCRZMBRSN8PVXQMW3V5ZB6ZCPPA', '3S8APUMBJXH9DI5TTEIXORGINPABF8', '3ACRLU860NCH745XY3YR69VIBOVEBT', '3LCXHSGDLT4GOQMZ9PKMWY5IFCHSE8', '3OPLMF3EU5LXEJ4MIFQ4AZLW1AANLE', '3H1C3QRA01H0H9X3C3UMHT10T9VEC0', '31YWE12TE0A3BSTOGRC3PWUT84F7X9', '338GLSUI4398U4MV404SEGQIRDWSFQ', '3OREP8RUT29FRI1O4YOF6CGUO4ABG7', '3HEM8MA6H9A881WCQZBSVAPLBVYPQ1', '3AJA9FLWSCWJ0QJZZOAEAEKVIFUFIL', '3OQQD2WO8I4OKE36FT9C95VYFPPI3F', '3G4VVJO6PZEMXYYHDURVZODCCP6KP4', '3DWGDA5POF2XWH9RRWUC016VP77V1X', '3ZRKL6Z1E81E84IQFKAAW1QUSSWSGP', '37VE3DA4YUFTREWZO46XRCSV34NBH6', '3AXFSPQOYQWFV135H6VEMFZVVUJFJX', '37ZQELHEQ0WHK1M7IRRJAC0L1F0NMG', '3MIVREZQVHW7XAZWD76KZT3MHLSKQV', '3JGHED38EDPSBXJYLNY58A12AA97YF', '3BVS8WK9Q0TTN97JUMOIROTA3H2BIZ', '3VEI3XUCZRV8K2BC3E0S00ZQ7H0PR2', '3O2Y2UIUCQSA6L9NZRVQNUZ8AJJFKW', '3H4IKZHALBGN2J77US6JNRPL65ANNG', '30IRMPJWDZH79BEWQMVK4JDRD7UKRW', '3ZG552ORAM24L2RNA0UQ5KG0HDLV2E', '3IVKZBIBJ07S80D1ZQSSH12V7S9SHO', '3TL87MO8CMNLQRR5KR7R2UCNZJOFLZ', '3DIIW4IV8PT92AMAF7EVJAN4TPLI40', '311HQEI8RSESGNT3H90OMHA4G4D7Z6', '3J5XXLQDHM9T6KN6E08VOULSNRLV3P', '3VADEH0UHCVKLHBPI30XQA4MTJJPSQ', '3MWOYZD5WVM5K47JV76W2GPQWFFNOV', '30EMX9PEVKHJAQE95BVPUTINZ9DKSK', '3B9XR6P1WETPIKRPC49I3P8AGWRBJB', '38XPGNCKHTYEOG6AEEDEYZDY1RHV4A', '3CVDZS288HY5W99QK38627RCZOEFM1', '3E22YV8GG14N54JKA7JBRGUG6KBNPC', '3W9XHF7WGKTKPQROUB0F4YE91SRKT9', '386T3MLZLNTZEGSWNKF2UI3D9H0802', '3PEG1BH7AEPKT4X7UP9U448NVLRBKA', '3R5OYNIC2C7L0HO4735N0F08V2XPTF', '3M0556243SIBERUQW4N6FMGC4EOFN1', '35A1YQPVFEERYTQATCX5O2TITLGI55', '3RKHNXPHGWUZP36DUGQP9U0BK6YKU0', '33TGB4G0LPFQ9QQVTUF4USMBX16XTB', '3MG8450X2O800U3T78VX5BMAEG4PU6', '3KG2UQJ0MJM85GKZAKY0RLKQBGXNQ3', '3XT3KXP24ZWNHXQOYV7AG87IPGRI6L', '324N5FAHSX9Z2IVDZ60U5G2UYD7KVP', '3MVY4USGB6LS4VOL58ADHD3A75OSIH', '3RSBJ6YZECOTWCU2XJNJUBGHUOTFOG', '3X0EMNLXEPN0D9STCY521XOTSNDPVV', '3GVPRXWRPHS6LX4TH5C2MM7EKRBI7K', '3MA5N0ATTC9CH7VIPG5QRIL85M9KWZ', '3Y40HMYLL1G5935KTZ5EZO8DGFDXU2', '37SDSEDIN90ZL5DEJ3J28NMNE96815', '3JYPJ2TAYI6BHC63CJ0YJBL74TPFPX', '3H6W48L9F4NDSYSY28AYNZ77ZWFPW5', '3YGYP1364178HHZZNZN0WBUV72ZNR4', '3IHWR4LC7DBALB9CRA480M39VBHI87', '39O6Z4JLX2V5MIUKYPFJVAAWUMMXVR', '30P8I9JKOIJWKZAASJWO3RJC1NCV5F', '3NOEP8XAU40SHYXOCD6KPILE8YLPXY', '3YCT0L9OMM7KIWZC2ON5MLZRT4INSS', '34OWYT6U3WFAZGSMW2AM0IYKFHZI9W', '3TC2K6WK9G06AQVA27JGD6WS6FK82M', '3PCPFX4U40OWHO7ICWFNJGBH9PBFQO', '356ZPKYPUHFHH3NYH9SM8WSNA4FPY4', '3SSN80MU8CMR6708ZL1CT1ZFEOFKXS', '3BO3NEOQM0FO4Z2RO8OMWD94LZHIAB', '3TUOHPJXYHVI17UPOZKFHCTA1VOXW1', '3D1UCPY6GG7VV8RT67XLWG1KCTK83X', '3S8A4GJRD31S33AOX26TVXXCXINV6V', '33CLA8O0MI9WTPMIPB4NO6LM5BDFRP', '3NRZ1LDP7W4HMTX3DVU5M31PGYJPZV', '3U74KRR67MJLXWCRROSVWQVDVNWNTH', '3AA88CN98P1G6CQI4HNECF6OGU9KYY', '3M93N4X8HKLOZ68RNQVDTEIAKKDSJT', '3ZC62PVYDH8XQ7ZFY4G1JV7HAXUXXU', '3BAWBGQGYLXB73OTGCBL1BX8ST7V7U', '338GLSUI4398U4MV404SEGQIRDWFSD', '3PIOQ99R7YK0X9RGRTI51MHFE13NU8', '3SV8KD29L4QGB20N03PXQMFQMODKZP', '3BVS8WK9Q0TTN97JUMOIROTA3H2IB6', '32TMVRKDGNWGD4AX6L246LTQQTG84I', '3HUR21WDDUNMQCPP302329EQC3OXY0', '3YO4AH2FPDI5KKSS9EW2G9OD3NMQ0V', '3Z3R5YC0P3L994ZAT09IOLM4TWAFT2', '32XN26MTXZHUTNCFH9O6ZNPJTHZL01', '3XAOZ9UYRZP5R9DA5X02UE7N8FSQ1Y', '3UEBBGULPFMO9HEZT5ZSTH86CAHFUT', '3UL5XDRDNCHY9NEXKQLEBDZ4QPB85N', '31J7RYECZLOU0CXXDSS6DS8TY95L14', '3JUDR1D0D6PCGUV6O10GZXHS0L6Q2F', '35YHTYFL1G1OMW3ZYV9XP3APQHQFVI', '30EMX9PEVKHJAQE95BVPUTINZ9DSKS', '3M47JKRKCXZUSREBP9VJ3JD4MKM863', '3N3WJQXELSO1PXFTWWSKIBIYQFJL2L', '3ZFRE2BDQ9CMV2ZUZM4MGGNSIXSXZR', '3IKDQS3DQEYCXEZP8MKD28ZSL22ICD', '3P4ZBJFX2V111L34O5ETB5T3XQSFWS', '3DW3BNF1GHGF7HTCQH3RFBT33DDV8H', '3MGHRFQY2LNETMJE0ODWTVRF2W3Y06', '3Z33IC0JC0KFLMCMV995F7OENJVV96']
	d = ['3PA41K45VN2Y2JRXU02MVJ8Q66A7PR', '3MGHRFQY2LNETMJE0ODWTVRF2X70YE', '3R15W654VDRIHGHEY7ILETG16M0QL9', '3N7PQ0KLI5NC491KJAJLOQX6NXM3EX', '341H3G5YF0CEYCTJWAFF720H8RB0Z5', '3ABAOCJ4R822NYZZYJJ0E6VQ6RQQMB', '3JMNNNO3B12H0RRS0AZMRMZEONQW2U', '3VO4XFFP15KJ2VSCUDHBVOY0B2W7QI', '3L21G7IH47UE0B4WW7HW70AP7PCY1E', '3J94SKDEKINAQIAO1YWUX5D51HC5DW', '3X4Q1O9UBHKGHJFWF0P76J30W1DO7Q', '39O0SQZVJN5JEW7C7S6B0E857OY7RJ', '3I4E7AFQ2KXPE6L6CQ033JEWM4UTJO', '386659BNTLFH3G8BOZ6NF8G5VCY10H', '3ZQX1VYFTD3KHXKFP5HDKJZV7LJO8D', '39KV3A5D185VFB7PMH6GQOD1TQH7S7', '37SQU136V7MHA5TTKIANTDZF04411K', '3W9XHF7WGKTKPQROUB0F4YE91TUTKN', '37OPIVELUU1O84R7IQ61W18PLN6AHG', '3KG2UQJ0MJM85GKZAKY0RLKQBH0QNB', '34R0BODSP1XFIOOOXT7E3NVQ1TD5EP', '33BFF6QPI196L9NB4ADRAW46U1QW35', '35F6NGNVM8HWUBK4BHB60T9NV9V7TW', '3MNJFORX8B27O2OKS4RKL5GQDUS5F7', '3TCFMTM8HEMOZQBP3MA1YW9KSAI121', '37MQ8Z1JQEULPWMSFBHACJKUZVQY2V', '3P888QFVX3SQN1KBBZYD6AKV1R5QOQ', '3I6NF2WGIGUD22KF3OX23QG2E9S5G6', '31S7M7DAGGOV9W96FBCGJYROQTZTLQ', '3D17ECOUOETDK8787MO6H6ECYOI13C', '3RBI0I35XE1AAEIBJBVFVTPM59QY36', '3BO3NEOQM0FO4Z2RO8OMWD94L0LAI9', '32TZXEA1OLIY24QC70TPRB6ICOE14X', '3HEM8MA6H9A881WCQZBSVAPLBW1QP7', '3B286OTISEFKZAMX6Q9M8EO4YFAAJL', '3S37Y8CWI8YR356F4OIAK1WC81MW4Q', '3G3AJKPCXLQVSA1FJP0Y5YHSJ9MY4R', '3NSM4HLQNRST8DXRQCQHVFFVGSNQQY', '3ULIZ0H1VA3GYNUCL5CZW3CWCK9152', '3P7RGTLO6EBFAUSFOB9Y9TOHD4AAKK', '3K2CEDRACBZFFERRFNDVJB6DQYPTMS', '38VTL6WC4ABDOT5FXUJ8AQN6J5HY5W', '3M4KL7H8KVLCHRUQQOM4O9QW8FK16I', '31ANT7FQN80R2YFQNUFKOQS3T955H5', '3538U0YQ1FS4AN1GEL3R68I6ZY13FF', '30EV7DWJTVT97X5T2DTD2W16F0SY6C', '3U74KRR67MJLXWCRROSVWQVDVOZTNS', '3566S7OX5DHVLR8V9YRWUNQS3Q417H', '3JGHED38EDPSBXJYLNY58A12ABCY7B', '3ZZAYRN1I6P3FHC3S3S8BFVILY4TO7', '31SIZS5W59DTKECR3RFH05P0CEPQRZ', '3L2OEKSTW98WBBOHVSQBMAXXLVIY8Y', '35A1YQPVFEERYTQATCX5O2TITMK5IY', '3UQ1LLR26A6QU0AX9BLZOT1W24FALN', '30QQTY5GMKIBUOZTBM1G5PVPEN27UN', '379OL9DBSSCWPG7R0KWPM6S8510Y9N', '31MCUE39BKKAONXBP599OTII0D13GE', '3R5OYNIC2C7L0HO4735N0F08V30TPO', '3KVQ0UJWPXJ9ZOAFIT1KPT2Q8XHW5V', '31ODACBENUD5LTC4IGFMQFUWYG8QSN', '37SDSEDIN90ZL5DEJ3J28NMNEAA184', '3BAWBGQGYLXB73OTGCBL1BX8SUB7VC', '3KQC8JMJGCQOOJSM9BRR9TUJFDE3HD', '35O6H0UNLSENT4AGBUI5038I6195JA', '3XJOUITW8UP60TPJ7GKC0KQI0ZMQTC', '3OQQD2WO8I4OKE36FT9C95VYFQT3I6', '3CESM1J3EI15ISATNCBPHZGQ4SSW6B', '3D06DR5225HA0ISI9NMEO6GL295AMP', '3SU800BH86QL06487LAM5GCKJDTQU3', '3O4VWC1GEW4KFPNCXBUCL6AYS5I3JI', '3VGET1QSZ0XOMSOY6MGHNDGMZ3D7WM', '33EEIIWHK75LDLT8CBKR12E3XK2QVS', '3VGET1QSZ0XOMSOY6MGHNDGMZ3CW7A', '3E9VAUV7BWCAUZHWSQAPI13SBJIYA2', '329E6HTMSW0FQ9TUFWUOMLAB7UI3KH', '3X2LT8FDHWGSM6THGR8N1DCHANIW8X', '3NKW03WTLM5YSATD2LPNN4XH4T4QW2', '3JTPR5MTZSAI4OGYTFIH1I8VLQ95K9', '3EGKVCRQFWQFD9MOY4ALDCNYT13YBX', '3T2EL38U0MIDHAY3CQL9PNBODVAQXV', '37SOB9Z0SSVQAFBC0W6P1LNQWUN3LK', '3OCZWXS7ZO5TOUYGEFUIGILAAQE5LC', '3TZDZ3Y0JS4ZZAWOOVPG8JHYYGS19T', '3N5YJ55YXG1GI0DILO1E1L5L7ZFANP', '37M4O367VJGDUCG1ERVXGV0ZAV45ME', '30ZKOOGW2W4D4T6TG13G4ESI4YA1A8', '3Q2T3FD0ON6AGXTX08741Y2FWZD3MM', '3XJOUITW8UP60TPJ7GKC0KQI0ZMTQF', '306996CF6WIIN3BLMF3CZPCOMGV1B3', '307L9TDWJYQGYFEXC9M4EDRF1PN3NM', '3HRWUH63QU0JCU11QSAXTAPZFLE5NE', '3BKZLF990ZX2HFODHM7B81IXF14QY1', '31YWE12TE0A3BSTOGRC3PWUT85J7XF', '35ZRNT9RUIWYG0E9DOMHT2RKRZS3O1', '3L55D8AUFAVYNEEUC46GOWTGBM3YC4', '3SX4X51T807Y0LDUM31RGA5QX9KAO4', '3BJKPTD2QCA6CU4JKV9C5A0NWLOTRG', '3AQN9REUTFE8S6K8C01R82BYWMFYD4', '3JGHED38EDPSBXJYLNY58A12ABD7YL', '3MJ28H2Y1E61UF1DR7AA8ZP45VJ5OT', '311HQEI8RSESGNT3H90OMHA4G5H7ZC', '3X55NP42EOEG10QASOZWI2WA14O3PI', '37VUR2VJ6AN1X83R0FZ7A9I641V1CA', '386T3MLZLNTZEGSWNKF2UI3D9I4808', '3BFF0DJK8XAID94WZK9HVK5JIN7TS4', '33J5JKFMK6W11CRPS1ELI7MK60A3Q9', '3V8JSVE8YYODKCY88VCBEKTJWYGYEX', '3EPG8DX9LKOJFFDE67NPXZUUF0F5PA', '3INZSNUD80OTR5C04O9B9LI7NDVD9V', '3K3IX1W4S6PGLLPV13E65AAG7EGAPL', '3HJ1EVZS2OH1DD6P5G3LNXWP2MC3RA', '3T5ZXGO9DEM2M5YID89UM8RZLV8QZS', '3D42WVSDH8T5QQY436WHW2EJ8ZVYFF', '3PN6H8C9R4O7WOM5WUNB5GTRTVDDAA', '3WGCNLZJKF6B20950BUIUF0OP171DA', '3HFWPF5AK9HDES62K53QD71LOOV3SY', '37SDSEDIN90ZL5DEJ3J28NMNEAA81B', '3K3G488TR264FRET6K2EX4K4KW15Q1', '3PUV2Q8SV42CFYRX28N70RDXBDYDB5', '3HY86PZXPYGGU6N5W6520XI9PD81E3', '3QHITW7OYO71LXQA1GTV5F0QCA2AQC', '3DA79LNS59TETSJH958GNCX7Q793TN', '39N6W9XWRDLB4QUZEQ2ZENEV9EVYGE', '3J9L0X0VDFKS0BCRLJE1197SUT0W9M', '3ZURAPD288L80KN1RHP8IF391EN1FL', '34HEO7RUG6S1XY5AEVIVA5AV8W4ARD', '38LRF35D5LUTT5Y69AYQS8J99LG3UE', '3Y3CZJSZ9KR4RSTTJZRE2UU9GI35R2', '3WJGKMRWVI7VP3J3G8J2BBJFTYYDCC', '3VDVA3ILIDDEEKJW21VQ003L2TN1GK', '34D9ZRXCYRSDYD5NTKI00FFRUYNAS1', '3TC2K6WK9G06AQVA27JGD6WS6GO82S', '3SR6AEG6W5RP4MPAYWKHZNQWOE8YHD', '3Q9SPIIRWJK65UMWDPS1X4IC0BIWA1', '3CO05SML7V35WL7SMTL2LZYIZARR02', '3J5XXLQDHM9T6KN6E08VOULSNSP3V3', '3YZ7A3YHR5RGS7T6YORJS4Z52KM5SQ', '3WRKFXQBOB5P0H0U4E22ZZRBORNYI6', '3QGHA0EA0JYBO4ROJ3SXSF2IIT3WBW', '3EHVO81VN5JSEGE7M7D8L0FMHT01HJ', '3D1UCPY6GG7VV8RT67XLWG1KCUO833', '3UUIU9GZC53H776LNOW929VR4305TF', '3X52SWXE0X3UY9JUX3OS3Z800E3WC3', '3VDVA3ILIDDEEKJW21VQ003L2TNG1Z', '3IH9TRB0FBXSABPRSPVTLCG1H6F1IC', '3W5PY7V3UP5LVSK0MWN2B06B16CYJI', '3MQKOF1EE2M431P8XZJ3N5QILEFWD3', '33BFF6QPI196L9NB4ADRAW46U1R3WD', '3P520RYKCH4W7KLANTMJ75HTNH75U6', '3AA88CN98P1G6CQI4HNECF6OGVCYKH', '3HXK2V1N4KDL351SL5V45JDQUZ1G2G', '308KJXFUJR4EDDI2IKNQAKBDWH1ATQ', '32TMVRKDGNWGD4AX6L246LTQQUK84O', '30P8I9JKOIJWKZAASJWO3RJC1OG5VV', '31MCUE39BKKAONXBP599OTII0D1G3R', '39TX062QX1MLA9S1EF9DCFID33X3X6', '3IVEC1GSLPXO5M9XA7GTXDV1UL41JO', '3UL5XDRDNCHY9NEXKQLEBDZ4QQF85T', '3QE4DGPGBR9V6JGFPJESYYAOEDXG4C', '3VJ4PFXFJ35TDQXRIPD0FGXFFV8AUH', '3RBI0I35XE1AAEIBJBVFVTPM59R3YC', '3M47JKRKCXZUSREBP9VJ3JD4MLQ869', '363A7XIFV4KTQ5MRNFN5B2ZYT2HAV6', '39WICJI5ATQAF4SGFXXY90YOB3V3Z3', '356TQKY9XFVDWRSG8J0B9XD0HWA878', '3Q9SPIIRWJK65UMWDPS1X4IC0BJAWG', '378G7J1SJLW9V738TUUNTN83LQGWEW', '3FTID4TN8LWRQI80PHZFRFJ35VHYLK', '3P4ZBJFX2V111L34O5ETB5T3XRVWFE', '3WRAAIUSBJXLUURMNUONZNWJ9DPAX9', '3L4YG5VW9NQ5UVPHG4EDVH1XEYADDC', '337F8MIIMZBYVF5UBHPPD6N87W840X', '3Y3N5A7N4G7BW0QLPT0URSYS507YMM', '3E9VAUV7BWCAUZHWSQAPI13SBJJAYF', '32TZXEA1OLIY24QC70TPRB6ICOE410', '3LN3BXKGC0T7FLZZZPKBTQTFY6VWGD', '36MUZ9VAE60AM13HCZPX1ZJIEABDE5', '388FBO7JZRRHEIBL1UFU47NSAQHYNM', '37S0QRNUFBEHW5XZIOSHNX9VSGG88V', '3W0XM68YZPTJG6FFSSG5YSVE9A41KN', '34R3P23QHSZLFHUAJV2TEQ5GD68WHC', '3WUVMVA7OB1AZPR1OCC8W8CUHDNAZ6', '3ODOP6T3ASI5RP88Q4T3WUGN4US42H', '3TZ0XG8CBUIHAAG9NGYVNT46CMY89K', '38RHULDV9YDLBC5UPDKEE26VDJNWI5', '3OID399FXG52SF3D7A93JH4IQBQDFN', '31J7RYECZLOU0CXXDSS6DS8TYA91LQ', '31KPKEKW4ABIGEQ3QWZCLKTEHOPB0A', '3KTCJ4SCVGZE6UFID4TLD5NIYFZ1MS']
	a = a+b
	# a = c+d
	print(len(a))
	a = ','.join(a)
	print(a)

def qa_summary():

	qa_dict = {}
	df = pd.read_csv('../annotation_use/amazon_qa_filtered_5_150_8q.csv',encoding='utf-8')
	for index, row in df.iterrows():
		# print(index, row['Product'],row['Product Id'],row['Question'],row['Answer'])
		if row['Product Id'] not in qa_dict:
			qa_dict[row['Product Id']] = {'question':[row['Question']],'answer':[row['Answer']]}
		else:
			qa_dict[row['Product Id']]['question'].append(row['Question'])
			qa_dict[row['Product Id']]['answer'].append(row['Answer'])

		# if index > 10:
		# 	pprint(qa_dict)
		# 	exit()
	print(len(qa_dict))
	json.dump(qa_dict, open("8qa_ref.json", 'w',encoding='utf-8'),indent=2)
	exit()

	with open('amazon_qa_summary.json','r',encoding='utf-8') as f:
		data = json.load(f)
		print(len(data))
		for each_item in data:
			pprint(each_item)
			exit()
	

def qa_correction():

	with open('8qa_ref.json','r',encoding='utf-8') as ref:
		ref_data = json.load(ref)
		print(len(ref_data))

	with open('amazon_qa_summary.json','r',encoding='utf-8') as qa:
		qa_data = json.load(qa)
		print(len(qa_data))

	check_list=[]
	cnt=0
	for each in qa_data:
		qa_info = ref_data[each['asin']]
		# print(qa_info)
		# print()
		# print(each)
		cnt+=1
		for qa_pair in each['qa_pair']:
			if qa_pair['question'] not in qa_info['question']:
				# print('Q correct !')
				print(qa_pair['qid'])
				print('Q:',qa_pair['question'])
				check_list.append(qa_pair['qid'])
			if qa_pair['answer'] not in qa_info['answer']:
				# print('A correct !')
				print(qa_pair['qid'])
				print('A:',qa_pair['answer'])
				check_list.append(qa_pair['qid'])
		if cnt>100:
			exit()
	print(len(set(check_list)))
	exit()



if __name__ == "__main__":

	qa_correction()
	# qa_summary()
	# eval_4_standard()
	# get_mturk_annotation()
	# eval_lcs()
	# parse_filtered_data()
	# sample_100_item_qa_pair()
	# parse_annotated_data()
	# sample_annotated_data()
	# cluster_represent()
	# qa_cluster_diff_sample()
	# parse_cluster_sample()
	# qa_statistic()
	# parse_sample_info()
	# check()
	# agreement()
	# hit_id()

		
