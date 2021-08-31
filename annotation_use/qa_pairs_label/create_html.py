# import requests
# import ujson as json
import json, csv
from pprint import pprint
# from interface import tokenize_pos
# from data import parsing, find_target_word_with_pos, preprocessing

if __name__ == "__main__":

	with open('../intern_project/sample_100_item_qa_pair.csv', newline='', encoding='utf-8') as f:
		reader = csv.reader(f)
		head = next(reader)
		print(head)
		cat, prod_id, ques, ans =[], [], [],[]
		cnt, file_count = 0, 0
		for row in reader:
			if cnt < 24:
				cat.append(row[0])
				prod_id.append(row[1])
				ques.append(row[2])
				ans.append(row[3])
				cnt += 1
			else:
				print('=============')
				print(cat)
				print(prod_id)
				print(ques)
				print(ans)
				cnt = 0
				file_count +=1
				s = open("annotation_template.html").read()
				s = s.replace('<script src="main.js"></script>', '<script src="main_{}.js"></script>'.format(str(file_count)))
				file_name = "annotation_set"+str(file_count)+".html"
				f = open(file_name,'w')
				f.write(s)

				s = open("main_template.js").read()
				s = s.replace('<replace_item_cat>', 'var prod_cat_list ='+str(cat))
				s = s.replace('<replace_item_prod>', 'var prod_idx_list ='+str(prod_id))
				s = s.replace('<replace_item_ques>', 'var ques_list ='+str(ques))
				s = s.replace('<replace_item_ans>', 'var ans_list ='+str(ans))
				file_name = "main_"+str(file_count)+".js"
				f = open(file_name,'w')
				f.write(s)
				cat, prod_id, ques, ans =[], [], [],[]
				if file_count ==3:
					exit()

		# print(cat,len(cat))


	
	# file_count = 0
	# with open("sentences_processed.json", 'r', encoding='utf-8') as outfile:	
	# 	data=json.load(outfile)	
	# 	for word, (key,value) in enumerate(data.items(), 0):
	# 		word = (word%2)+1

	# 		if word ==1:
	# 			file_count +=1
	# 			s = open("html/student_template.html").read()
	# 			file_name = "test_file/student_pair"+str(file_count)+".html"
	# 			f = open(file_name,'w')

	# 		for drag, item in enumerate(value, 1):
	# 			ori = item["sentence"].replace(
	# 				item["tokens"][item["target_index"]][0],
	# 				"<strong>{}</strong>".format(item["tokens"][item["target_index"]][0])
	# 			)
	# 			new = item["sentence"].replace(
	# 				item["tokens"][item["target_index"]][0],
	# 				'<span word="{}">{}</span>'.format(key,"____")
	# 			)
				
	# 			s = s.replace('{{word'+str(word)+'}}',key)
	# 			s = s.replace('{{w'+str(word)+'s'+str(drag)+'}}',new)

	# 		if word==2:
	# 			f.write(s)
	# 			