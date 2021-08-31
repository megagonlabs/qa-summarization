# import requests
# import ujson as json
import json, csv
import pandas as pd
from pprint import pprint

# path = '/var/www/html/'
path = 'post_edit_'

if __name__ == "__main__":

    with open('amazon_qa_filtered_5_150_8q.csv', newline='', encoding='utf-8') as f:
        df = pd.read_csv(f)
        cat, prod_id, ques, ans =[], [], [],[]
        cnt, file_count = 0, 0
        for index, row in df.iterrows():
            if not index or (index % 24):
                cat.append(row['Product'])
                prod_id.append(row['Product Id'])
                ques.append(row['Question'])
                ans.append(row['Answer'])
            else:
                print('=============')
                print(prod_id)
                file_count +=1
                s = open(path+"annotation_template.html").read()
                s = s.replace('<script src="main.js"></script>', '<script src="main_{}.js"></script>'.format(str(file_count)))
                file_name = path+"annotation_set"+str(file_count)+".html"
                f = open(file_name,'w')
                f.write(s)
                
                s = open(path+"main_template.js").read()
                s = s.replace('<replace_item_cat>', 'var prod_cat_list ='+str(cat))
                s = s.replace('<replace_item_prod>', 'var prod_idx_list ='+str(prod_id))
                s = s.replace('<replace_item_ques>', 'var ques_list ='+str(ques))
                s = s.replace('<replace_item_ans>', 'var ans_list ='+str(ans))
                file_name = path+"main_"+str(file_count)+".js"
                f = open(file_name,'w')
                f.write(s)
                
                cat, prod_id, ques, ans = [row['Product']], [row['Product Id']], [row['Question']],[row['Answer']]
                if file_count ==1:
                    exit()

