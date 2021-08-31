# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division, print_function, unicode_literals

from sumy.parsers.html import HtmlParser
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer

# from sumy.summarizers.luhn import LuhnSummarizer as Summarizer
# from sumy.summarizers.edmundson import EdmundsonSummarizer as Summarizer
# from sumy.summarizers.lsa import LsaSummarizer as Summarizer
# from sumy.summarizers.text_rank import TextRankSummarizer as Summarizer
# from sumy.summarizers.lex_rank import LexRankSummarizer as Summarizer
# from sumy.summarizers.sum_basic import SumBasicSummarizer as Summarizer
from sumy.summarizers.kl import KLSummarizer as Summarizer

# opt_summarizer = ['LuhnSummarizer','LsaSummarizer','TextRankSummarizer',
#                   'LexRankSummarizer','SumBasicSummarizer','KLSummarizer']

from sumy.nlp.stemmers import Stemmer
from sumy.utils import get_stop_words
from nltk.tokenize import word_tokenize

import json, csv 
import os, time
import pandas as pd
from tqdm import tqdm
from glob import glob
from sumeval.metrics.rouge import RougeCalculator
from pprint import pprint


LANGUAGE = "english"
SENTENCES_COUNT_list = ['25%','50%','75%','100%']
SENTENCES_COUNT = 10

sample_text = "The tiles are white. It's definitely far from ivory. It's a great set. The pushers are attached to the racks and fit in the box. The case is spacious enough for storing the set after using. There are numbers on the character suit tiles: bams, cracks, dots, flowers. It's a wonderful set. The character tiles are engraved. The total weight of the mahjong set including tiles, case and pushers is 9.5 pounds. The character tiles are the standard white ones of standard size. The set comes with a pamphlet type book."

invalid_worker = ['A2ZLJQWCM8KU36', 'A1916MMCDQKV6P', 'A2G94HZV0UJEKU', 'A1OKE44UD16QJV', 'A2EYQ1G6U562R', 'A1YQVI76FI8GZ1', 'A19Z68L0RHQVWL', 'AVKRMPJXUCHUM', 'A11V7HCHHQIJWQ', 'A189OOQZIULCDV', 'A3VBUVGBQ74JJF', 'A2J9RHWDZ9HHUA', 'A2KNKD15KUMW7Q', 'AFXXRR3UKQPC6', 'A76WEZ9QGREEM', 'A8Q3CXMIF3HJ9', 'A13DKPCP0O48WN', 'AR8M7X4W9PJ4F', 'AYKUOC2P53YUY', 'AHKUEETLLGIO8', 'A1KG4QAQMXVIPG', 'A2C9Z4YND5DX0E', 'A10DZH6L8TBYRC', 'A1OKFVGA9A0VD3', 'A3QHX7BMC3RJYQ', 'A2QBAID9PB6A58', 'A2SUSFPE5ETNPY', 'A1YVERGDC9B34B', 'A2XT04FU0SVVMM', 'A1ZLR6WXLCXRRO', 'A3HCWJBU0TWIY3', 'A1PMIFL7VI16LI', 'A2MYLV7DGUL42B', 'A35X5ABMK2SUP3', 'A1CJ2RK525JSMZ', 'A3GIHDWEFSPZK0']
# , 'A1GMY3KWTABONG', 'ANXIYA9KKTO2Y', 'A2MOKIEQZ0OF2M', 'ARQR5NIFA1AJ', 'A34QZDSTKZ3JO9', 'AT8A9XACJK1UR', 'ASQ460H8M8OOT', 'AVFZLWRV14IYN', 'A1MC3CCKEU9UTR', 'A17Q4QN6UE0EZC', 'A2J1A9GE02VQHR', 'A1VWJ6LH1E3GLN', 'A1LDO8EYGXOA9D', 'A1Z0DMQTDUU95E', 'A3HRNH1WGC4UI3', 'A1KAIOBTXQ77A6', 'A2CS67AX8RPJUB', 'A2KHMKJUVDMK1N', 'A1SOFLJOEQB591', 'A34L4OHPHNU8ZK', 'A3O5RKGH6VB19C', 'A3CP3V6ROWGYDF', 'A1198W1SPF1R4', 'AM65LGXJBTJ0I', 'A2W5MSZP0O4Y2P', 'AI6TD8PM938FQ', 'A3C8NUIBNZYMT2', 'A340FMNO9WJG4J', 'AKR9067Q09RZS', 'A25AYMSZNDW1VJ', 'A2BC9ZEA35UDMA', 'A2WX434EAQOE29', 'AZ4IGXEZRJSTP', 'AOMFEAWQHU3D8', 'A3FDSQADGUY42E', 'A2IVLRO164XH51', 'A2902FWLHHPSDX', 'A1BZ1VD8V8VJML', 'A23Y70XTKVLFO7', 'A2VNR6984SDFGQ', 'AAC9DJ81ZXUE7', 'A2QPAGBQWGDMX2', 'A2RWESHZZ2VYN7', 'A3VPD34C23PQTQ', 'AZZA3J049G7R5', 'A13JDB1LY9ZTQY', 'A1T9BRIZU3B0IY', 'A2PD9SHVWNX7Q2', 'A27AK750Y9M9KH', 'A12FTSX85NQ8N9', 'A2LDDDJ4WBKB8Y', 'A3CVY8G619MGTI', 'AV4584AQEQAEN', 'A3LPHYONE222OY', 'A34H7UVYQTC390', 'A249LDVPG27XCE', 'A31BA1WLSWUSY1', 'A7GIIA9EI1Y3T', 'A3VEF4M5FIN7KH', 'AEJKD2E3MQKW5', 'A2UYXRPN41PLQE', 'A1LK72MZIUGXT8', 'A1LTAYZAMU1A4C', 'A381DV1DQVPBHN', 'A1KLTPBLZUYKFO', 'A2JXPT39AWRES7', 'A36QGCT3MMXC4Q', 'A3LW9WQOQQMA5V', 'AU2C0Q45DVGJO', 'A193D9OQUVFSVT', 'A2DDPSXH2X96RF', 'A263Y9ZPYSSTB9', 'A2NZAL7KHOR6VF']

worker_eval_file = 'worker_error_eval.json'
item_summary_file = 'item_summary.json'

def QA_simple_eval():


    with open('qa-summarization/amz_qa_sum_test_ori.json', 'r', encoding='utf-8') as f:
        qa_summary = json.load(f)
        print(len(qa_summary))

    rouge = RougeCalculator(stopwords=True, lang="en")
    rouge_1_list, rouge_2_list, rouge_l_list = [], [], []
    cnt = 0

    for each_item in tqdm(qa_summary):
        cnt += 1
        gold_sum = ''
        sample_text = ''
        for each_qa in each_item['qa_pair']:
            # extract gold summary
            if 'annotation' in each_qa:
                for each_edit in each_qa['annotation']['rewrite']:
                    if each_edit['is_selected']=='True':
                        gold_sum += each_edit['edit'] if each_edit['edit'][-1]==' ' else each_edit['edit']+' '
            sample_text += (each_qa['question']+' '+each_qa['answer'])

        # lexrank
        parser = PlaintextParser.from_string(sample_text, Tokenizer(LANGUAGE))
        stemmer = Stemmer(LANGUAGE)

        summarizer = Summarizer(stemmer)
        summarizer.stop_words = get_stop_words(LANGUAGE)

        # count sentences

        sentences = ''
        for sentence in summarizer(parser.document, SENTENCES_COUNT):
            # sentences += str(sentence)
            if (len(word_tokenize(sentences))+len(word_tokenize(str(sentence)))) < 156:
                sentences += str(sentence) if str(sentence)[-1]==' ' else str(sentence)+' '
            else:
                break
        
        if cnt == 1:
            print(len(word_tokenize(sentences)),sentences)
            print(len(word_tokenize(gold_sum)),gold_sum)
        # exit()

        rouge_1 = rouge.rouge_n(summary=sentences,references=gold_sum,n=1)*100
        rouge_2 = rouge.rouge_n(summary=sentences,references=gold_sum,n=2)*100
        rouge_l = rouge.rouge_l(summary=sentences,references=gold_sum)*100
        rouge_1_list.append(rouge_1)
        rouge_2_list.append(rouge_2)
        rouge_l_list.append(rouge_l)
        # exit()


    # print rouge socres
    print("ROUGE-1: {}, ROUGE-2: {}, ROUGE-L: {}".format(
        round(sum(rouge_1_list)/len(rouge_1_list),4), round(sum(rouge_2_list)/len(rouge_2_list),4), 
        round(sum(rouge_l_list)/len(rouge_l_list),4)).replace(", ", "\n"))
    print()
    exit()

def multiQA_eval():

    rouge = RougeCalculator(stopwords=True, lang="en")

    with open(item_summary_file, 'r') as f:
        item_sum = json.load(f)
    f.close()
    # pprint(item_sum)

    for SENTENCES_COUNT in SENTENCES_COUNT_list:

        # good worker 
        csv_file = open('intern_project/'+SENTENCES_COUNT+'_naive_baseline_eval_all(multi-QA).csv', 'w')
        writer = csv.writer(csv_file)
        writer.writerow(['Prodct Id','Question', 'Answer','Q-A pairs','Q-A summary','Gold summary'])

        sample_text, gold_sum = '', ''
        data_queue, item_list = [('default', 'default', 'default')], []
        rouge_1_list, rouge_2_list, rouge_l_list = [], [], []
        cnt = 0

        path='amazon_qa_filtered_5_150/'

        for filename in glob(os.path.join(path, '*.csv')):
            print(filename)
            df = pd.read_csv(filename,encoding='utf-8')
            # print(df.head())

            for index, row in df.iterrows():
            
                # check if in valid item
                if row['Prodct Id'] in item_sum:
                
                    if row['Prodct Id'] in set(item_list):
                        data_queue.append([row['Prodct Id'],row['Question'],row['Answer']])
                        
                    else:
                        if item_list:

                            for each_row in data_queue:
                                sample_text += (each_row[1]+' '+each_row[2])

                            # lexrank
                            parser = PlaintextParser.from_string(sample_text, Tokenizer(LANGUAGE))
                            stemmer = Stemmer(LANGUAGE)

                            summarizer = Summarizer(stemmer)
                            summarizer.stop_words = get_stop_words(LANGUAGE)

                            sentences = ''
                            for sentence in summarizer(parser.document, SENTENCES_COUNT):
                                sentences += str(sentence)
                            # print(sentences)

                            # print(data_queue)
                            gold_sum = item_sum[data_queue[0][0]]
                            # print(gold_sum)
                            
                            # output 
                            # for i in range(len(data_queue)):
                            #     if i == 0:
                            #         writer.writerow([data_queue[i][0],data_queue[i][1],data_queue[i][2],
                            #                         sample_text, sentences ,gold_sum])
                            #     else:
                            #         writer.writerow([data_queue[i][0],data_queue[i][1],data_queue[i][2]])
                            
                            # calculate rouge scores
                            rouge_1 = rouge.rouge_n(summary=sentences,references=gold_sum,n=1)
                            rouge_2 = rouge.rouge_n(summary=sentences,references=gold_sum,n=2)
                            rouge_l = rouge.rouge_l(summary=sentences,references=gold_sum)
                            rouge_1_list.append(rouge_1)
                            rouge_2_list.append(rouge_2)
                            rouge_l_list.append(rouge_l)
                            
                        # next item
                        data_queue = []
                        sample_text = ''
                        data_queue.append([row['Prodct Id'],row['Question'],row['Answer']])
                        item_list.append(row['Prodct Id'])
                        
        # print rouge socres
        print("ROUGE-1: {}, ROUGE-2: {}, ROUGE-L: {}".format(
            round(sum(rouge_1_list)/len(rouge_1_list),3), round(sum(rouge_2_list)/len(rouge_2_list),3), 
            round(sum(rouge_l_list)/len(rouge_l_list),3)).replace(", ", "\n"))
        print()

def lex_rank():

    rouge = RougeCalculator(stopwords=True, lang="en")

    with open('intern_project/'+worker_eval_file, 'r') as f:
        worker_eval = json.load(f)
    # pprint(worker_eval)

    df = pd.read_csv('intern_project/mturk_results/post-edit_v3_batch_all/mturk_30_items_results_filtered_post-edit_v3(stat).csv',encoding='utf-8')
    # print(df.head())
    item_sum = {}

    for SENTENCES_COUNT in SENTENCES_COUNT_list:

        # good worker 
        csv_file = open(SENTENCES_COUNT+'_naive_baseline_eval.csv', 'w')
        writer = csv.writer(csv_file)
        writer.writerow(['Prodct Id','Worker Id','Question', 'Answer','Q-A pairs','Q-A summary','Gold summary'])

        sample_text, gold_sum = '', ''
        item_list = ['B00008RW9U']
        rouge_1_list, rouge_2_list, rouge_l_list = [], [], []
        cnt = 0
        for index, row in df.iterrows():
            if (row['Prodct Id'] in worker_eval) and (row['Worker Id'] == worker_eval[row['Prodct Id']]):

                sample_text += (row['Question']+' '+row['Answer'])
                gold_sum += ' '+row['Q-A post-edit']

                # write each row
                if (cnt+1)%8==0:

                    # lexrank
                    parser = PlaintextParser.from_string(sample_text, Tokenizer(LANGUAGE))
                    stemmer = Stemmer(LANGUAGE)

                    summarizer = Summarizer(stemmer)
                    summarizer.stop_words = get_stop_words(LANGUAGE)

                    sentences = ''
                    for sentence in summarizer(parser.document, SENTENCES_COUNT):
                        sentences += str(sentence)
                    # print(sentences)

                    writer.writerow([row['Prodct Id'],row['Worker Id'],row['Question'],row['Answer'],
                                     sample_text, sentences ,gold_sum])
                    item_sum[row['Prodct Id']] = gold_sum

                    # calculate rouge scores
                    rouge_1 = rouge.rouge_n(summary=sentences,references=gold_sum,n=1)
                    rouge_2 = rouge.rouge_n(summary=sentences,references=gold_sum,n=2)
                    rouge_l = rouge.rouge_l(summary=sentences,references=gold_sum)
                    rouge_1_list.append(rouge_1)
                    rouge_2_list.append(rouge_2)
                    rouge_l_list.append(rouge_l)

                    # next item
                    sample_text, gold_sum = '', ''
                    item_list.append(row['Prodct Id'])

                else:
                    writer.writerow([row['Prodct Id'],row['Worker Id'],row['Question'],row['Answer']])

                cnt+=1
                    
        json.dump(item_sum, open("item_summary.json", 'w', encoding='utf-8')) 
        exit()      

        # print rouge socres
        print("ROUGE-1: {}, ROUGE-2: {}, ROUGE-L: {}".format(
            round(sum(rouge_1_list)/len(rouge_1_list),3), round(sum(rouge_2_list)/len(rouge_2_list),3), 
            round(sum(rouge_l_list)/len(rouge_l_list),3)).replace(", ", "\n"))
        print()

if __name__ == "__main__":

    QA_simple_eval()
    exit()
    # lex_rank()
    # multiQA_eval()
    # exit()

    SENTENCES_COUNT = 3


    # webpage
    # url = "https://en.wikipedia.org/wiki/Automatic_summarization"
    # parser = HtmlParser.from_url(url, Tokenizer(LANGUAGE))

    # plain text files
    # parser = PlaintextParser.from_file("document.txt", Tokenizer(LANGUAGE))
    parser = PlaintextParser.from_string(sample_text, Tokenizer(LANGUAGE))
    stemmer = Stemmer(LANGUAGE)

    summarizer = Summarizer(stemmer)
    summarizer.stop_words = get_stop_words(LANGUAGE)

    for sentence in summarizer(parser.document, SENTENCES_COUNT):
        print(sentence)
