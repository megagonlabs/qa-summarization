# Get Tuple algorithms 
import re, os
import math
import json, csv
import numpy as np
from itertools import chain
from collections import Counter
import nltk
from nltk.util import ngrams # This is the ngram magic.
from textblob import TextBlob

#NGRAM = 4

re_sent_ends_naive = re.compile(r'[.\n]')
re_stripper_alpha = re.compile('[^a-zA-Z]+')
re_stripper_naive = re.compile('[^a-zA-Z\.\n]')

splitter_naive = lambda x: re_sent_ends_naive.split(re_stripper_naive.sub(' ', x))

sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')

def get_tuples_nosentences(txt):
    """Get tuples that ignores all punctuation (including sentences)."""
    if not txt: return None
    ng = ngrams(re_stripper_alpha.sub(' ', txt).split(), NGRAM)
    return list(ng)

def get_tuples_manual_sentences(txt):
    """Naive get tuples that uses periods or newlines to denote sentences."""
    if not txt: return None
    sentences = (x.split() for x in splitter_naive(txt) if x)
    ng = (ngrams(x, NGRAM) for x in sentences if len(x) >= NGRAM)
    return list(chain(*ng))

def get_tuples_nltk_punkt_sentences(txt):
    """Get tuples that doesn't use textblob."""
    if not txt: return None
    sentences = (re_stripper_alpha.split(x) for x in sent_detector.tokenize(txt) if x)
    # Need to filter X because of empty 'words' from punctuation split
    ng = (ngrams(filter(None, x), NGRAM) for x in sentences if len(x) >= NGRAM)
    return list(chain(*ng))

def get_tuples_textblob_sentences(txt):
    """New get_tuples that does use textblob."""
    if not txt: return None
    tb = TextBlob(txt)
    ng = (ngrams(x.words, NGRAM) for x in tb.sentences if len(x.words) > NGRAM)
    return [item for sublist in ng for item in sublist]

def jaccard_distance(a, b):
    """Calculate the jaccard distance between sets A and B"""
    a = set(a)
    b = set(b)
    return 1 - len(a&b)/len(a)
    #return 1 - len(a&b)/len(a|b)
    #return 1.0 * len(a&b)/len(a|b)

def cosine_similarity_ngrams(a, b):
    vec1 = Counter(a)
    vec2 = Counter(b)
    
    intersection = set(vec1.keys()) & set(vec2.keys())
    numerator = sum([vec1[x] * vec2[x] for x in intersection])

    sum1 = sum([vec1[x]**2 for x in vec1.keys()])
    sum2 = sum([vec2[x]**2 for x in vec2.keys()])
    denominator = math.sqrt(sum1) * math.sqrt(sum2)

    if not denominator:
        return 0.0
    return float(numerator) / denominator


def test(summary, qa_pairs):
    #paragraph = """It was the best of times, it was the worst of times.
    #           It was the age of wisdom? It was the age of foolishness!
    #           I first met Dr. Frankenstein in Munich; his monster was, presumably, at home."""
    #print(paragraph)
    #_ = get_tuples_nosentences(paragraph);print("Number of N-grams (no sentences):", len(_));_

    #_ = get_tuples_manual_sentences(paragraph);print("Number of N-grams (naive sentences):", len(_));_

    #_ = get_tuples_nltk_punkt_sentences(paragraph);print("Number of N-grams (nltk sentences):", len(_));_

    #_ = get_tuples_textblob_sentences(paragraph);print("Number of N-grams (TextBlob sentences):", len(_));_

    #a = get_tuples_nltk_punkt_sentences(a)
    #b = get_tuples_nltk_punkt_sentences(b)
    a = get_tuples_textblob_sentences(summary)
    b = get_tuples_textblob_sentences(qa_pairs)
    #print("Jaccard: {}   Cosine: {}".format(jaccard_distance(a,b), cosine_similarity_ngrams(a,b)))

    #a = get_tuples_nosentences("Above is a bad example of four-gram similarity.")
    #b = get_tuples_nosentences("This is a better example of four-gram similarity.")
    #print("Jaccard: {}   Cosine: {}".format(jaccard_distance(a,b), cosine_similarity_ngrams(a,b)))

    #a = get_tuples_nosentences("Jaccard Index ignores repetition repetition repetition repetition repetition.")
    #b = get_tuples_nosentences("Cosine similarity weighs repetition repetition repetition repetition repetition.")
    #print("Jaccard: {}   Cosine: {}".format(jaccard_distance(a,b), cosine_similarity_ngrams(a,b)))
    return jaccard_distance(a,b)#, cosine_similarity_ngrams(b)

def analysis(tokens_list, total):
    tokens_list = sorted(tokens_list)
    print(tokens_list[:50],len(tokens_list))
    mean = sum(tokens_list)/len(tokens_list)
    _max = max(tokens_list)
    _min = min(tokens_list)
    a = tokens_list[int(len(tokens_list)*0.8)]
    b = tokens_list[int(len(tokens_list)*0.9)]
    c = tokens_list[int(len(tokens_list)*0.95)]
    print(mean, _min, _max, a, b, c)
    print("std of arr : ", np.std(tokens_list))

def get_qa_data():
    
    output = open(str(NGRAM)+'-gram.csv', 'w')
    writer = csv.writer(output)
    writer.writerow(['category','entity','score'])
    
    path = '/data01/tingyao/qa-summarization/amazon_qa_dataset/cat_data'    
    cat_file = os.listdir('/data01/tingyao/qa-summarization/amazon_qa_dataset/cat_data')
    cnt, total_scores = 0, 0 
    for cat in cat_file:
        with open(os.path.join(path,cat)) as f:
        #with open('amazon_qa_dataset/amazon_qa_summary_filtered.json') as f:
            data = json.loads(f.read())
            scores = 0
            summary, qa_pairs, qa_edit_pairs, ques, ans = [], [], [], [], []
            qa_w_cnt, qa_r_w_cnt, sum_w_cnt = [], [], []
            for product in data:
                # input QA pairs
                summary, qa_pairs = [], []
                for each in product["summary"]:
                    summary.append(each) 
                for qa_pair in product["qa_pair"]:
                    qa_pairs.append(qa_pair["question"]+" "+qa_pair["answer"])
                    ques.append(qa_pair["question"])
                    ans.append(qa_pair["answer"])
                    if 'annotation' in qa_pair:
                        for edit in qa_pair['annotation']['rewrite']:
                            if edit['is_selected']=='True':
                                qa_edit_pairs.append(edit['edit'])
                # calculate scores
                scores += test(summary[0], ' '.join(qa_pairs))
            total_scores += scores
            print(cat, len(data), round(scores/len(data)*100,2))
            writer.writerow([cat,len(data),round(scores/len(data)*100,2)])
            cnt += len(data)
    writer.writerow(['all',cnt,round(total_scores*100/cnt,2)])

for NGRAM in [1,2,3,4]:
    print(NGRAM)
    get_qa_data()
