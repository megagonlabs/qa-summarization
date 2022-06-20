import os
import json, rouge
import collections, random
import torch
from pprint import pprint
from nltk.tokenize import sent_tokenize, word_tokenize
from datasets import load_dataset, load_metric
from transformers import LEDTokenizer, LEDForConditionalGeneration
from statistics import median, mean

# load qa dataset
#qa_summary_test = load_dataset("qa_summary.py", ignore_verifications=True, split="test")
#qa_summary_train = load_dataset("qa_summary.py", ignore_verifications=True, split="train")

# load tokenizer & model from checkpoint
#tokenizer = LEDTokenizer.from_pretrained("{name}/led-large-16384-pubmed")
#model = LEDForConditionalGeneration.from_pretrained("{name}/led-large-16384-pubmed").to("cuda").half()
#tokenizer = LEDTokenizer.from_pretrained("allenai/led-base-16384")
#print(tokenizer.tokenize("I have a new GPU!"))
random.seed(123)

def generate_answer(batch):
  inputs_dict = tokenizer(batch["qa_pairs_text"], padding="max_length", max_length=3096, return_tensors="pt", truncation=True)
  input_ids = inputs_dict.input_ids.to("cuda")
  attention_mask = inputs_dict.attention_mask.to("cuda")
  global_attention_mask = torch.zeros_like(attention_mask)
  # put global attention on <s> token
  global_attention_mask[:, 0] = 1

  predicted_abstract_ids = model.generate(input_ids, attention_mask=attention_mask, global_attention_mask=global_attention_mask)
  batch["predicted_summary"] = tokenizer.batch_decode(predicted_abstract_ids, skip_special_tokens=True)
  return batch


#result = qa_summary_test.map(generate_answer, batched=True, batch_size=4)

# load rouge
#metrics = load_metric("rouge","bertscore")
#metrics = load_metric("bertscore")

#print("Result:", rouge.compute(predictions=result["predicted_summary"], references=result["summary"], rouge_types=["rouge2"])["rouge2"].mid)

#print(qa_summary_train)
#print(qa_summary_test)

def huggingface_rouge():

    for batch in qa_summary_test:
        inputs = batch['qa_pairs_text'] 
        references = batch['summary']
        #print(inputs)
        #print(references)
        metrics.add(prediction=inputs, reference=references)
    
    score = metrics.compute(lang='en')
    
    f1 = sum(score['f1'])/len(score['f1']) 
    recall = sum(score['recall'])/len(score['recall']) 
    precision = sum(score['precision'])/len(score['precision']) 

    print(f1)
    print(recall)
    print(precision)

vocab_counter = collections.Counter()

def analysis(sentence_list):

    token_list = []
    for sent in sentence_list:
        #token_list.append(len(word_tokenize(sent)))
        tokens = word_tokenize(sent)
        vocab_counter.update(tokens)

    #token_list = sorted(token_list, key=lambda x:x)
    #mean = sum(token_list)/len(token_list)
    #_max = max(token_list)
    #_min = min(token_list)
    #a = token_list[int(len(token_list)*0.8)]
    #b = token_list[int(len(token_list)*0.9)]
    #c = token_list[int(len(token_list)*0.99)]
    #print(mean, _min, _max, a, b, c)
    #print(token_list[:3], len(token_list))

def split_train_data():
    
    #with open('amazon_qa_dataset/qa_summary_filtered_train.json') as f:
    with open('amazon_qa_dataset/ratio_data/qa_summary_filtered_train_80%.json') as f:
        data = json.loads(f.read())
        print(len(data))
        exit()
        random.shuffle(data)
        total = len(data)
        #single = int(len(data)/4)
    
    with open('amazon_qa_dataset/ratio_data/qa_summary_filtered_train_20%.json','w') as f:
        json.dump(data[:int(total*0.2)], f, indent=2)

    with open('amazon_qa_dataset/ratio_data/qa_summary_filtered_train_40%.json','w') as f:
        json.dump(data[:int(total*0.4)], f, indent=2)
    
    with open('amazon_qa_dataset/ratio_data/qa_summary_filtered_train_60%.json','w') as f:
        json.dump(data[:int(total*0.6)], f, indent=2)
    
    with open('amazon_qa_dataset/ratio_data/qa_summary_filtered_train_80%.json','w') as f:
        json.dump(data[:int(total*0.8)], f, indent=2)
        
        #one = [each['asin'] for each in data[:single]]
        #two = [each['asin'] for each in data[:single*2]]
        #three = [each['asin'] for each in data[:single*3]]
        #print(len(one), len(two), len(three))
        #print(one[:10])
        #print(two[:10])
        #print(three[:10])

def split_data_cat():
    
    with open('amazon_qa_dataset/amazon_qa_summary_filtered.json') as f:
        data = json.loads(f.read())
        print(len(data))
        category = sorted(list(set([each['category'] for each in data])))
        print(category)
        for cat in category:
            each_cat_data = [product for product in data if product['category'] == cat]
            print(cat, len(each_cat_data))
            with open('amazon_qa_dataset/cat_data/qa_summary_filtered_'+cat.lower()+'.json','w') as outfile:
                json.dump(each_cat_data, outfile, indent=2)

def get_data_from_raw(i):
    
    with open('amazon_qa_dataset/amazon_qa_summary_filtered.json') as f:
    #with open('amazon_qa_summary_nofilter.json') as f:
        data = json.loads(f.read())
        print(len(data))
        category = sorted(list(set([each['category'] for each in data])))
        print(category)
        summary, qa_pairs, ques, ans = [], [], [], []
        #cnt_list = []
        cnt = 0
        for product in data:
            # input QA pairs
            if product['category'] == category[i]:
                cnt += 1
                #pprint(product)
                for each in product["summary"]:
                    summary.append(each) 
                for qa_pair in product["qa_pair"]:
                    qa_pairs.append(qa_pair["question"]+" "+qa_pair["answer"])
                    ques.append(qa_pair["question"])
                    ans.append(qa_pair["answer"])
                #cnt_list.append(cnt)
    
    print(len(ques), len(set(ques)))
    qw_cnt = [len(word_tokenize(each)) for each in list(set(ques))]
    aw_cnt = [len(word_tokenize(each)) for each in ans]

    stats = {"max_q":max(qw_cnt), "max_a":max(aw_cnt),
             "median_q":median(qw_cnt), "median_a":median(aw_cnt),
             "mean_q":mean(qw_cnt), "mean_a":mean(aw_cnt),
             "avg_q":sum(qw_cnt)/len(set(ques)), "avg_a":sum(aw_cnt)/len(ans),
             "q":len(set(ques))/cnt, "a":len(ans)/cnt, "entities":cnt}
    
    print(stats)
    with open('stats_ana/'+category[i]+'_stats.json','w') as f:
        json.dump(stats, f, indent=2)
    
    #exit()
    #print() 
    #print(len(qa_pairs)) 
    #print(max(cnt_list))
    #exit()
    #    mean = sum(cnt_list)/len(cnt_list)
    #    _max = max(cnt_list)
    #    _min = min(cnt_list)
    #    a = cnt_list[int(len(cnt_list)*0.8)]
    #    b = cnt_list[int(len(cnt_list)*0.9)]
    #    c = cnt_list[int(len(cnt_list)*0.99)]
    #
    #print(mean, _min, _max, a, b, c)
    #exit()

    #analysis(summary)
    #print('sum:',len(vocab_counter))
    ##print(summary[:3])
    #analysis(qa_pairs)
    #print('sum+qa:',len(vocab_counter))
    ##print(qa_pairs[:3])
    ##analysis(ques)
    ##print(ques[:3])
    ##analysis(ans)
    #vocab_counter_sorted = {k: v for k, v in sorted(vocab_counter.items(), key=lambda x: x[1], reverse=True)}
    #with open('vocab_amazon_qa','w',encoding='utf-8') as f:
    #    for k, v in vocab_counter_sorted.items():
    #        if v >= 3: 
    #            f.write(k+' '+str(v)+'\n')
    #f.close()
    #print(ans[:3])

def get_data_from_datasets(qa_summary_data):

    all_hypothesis = []
    all_references = []
    
    for batch in qa_summary_data:
        inputs = batch['qa_pairs_text'] 
        references = batch['summary']
        #print('input:',inputs)
        #print('ref:',references)
        all_hypothesis.append(inputs)
        all_references.append(references)
        #all_references.append([references])
   
    #analysis(all_hypothesis)
    #analysis(all_references)
    
    #print(len(all_hypothesis),all_hypothesis[:3])
    print(len(all_references))
    #pprint(all_references)
    return all_references

def prepare_results(m, p, r, f):
    return '\t{}:\t{}: {:5.2f}\t{}: {:5.2f}\t{}: {:5.2f}'.format(m, 'P', 100.0 * p, 'R', 100.0 * r, 'F1', 100.0 * f)

def evaluate(all_references, all_hypothesis):
        

    #print(all_references[:3])
    #print()
    #print(all_hypothesis[:3])
    for aggregator in ['Avg', 'Best', 'Individual']:
        print('Evaluation with {}'.format(aggregator))
        apply_avg = aggregator == 'Avg'
        apply_best = aggregator == 'Best'
        
        evaluator = rouge.Rouge(metrics=["rouge-n", "rouge-l"], max_n=2, limit_length=False,
                                apply_avg=True, stemming=True)

        #hypothesis_1 = "King Norodom Sihanouk has declined requests to chair a summit of Cambodia 's top political leaders , saying the meeting would not bring any progress in deadlocked negotiations to form a government .\nGovernment and opposition parties have asked King Norodom Sihanouk to host a summit meeting after a series of post-election negotiations between the two opposition groups and Hun Sen 's party to form a new government failed .\nHun Sen 's ruling party narrowly won a majority in elections in July , but the opposition _ claiming widespread intimidation and fraud _ has denied Hun Sen the two-thirds vote in parliament required to approve the next government .\n"
        #references_1 = ["Prospects were dim for resolution of the political crisis in Cambodia in October 1998.\nPrime Minister Hun Sen insisted that talks take place in Cambodia while opposition leaders Ranariddh and Sam Rainsy, fearing arrest at home, wanted them abroad.\nKing Sihanouk declined to chair talks in either place.\nA U.S. House resolution criticized Hun Sen's regime while the opposition tried to cut off his access to loans.\nBut in November the King announced a coalition government with Hun Sen heading the executive and Ranariddh leading the parliament.\nLeft out, Sam Rainsy sought the King's assurance of Hun Sen's promise of safety and freedom for all politicians.",
        #                "Cambodian prime minister Hun Sen rejects demands of 2 opposition parties for talks in Beijing after failing to win a 2/3 majority in recent elections.\nSihanouk refuses to host talks in Beijing.\nOpposition parties ask the Asian Development Bank to stop loans to Hun Sen's government.\nCCP defends Hun Sen to the US Senate.\nFUNCINPEC refuses to share the presidency.\nHun Sen and Ranariddh eventually form a coalition at summit convened by Sihanouk.\nHun Sen remains prime minister, Ranariddh is president of the national assembly, and a new senate will be formed.\nOpposition leader Rainsy left out.\nHe seeks strong assurance of safety should he return to Cambodia.\n",
        #                ]

        #hypothesis_2 = "China 's government said Thursday that two prominent dissidents arrested this week are suspected of endangering national security _ the clearest sign yet Chinese leaders plan to quash a would-be opposition party .\nOne leader of a suppressed new political party will be tried on Dec. 17 on a charge of colluding with foreign enemies of China '' to incite the subversion of state power , '' according to court documents given to his wife on Monday .\nWith attorneys locked up , harassed or plain scared , two prominent dissidents will defend themselves against charges of subversion Thursday in China 's highest-profile dissident trials in two years .\n"
        #references_2 = "Hurricane Mitch, category 5 hurricane, brought widespread death and destruction to Central American.\nEspecially hard hit was Honduras where an estimated 6,076 people lost their lives.\nThe hurricane, which lingered off the coast of Honduras for 3 days before moving off, flooded large areas, destroying crops and property.\nThe U.S. and European Union were joined by Pope John Paul II in a call for money and workers to help the stricken area.\nPresident Clinton sent Tipper Gore, wife of Vice President Gore to the area to deliver much needed supplies to the area, demonstrating U.S. commitment to the recovery of the region.\n"

        #all_hypothesis = [hypothesis_1, hypothesis_2]
        #all_references = [references_1, references_2]

        scores = evaluator.get_scores(all_hypothesis, all_references)

        for metric, results in sorted(scores.items(), key=lambda x: x[0]):
            if not apply_avg and not apply_best: # value is a type of list as we evaluate each summary vs each reference
                for hypothesis_id, results_per_ref in enumerate(results):
                    nb_references = len(results_per_ref['p'])
                    for reference_id in range(nb_references):
                        print('\tHypothesis #{} & Reference #{}: '.format(hypothesis_id, reference_id))
                        print('\t' + prepare_results(metric,results_per_ref['p'][reference_id], results_per_ref['r'][reference_id], results_per_ref['f'][reference_id]))
                print()
            else:
                print(prepare_results(metric, results['p'], results['r'], results['f']))
        print()     

def load_data(data_list):
    summary_list = []
    for each in data_list:
        #print(each)
        with open(each) as f:
            text = f.read()
            text = ' '.join(text.split('\n'))
            #print(text)
            #print(type(text))
            summary_list.append(text)
    return summary_list

ref_dirs = '/data01/tingyao/qa-summarization/fast_abs_rl/amazon_qa_dataset/finished_files/refs/test'
pred_dirs = '/data01/tingyao/qa-summarization/fast_abs_rl/decoded'

def fast_abs_result():
    
    #ref_list = os.listdir(ref_dirs)
    #print(ref_list)
    #ref_list = sorted(ref_list, key=lambda x:int(x.split('.')[0]))
    #ref_list = [os.path.join(ref_dirs,each) for each in ref_list]
    #print(ref_list)
    #references = load_data(ref_list)
    #print(references[:4])
    #references = [[each] for each in references]
    #print('ref:',len(references))
    #print()
    references = get_data_from_datasets(qa_summary_test)    
    for beam in ['beam_1', 'beam_5']:
        pred_list = os.listdir(os.path.join(pred_dirs,beam)+'/output')
        pred_list = sorted(pred_list, key=lambda x:int(x.split('.')[0]))
        pred_list = [os.path.join(pred_dirs,beam)+'/output/'+each for each in pred_list] 
        if beam == 'beam_1':
            hypo_1 = load_data(pred_list)
            #hypo_1 = [each[0] for each in hypo_1]
        else:
            hypo_5 = load_data(pred_list)
            #hypo_5 = [each[0] for each in hypo_5]
    
    fast_abs_list = []
    for (r1, h1, h5) in zip(references, hypo_1, hypo_5):
        fast_abs_list.append({"pred (beam = 1)":h1, 
                              "pred (beam = 5)":h5,
                              "ref":r1})
    
    with open('fast_abs_rl.json','w',encoding='utf-8') as outfile:
        json.dump(fast_abs_list, outfile, indent=2)
    
    #print(len(hypo_1))
    #print(len(hypo_5))
    #evaluate(references, hypo_1)
    #print('============')
    #evaluate(references, hypo_5)

def fast_abs_eval():
    with open('fast_abs_rl.json') as infile:
        data = json.loads(infile.read())
        print(len(data))
        ref_list, pred_1, pred_5 = [], [], []
        for each in data:
            ref_list.append([each['ref']])
            pred_1.append(each['pred (beam = 1)'])
            pred_5.append(each['pred (beam = 5)'])
   
    print(len(ref_list))
    print(len(pred_1))
    print(len(pred_5))
    #evaluate(ref_list, pred_1)
    #evaluate(ref_list, pred_5)
    eval_bert_scores(pred_5, ref_list)

def single_pairs_result():
    path = '/home/txh357/tingyao/summarization-sing-pair-mix/logs/amazon_qa_pg_bert_both'
    all_info = []
    with open(os.path.join(path,'beam_1.json')) as infile:
        data_1 = json.loads(infile.read())
        print(len(data_1))
    with open(os.path.join(path,'beam_5.json')) as infile:
        data_5 = json.loads(infile.read())
        print(len(data_5))
        for beam_1, beam_5 in zip(data_1, data_5):
            dicts = {}
            dicts['pred (beam = 1)'] = ' '.join(beam_1['pred (beam = 1)'])
            dicts['pred (beam = 5)'] = ' '.join(beam_5['pred (beam = 5)'])
            dicts['ref'] = [' '.join(item) for item in beam_1['ref']]
            all_info.append(dicts)
    
    with open('single_pair_sum.json','w') as outfile:
        json.dump(all_info, outfile, indent=2)

def single_pairs_eval():
    with open('single_pair_sum.json') as infile:
        data = json.loads(infile.read())
        print(len(data))
        ref_list, pred_1, pred_5 = [], [], []
        for each in data:
            ref_list.append(each['ref'])
            pred_1.append(each['pred (beam = 1)'])
            pred_5.append(each['pred (beam = 5)'])
   
    #print(len(ref_list))
    #print(len(pred_1))
    #print(len(pred_5))
    eval_bert_scores(pred_5, ref_list)
    #evaluate(ref_list, pred_1)
    #evaluate(ref_list, pred_5)
        #print(ref_list)

def eval_bert_scores(pred, ref):
    bertscore = load_metric("bertscore")
    results = bertscore.compute(predictions=pred, references=ref, lang="en")
    #print(results['precision'], type(results['precision']))
    scores = {'precision':sum(results['precision'])/len(results['precision']),
              'recall':sum(results['recall'])/len(results['recall']),
              'f1':sum(results['f1'])/len(results['f1'])}
    print(scores)

if __name__ == "__main__":
   
    #split_data_cat()
    #get_data_from_datasets(qa_summary_test)
    #fast_abs_eval()
    #fast_abs_result()
    #single_pairs_result()
    #single_pairs_eval()
    split_train_data()
    #get_data_from_raw()
    #analysis()
