# MegagonLabs_QA_summarization
QA summary

**"amazon_qa_filtered_5_150_8q.csv":** selected 8 QA pairs for rewrite

**"amazon_qa_summary_all.json":** QA summary include augmented data and mturk rewrite

**"mturk_best_rewrite.json":** best QA rewrite data

**"qa_annotated_summary_full.json":** QA rewrite data include error evaluation

**"model_config.txt":** transformer-based model training script. Reference Hugginface [summarization repo](https://github.com/huggingface/transformers/tree/master/examples/pytorch/summarization). 

**"sample_data":** data format for transformer-based model experiment.

**"transformer_summarization.py":** main part for transformer-based model experiment. Further modification needed.

**"baseline_summerize.py":** unsuperived methods (e.g. lexrank, sumbasic) and evaluation. 
