import argparse
from collections import defaultdict, Counter
import json

import pandas as pd
from nltk.util import ngrams


def get_ngrams(sentence: str, n: int):
    """Simple tokenization + NLTK ngrams"""
    return ngrams(sentence.lower().split(), n)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--filepath", type=str, default="amazon_qa_summary_filtered.json")
    parser.add_argument("--ignore_enhanced", action="store_true")
    parser.add_argument("--print_style", type=str, choices=["text", "latex", "markdown"], default="text")
    args = parser.parse_args()

    N = [1, 2, 3, 4]

    q_ngram_dict = defaultdict(lambda: defaultdict(int))
    a_ngram_dict = defaultdict(lambda: defaultdict(int))
    qa_ngram_dict = defaultdict(lambda: defaultdict(int))
    ref_ngram_dict = defaultdict(lambda: defaultdict(int))
    
    with open(args.filepath) as fin:
        products = json.load(fin)
        for product in products:
            for qa_pair in product["qa_pair"]:
                if args.ignore_enhanced and "annotation" not in qa_pair:
                    # Skip enhanced QA pairs
                    continue

                for n in N:
                    for ngram_tuple in get_ngrams(qa_pair["question"], n):
                        q_ngram_dict[n][ngram_tuple] += 1
                        qa_ngram_dict[n][ngram_tuple] += 1
                    for ngram_tuple in get_ngrams(qa_pair["answer"], n):
                        a_ngram_dict[n][ngram_tuple] += 1
                        qa_ngram_dict[n][ngram_tuple] += 1                        

            for summary in product["summary"]:
                for n in N:
                    for ngram_tuple in ngrams(summary.lower().split(), n):
                        ref_ngram_dict[n][ngram_tuple] += 1

    for name, input_ngram_dict in [("Q", q_ngram_dict),
                                   ("A", a_ngram_dict),
                                   ("QA", qa_ngram_dict)]:
        # for each n for N-gram
        novel_ngram_ratio_dict = {}
        for n in N:
            num_novel_ngram = len(list(filter(lambda x: x not in input_ngram_dict[n], ref_ngram_dict[n].keys())))
            novel_ngram_ratio = num_novel_ngram / float(len(ref_ngram_dict[n]))
            novel_ngram_ratio_dict[n] = novel_ngram_ratio

        ngram_df = pd.DataFrame(pd.Series(novel_ngram_ratio_dict)).reset_index()
        ngram_df.columns = ["n-gram", "novel ratio"]

        print(name)
        if args.print_style == "text":
            print(ngram_df)
        elif args.print_style == "latex":
            print(ngram_df.to_latex())
        elif args.print_style == "markdown":
            print(ngram_df.to_markdown())
        else:
            raise ValueError(args.print_style)
        print("-----\n")
        


