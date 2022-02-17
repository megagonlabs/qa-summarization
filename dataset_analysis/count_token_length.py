import argparse
import json

import pandas as pd
from transformers import AutoTokenizer


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--filepath", type=str, default="amazon_qa_summary_filtered.json")
    parser.add_argument("--ignore_enhanced", action="store_true")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained("allenai/led-base-16384")

    q_length_list = []
    a_length_list = []
    qa_length_list = []
    summary_length_list = []
    with open(args.filepath) as fin:
        products = json.load(fin)
        for product in products:
            for qa_pair in product["qa_pair"]:
                if args.ignore_enhanced and "annotation" not in qa_pair:
                    # Skip enhanced QA pairs
                    continue

                q_len = len(tokenizer.encode(qa_pair["question"]))
                a_len = len(tokenizer.encode(qa_pair["answer"]))

                q_length_list.append(q_len)
                a_length_list.append(a_len)
                qa_length_list.append(q_len + a_len)

            for summary in product["summary"]:
                summary_length_list.append(
                    len(tokenizer.encode(summary)))

    summ_len_s = pd.Series(summary_length_list)
    q_len_s = pd.Series(q_length_list)
    a_len_s = pd.Series(a_length_list)
    qa_len_s = pd.Series(qa_length_list)
