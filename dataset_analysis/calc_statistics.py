import argparse
from collections import defaultdict, Counter
import json

import pandas as pd


"""
TODO(Yoshi): Fix code
"""


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--filepath", type=str, default="amazon_qa_summary_filtered.json")
    args = parser.parse_args()

    product_cnt = Counter()
    questiontype_cnt_dict = defaultdict(Counter)
    with open(args.filepath) as fin:
        products = json.load(fin)
        for product in products:
            print(product["asin"])
            product_cnt[product["category"]] += 1
            for qa_pair in product["qa_pair"]:
                questiontype_cnt_dict[product["category"]][qa_pair["questionType"]] += 1
                print("Q:", qa_pair["question"])
                print("A:", qa_pair["answer"])
            for summary in product["summary"]:
                print("Summary:", summary)

    df = pd.DataFrame(questiontype_cnt_dict).T
    df["total"] = df["open-ended"] + df["yes/no"]
    # df.sort_values("total", ascending=False).to_markdown()

    products = json.load(open(args.filepath))

    original_qapair_cnt = Counter()
    all_qapair_cnt = Counter()

    for product in products:
        # Input QA pairs
        input_qa_pairs = []
        for qa_pair in product["qa_pair"]:
            if "annotation" in qa_pair and qa_pair["annotation"] is not None:
                original_qapair_cnt[product["category"]] += 1
            all_qapair_cnt[product["category"]] += 1

    qapair_cnt_df = pd.concat([pd.Series(product_cnt),
                               pd.Series(original_qapair_cnt),
                               pd.Series(all_qapair_cnt)], axis=1)
    qapair_cnt_df.columns = ["product_cnt", "orig_qapair_cnt", "all_qapair_cnt"]
                    

#        # Generate a summary from QA pairs
#        generated_summary = summarizer(input_qa_pairs)
#
#        # Reference summaries
#        reference_summaries = product["summary"]
