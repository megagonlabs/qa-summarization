import json
import sys


if __name__ == "__main__":
    score_names = [
        "test_rouge-1_p",
        "test_rouge-1_r",
        "test_rouge-1_f",
        "test_rouge-2_p",
        "test_rouge-2_r",
        "test_rouge-2_f",
        "test_rouge-l_p",
        "test_rouge-l_r",
        "test_rouge-l_f",
        "test_bertscore_p",
        "test_bertscore_r",
        "test_bertscore_f"]

    with open(sys.argv[1]) as fin:
        score_dict = json.load(fin)[0]
        scores = [str(score_dict[name] * 100) for name in score_names]
        print(" ".join(scores))


