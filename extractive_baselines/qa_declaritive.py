import json
import pandas as pd

data = json.load(open("../amazon_qa_dataset/qa_summary_filtered_train.json"))

pairs = []

for entity in data:
	for qa in entity["qa_pair"]:
		if "annotation" not in qa:
			continue
		q = qa["question"]
		a = qa["answer"]
		for annotation in qa["annotation"]["rewrite"]:
			if "?" in annotation["edit"]:
				continue
			if annotation["is_selected"] == "False":
				continue
			pairs.append(("{} {}".format(q, a), annotation["edit"]))

pairs = pd.DataFrame(pairs, columns=["text", "summary"])
pairs.to_csv("data/qa_declaritive.csv", index=None)
print(len(pairs))