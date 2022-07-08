from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import json
from tqdm import tqdm

def process(data_file, tokenizer, model, output_file):
	data = json.load(open(data_file))
	qa_pairs = []
	for entity in tqdm(data):
		for qa in entity["qa_pair"]:
			q = qa["question"]
			a = qa["answer"]
			qa["declaritive"] = convert_qas("{} {}".format(q, a), tokenizer, model)

	with open(output_file, "w") as file:
		file.write(json.dumps(data, indent=4))

def convert_qas(input_seq, tokenizer, model):
	input_ids = tokenizer(input_seq, return_tensors="pt").input_ids.to("cuda")
	outputs = model.generate(input_ids, min_length=5, max_length=input_ids.size(1), repetition_penalty=2.0, length_penalty=0.5)
	return tokenizer.decode(outputs[0], skip_special_tokens=True).replace("\n", "")

if __name__=="__main__":
	tokenizer_model = "allenai/led-base-16384"
	led_model = "../models/led-declaritive-slim"
	tokenizer = AutoTokenizer.from_pretrained(tokenizer_model)
	model = AutoModelForSeq2SeqLM.from_pretrained(led_model).to("cuda")
	data_files = ["../amazon_qa_dataset/qa_summary_filtered_train.json",
		"../amazon_qa_dataset/qa_summary_filtered_val.json",
		"../amazon_qa_dataset/qa_summary_filtered_test.json"]
	output_files = ["data/qa_summary_filtered_train_dec.json",
		"data/qa_summary_filtered_val_dec.json",
		"data/qa_summary_filtered_test_dec.json"]

	for i in range(len(data_files)):
		process(data_files[i], tokenizer, model, output_files[i])