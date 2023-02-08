import json
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm import tqdm
from nltk.tokenize import sent_tokenize

def convert_qas(summary, tokenizer, model):
	qas = summary.strip().split("     ")
	revised = []
	for qa in qas:
		input_ids = tokenizer(qa, return_tensors="pt").input_ids.to("cuda")
		outputs = model.generate(input_ids, min_length=5, max_length=input_ids.size(1), repetition_penalty=2.0, length_penalty=0.5)
		qa_revised = tokenizer.decode(outputs[0], skip_special_tokens=True).replace("\n", "")
		revised.append(qa_revised)
		#print(input_ids.size(1), qa, " #### ", qa_revised)
	return " ".join(revised)

def run_lexrank_led(tokenizer_model, led_model, lexrank_file, lexrank_led_file):
	tokenizer = AutoTokenizer.from_pretrained(tokenizer_model)
	model = AutoModelForSeq2SeqLM.from_pretrained(led_model).to("cuda")
	lexrank = json.load(open(lexrank_file))
	lexrank_led = []
	for item in tqdm(lexrank):
		summary = convert_qas(item["pred"], tokenizer, model)
		lexrank_led.append({"pred": summary, "ref": item["ref"]})
	with open(lexrank_led_file, "w") as file:
		file.write(json.dumps(lexrank_led, indent=4))
		file.close()

if __name__=="__main__":
	tokenizer_model = "allenai/led-base-16384"
	led_model = "../models/led-declaritive-slim"
	lexrank_file = "../generation/lexrank.json"
	lexrank_led_file = "../generation/lexrank_led.json"
	run_lexrank_led(tokenizer_model, led_model, lexrank_file, lexrank_led_file)