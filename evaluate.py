import torch

from datasets import load_dataset, load_metric
from transformers import LEDTokenizer, LEDForConditionalGeneration

# load qa dataset
#pubmed_test = load_dataset("scientific_papers", "pubmed", ignore_verifications=True, split="test")
qa_summary_test = load_dataset("qa_summary.py", ignore_verifications=True, split="test")

# load tokenizer & model from checkpoint
tokenizer = LEDTokenizer.from_pretrained("{name}/led-large-16384-pubmed")
model = LEDForConditionalGeneration.from_pretrained("{name}/led-large-16384-pubmed").to("cuda").half()


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


result = qa_summary_test.map(generate_answer, batched=True, batch_size=4)

# load rouge
rouge = load_metric("rouge")

print("Result:", rouge.compute(predictions=result["predicted_summary"], references=result["summary"], rouge_types=["rouge2"])["rouge2"].mid)
