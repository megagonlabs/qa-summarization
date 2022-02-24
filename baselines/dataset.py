from datetime import datetime
import json
import os
from typing import Optional

import datasets
import torch
import pandas as pd
from pytorch_lightning import LightningDataModule, LightningModule, Trainer, seed_everything
from torch.utils.data import DataLoader
from transformers import (
    AdamW,
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
    PreTrainedTokenizer
)

class QASumDataModule(LightningDataModule):
    def __init__(self,
                 tokenizer: PreTrainedTokenizer,
                 json_filepath: str = None,
                 version_name: str = None,
                 concat_style: str = None,
                 q_prefix: str = "Q: ",
                 a_prefix: str = "A: ",
                 max_input_length: int = 1024,
                 max_target_length: int = 512,
                 batch_size: int = 32):
        """
        tokenizer (transformers.PreTrainedTokenizer): Pretrained tokenizer

        json_filepath (str): JSON filepath

        version_name (str):
            original: Only annotated QA pairs
            enhanced: All QA pairs including enhanced

        concat_style (str):
            qa_pair: question1 answer1-1 question1 answer-1-2, ...
            qa_list: question1 answer1-1 answer1-2, ...
        """

        super().__init__()
        self.tokenizer = tokenizer

        if json_filepath is None:
            json_filepath = "./amazon_qa_summary_filtered.json" # TODO
        assert os.path.exists(json_filepath)

        self.json_filepath = json_filepath

        # Different dataset split to be discussed
        assert version_name in [None, "original", "enhanced"]
        if version_name is None:
            version_name = "original"
        self.version_name = version_name

        assert concat_style in [None, "qa_pair", "qa_list"]
        if concat_style is None:
            concat_style = "qa_pair"
        self.concat_style = concat_style

        self.q_prefix = q_prefix
        self.a_prefix = a_prefix

        self.max_input_length = max_input_length
        self.max_target_length = max_target_length

        self.batch_size = batch_size

    def _preprocess_function(self, example):
        # The input_ids, attention_mask, labels will be converted into torch.Tensor using datasets.set_format later
        model_inputs = self.tokenizer(example["input_data"],
                                      max_length=self.max_input_length,
                                      padding="max_length",
                                      truncation=True)

        # Setup the tokenizer for targets
        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(example["summary_list"][0],  # Use only first summary
                                    max_length=self.max_target_length,
                                    padding="max_length",
                                    truncation=True)

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    def prepare_data(self):
        data_list = []
        with open(self.json_filepath) as fin:
            products = json.load(fin)
            for product in products:                
                qa_data_list = []
                for qa_pair in product["qa_pair"]:
                    if "original" in self.version_name and "annotation" not in qa_pair:
                        # Skip enhanced QA pairs
                        continue

                    qa_data_list.append({"Q": qa_pair["question"],
                                         "A": qa_pair["answer"],
                                         "qaid": qa_pair["qaid"],
                                         "qid": qa_pair["qid"],
                                         "qa_index": qa_pair["qa_index"] if "qa_index" in qa_pair else -1})
                                         # qa_index may not be available for enhanced QA pairs

                # Most products have one reference summary
                summary_list = product["summary"]
                data_list.append([product["asin"],
                                  product["category"],
                                  qa_data_list,
                                  summary_list])

        self.data_df = pd.DataFrame(data_list,
                columns=["asin", "category", "qa_data_list", "summary_list"])

        input_data_list = []
        for index, row in self.data_df.iterrows():
            if self.concat_style == "qa_pair":
                qa_data_list = row["qa_data_list"]
                input_qa_list = []
                for qa_data in qa_data_list:
                    input_qa_list.append(
                        "{} {}".format(
                            self.q_prefix + qa_data["Q"],
                            self.a_prefix + qa_data["A"]))
                input_data_list.append(" ".join(input_qa_list))
            elif self.concat_style == "qa_list":
                # TODO(Yoshi)
                raise NotImplementedError("Coming soon!")
            else:
                raise ValueError("Invalid concat_style: {}".format(self.concat_style))
        
        self.data_df["input_data"] = input_data_list

        # Shuffle dataframe
        self.data_df = self.data_df.sample(frac=1, random_state=4649)

        # TODO(Yoshi): Organize train/dev/test split
        train_idx = int(len(self.data_df) * 0.6)
        valid_idx = int(len(self.data_df) * 0.8)

        self.train_data_df = self.data_df.iloc[:train_idx][["input_data", "summary_list"]]
        self.valid_data_df = self.data_df.iloc[train_idx:valid_idx][["input_data", "summary_list"]]
        self.test_data_df  = self.data_df.iloc[valid_idx:][["input_data", "summary_list"]]

        # Dataset.from_pandas(df)
        self.train_dataset = datasets.Dataset.from_pandas(self.train_data_df, preserve_index=False)
        self.valid_dataset = datasets.Dataset.from_pandas(self.valid_data_df, preserve_index=False)
        self.test_dataset = datasets.Dataset.from_pandas(self.test_data_df, preserve_index=False)

        self.train_dataset = self.train_dataset.map(self._preprocess_function,
                                                    remove_columns=["input_data", "summary_list"])
        self.train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])                                           
        self.valid_dataset = self.valid_dataset.map(self._preprocess_function,
                                                    remove_columns=["input_data", "summary_list"])
        self.valid_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])                                           
        self.test_dataset = self.test_dataset.map(self._preprocess_function,
                                                  remove_columns=["input_data", "summary_list"])
        self.test_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])                                           

    def setup(self, stage: Optional[str] = None):
        pass

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.valid_dataset,
                batch_size=self.batch_size)        

    def test_dataloader(self):
        return DataLoader(self.test_dataset,
                batch_size=self.batch_size)

    @staticmethod
    def collate_fn(batch):
        """
        Groups multiple examples into one batch with padding and tensorization.
        The collate function is called by PyTorch DataLoader
        """
        pad_token_id = 1
        input_ids, output_ids = list(zip(*batch))
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=pad_token_id)
        output_ids = torch.nn.utils.rnn.pad_sequence(output_ids, batch_first=True, padding_value=pad_token_id)
        return input_ids, output_ids
        

if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("t5-base")
    dm = QASumDataModule(tokenizer)
    dm.prepare_data()

    # dataset.set_format(type='torch', columns=['input_ids', 'token_type_ids', 'attention_mask', 'label'])

    dl = DataLoader(dm.train_dataset, batch_size=8)
