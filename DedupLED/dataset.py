from datetime import datetime
import itertools
import json
import os
import random
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



class QASumPreSplitDataModule(LightningDataModule):
    def __init__(self,
                 tokenizer: PreTrainedTokenizer,
                 train_json_filepath: str = None,
                 valid_json_filepath: str = None,
                 test_json_filepath: str = None,
                 version_name: str = None,
                 concat_style: str = None,
                 q_prefix: str = "Q: ",
                 a_prefix: str = "A: ",
                 max_input_length: int = 1024,
                 max_clss: int=50,
                 max_target_length: int = 512,
                 batch_size: int = 32,
                 *args,
                 **kwargs):
        """
        tokenizer (transformers.PreTrainedTokenizer): Pretrained tokenizer

        train_json_filepath (str): JSON filepath
        valid (str): JSON filepath
        test_json_filepath (str): JSON filepath

        version_name (str):
            original: Only annotated QA pairs
            enhanced: All QA pairs including enhanced

        concat_style (str):
            qa_pair: question1 answer1-1 question1 answer-1-2, ...
            qa_list: question1 answer1-1 answer1-2, ...
        """

        super().__init__()
        self.tokenizer = tokenizer
        if train_json_filepath is None:
            train_json_filepath = "../amazon_qa_dataset/qa_summary_filtered_train.json"
        assert os.path.exists(train_json_filepath)
        if valid_json_filepath is None:
            valid_json_filepath = "../amazon_qa_dataset/qa_summary_filtered_val.json"
        assert os.path.exists(valid_json_filepath), valid_json_filepath
        if test_json_filepath is None:
            test_json_filepath = "../amazon_qa_dataset/qa_summary_filtered_test.json"
        assert os.path.exists(test_json_filepath)

        self.train_json_filepath = train_json_filepath
        self.valid_json_filepath = valid_json_filepath
        self.test_json_filepath = test_json_filepath

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
        self.max_clss = max_clss
        self.max_target_length = max_target_length

        self.batch_size = batch_size

    def _truncate(self, input_ids, input_labels):
        seq_len = sum([len(ids) for ids in input_ids])
        while seq_len > self.max_input_length or len(input_labels) > self.max_clss:
            i = random.randint(0, len(input_labels)-1)
            if input_labels[i] == 1:
                continue
            seq_len -= len(input_ids[i])
            input_ids.pop(i)
            input_labels.pop(i)
        assert len(input_ids) == len(input_labels)
        return input_ids, input_labels

    def _preprocess_function(self, example):
        # The input_ids, attention_mask, labels will be converted into torch.Tensor using datasets.set_format later
        encoded = self.tokenizer(example["input_data"])
        input_ids, clss_label = self._truncate(encoded.input_ids, example["input_data_label"])
        clss = [0]
        for ids in input_ids:
            clss.append(clss[-1]+len(ids))
        clss.pop()
        assert len(clss) == len(clss_label)
        # prepare input
        input_ids = list(itertools.chain(*input_ids))
        attention_mask = [1] * len(input_ids)
        clss_mask = [1] * len(clss)
        # padding
        num_input_padding = self.max_input_length - len(input_ids)
        num_clss_padding = self.max_clss - len(clss)
        input_ids += [self.tokenizer.pad_token_id] * num_input_padding
        attention_mask += [0] * num_input_padding
        clss += [self.tokenizer.pad_token_id] * num_clss_padding
        clss_mask += [0] * num_clss_padding
        clss_label += [0] * num_clss_padding

        # Setup the tokenizer for targets
        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(example["summary_list"][0],  # Use only first summary
                                    max_length=self.max_target_length,
                                    padding="max_length",
                                    truncation=True)
        model_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels["input_ids"],
            "clss": clss,
            "clss_mask": clss_mask,
            "clss_label": clss_label
        }
        return model_inputs

    def prepare_data(self):
        for name, json_filepath in [("train_data_df", self.train_json_filepath),
                                    ("valid_data_df", self.valid_json_filepath),
                                    ("test_data_df", self.test_json_filepath)]:
            data_list = []
            with open(json_filepath) as fin:
                products = json.load(fin)
                for product in products:                
                    qa_data_list = []
                    for qa_pair in product["qa_pair"]:
                        if "original" in self.version_name and "annotation" not in qa_pair:
                            # Skip enhanced QA pairs
                            continue

                        qa_data_list.append({"Q": qa_pair["question"],
                                             "A": qa_pair["answer"],
                                             "label": 1 if "annotation" in qa_pair else 0,
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
            data_df = pd.DataFrame(data_list,
                                   columns=["asin", "category", "qa_data_list", "summary_list"])

            input_data_list = []
            input_data_label = []
            for index, row in data_df.iterrows():
                if self.concat_style == "qa_pair":
                    qa_data_list = row["qa_data_list"]
                    input_qa_list = []
                    input_qa_label = []
                    for qa_data in qa_data_list:
                        input_qa_list.append(
                            "{} {}".format(
                                self.q_prefix + qa_data["Q"],
                                self.a_prefix + qa_data["A"]))
                        input_qa_label.append(qa_data["label"])
                    input_data_list.append(input_qa_list)
                    input_data_label.append(input_qa_label)
                elif self.concat_style == "qa_list":
                    # TODO(Yoshi)
                    raise NotImplementedError("Coming soon!")
                else:
                    raise ValueError("Invalid concat_style: {}".format(self.concat_style))
        
            data_df["input_data"] = input_data_list
            data_df["input_data_label"] = input_data_label
            setattr(self, name, data_df) # e.g., self.train_data_df = data_df

        # Dataset.from_pandas(df)
        self.train_dataset = datasets.Dataset.from_pandas(self.train_data_df, preserve_index=False)
        self.valid_dataset = datasets.Dataset.from_pandas(self.valid_data_df, preserve_index=False)
        self.test_dataset = datasets.Dataset.from_pandas(self.test_data_df, preserve_index=False)

        self.train_dataset = self.train_dataset.map(self._preprocess_function,
                                                    remove_columns=["input_data", "input_data_label", "summary_list"])
        self.train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels", "clss", "clss_mask", "clss_label"])                                           
        self.valid_dataset = self.valid_dataset.map(self._preprocess_function,
                                                    remove_columns=["input_data", "input_data_label", "summary_list"])
        self.valid_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels", "clss", "clss_mask", "clss_label"])                                           
        self.test_dataset = self.test_dataset.map(self._preprocess_function,
                                                  remove_columns=["input_data", "input_data_label", "summary_list"])
        self.test_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels", "clss", "clss_mask", "clss_label"])                                           

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

    @staticmethod
    def add_dataset_specific_args(parser):
        parser.add_argument("--train_json_filepath", type=str, default=None)
        parser.add_argument("--valid_json_filepath", type=str, default=None)
        parser.add_argument("--test_json_filepath", type=str, default=None)
        parser.add_argument("--version_name", type=str, default=None)
        parser.add_argument("--concat_style", type=str, default=None)
        parser.add_argument("--q_prefix", type=str, default="Q: ")
        parser.add_argument("--a_prefix", type=str, default="A: ")
        parser.add_argument("--max_input_length", type=int, default=1024)
        parser.add_argument("--max_target_length", type=int, default=512)

        return parser        


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
            json_filepath = "../amazon_qa_dataset/amazon_qa_summary_filtered.json"  # TODO
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
                raise NotImplementedError("Coming soon!")
            else:
                raise ValueError("Invalid concat_style: {}".format(self.concat_style))
        
        self.data_df["input_data"] = input_data_list

        # Shuffle dataframe
        self.data_df = self.data_df.sample(frac=1, random_state=4649)

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
    tokenizer = AutoTokenizer.from_pretrained("allenai/led-base-16384")
    # tokenizer = AutoTokenizer.from_pretrained("t5-base")
    dm = QASumPreSplitDataModule(tokenizer, version_name="enhanced")
    dm.prepare_data()

    dl = DataLoader(dm.train_dataset, batch_size=8)
    for batch in dl:
        for key, value in batch.items():
            print(key, value.size())
        print(batch["clss"][0], batch["input_ids"][0])
        break
