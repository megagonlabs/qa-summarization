# coding=utf-8
# Copyright 2020 The TensorFlow Datasets Authors and the HuggingFace Datasets Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
"""Amazon QA Dataset."""


import json
import os

import datasets


_CITATION = """
"""

_DESCRIPTION = """
Amazon QA summarization datasets contains two sets of long and structured documents.
  
  - qa_pairs_text: all qa pairs of each item
  - summary: human post-edit qa summarization

"""



class AmazonQAConfig(datasets.BuilderConfig):
    """BuilderConfig for Amazon QA dataset."""

    def __init__(self, filename=None, **kwargs):
        """BuilderConfig for QA data

        Args:
          filename: filename of different configs for the dataset.
          **kwargs: keyword arguments forwarded to super.
        """
        # 1.1.0 remove sentence breaker <S> and </S> in summary.
        super(AmazonQAConfig, self).__init__(version=datasets.Version("1.1.1"), **kwargs)
        self.filename = filename


class AmazonQA(datasets.GeneratorBasedBuilder):

    BUILDER_CONFIGS = [
        AmazonQAConfig(name="amazon_qa", description="Documents from Amazon QA dataset."),
    ]

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "qa_pairs_text": datasets.Value("string"),
                    "summary": datasets.Value("string")
                }
            ),
            supervised_keys=None,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        path = "amazon_qa_dataset"
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"path": os.path.join(path, "qa_summary_filtered_train.json")},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={"path": os.path.join(path, "qa_summary_filtered_val.json")},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={"path": os.path.join(path, "qa_summary_filtered_test.json")},
            ),
        ]

    def _generate_examples(self, path=None):
        """Yields examples."""
        with open(path, encoding="utf-8") as infile:
            data = json.load(infile)
            for product in data:
                # input QA pairs
                input_qa_pairs = []
                for qa_pair in product["qa_pair"]:
                    input_qa_pairs.append(qa_pair["question"])
                    input_qa_pairs.append(qa_pair["answer"])
                input_qa_text = " ".join(input_qa_pairs)
                input_qa_text = input_qa_text.replace("<S>"," ").replace("<S>","")
                yield product["asin"], {
                    "qa_pairs_text":input_qa_text,
                    "summary":product["summary"][0]
                }

