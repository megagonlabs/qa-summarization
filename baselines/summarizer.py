import argparse
import os

import datasets
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoConfig, AutoModelForSeq2SeqLM, set_seed, get_linear_schedule_with_warmup

from dataset import QASumDataModule


class Summarizer(pl.LightningModule):
    """https://github.com/allenai/naacl2021-longdoc-tutorial/blob/main/summarization.py"""
    """Pytorch Lightning module. It wraps up the model, data loading and training code"""

    def __init__(self, params):
        """Loads the model, the tokenizer and the metric."""
        super().__init__()
        self.args = params

        # Load and update config then load a pretrained LEDForConditionalGeneration
        config = AutoConfig.from_pretrained(self.args.model_name)
        config.gradient_checkpointing = self.args.grad_ckpt
        if self.args.model_name in ["allenai/led-base-16384"]:
            config.attention_window = [self.args.attention_window] * len(config.attention_window)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.args.model_name, config=config)

        # Load tokenizer and metric
        self.tokenizer = AutoTokenizer.from_pretrained(self.args.model_name, use_fast=True)
        self.rouge = datasets.load_metric('rouge')

    def _set_global_attention_mask(self, input_ids):
        """Configure the global attention pattern based on the task"""

        # Local attention everywhere - no global attention
        global_attention_mask = torch.zeros(input_ids.shape, dtype=torch.long, device=input_ids.device)

        # Gradient Checkpointing caveat 1:
        # For gradient checkpointing to work, all model parameters should contribute
        # to the computation of the loss. Remember that the self-attention layers in the LED model
        # have two sets of qkv layers, one for local attention and another for global attention.
        # If we don't use any global attention, the global qkv layers won't be used and
        # PyTorch will throw an error. This is just a PyTorch implementation limitation
        # not a conceptual one (PyTorch 1.8.1).
        # The following line puts global attention on the <s> token to make sure all model
        # parameters which is necessery for gradient checkpointing to work.
        global_attention_mask[:, 0] = 1

        # # Global attention on the first 100 tokens
        # global_attention_mask[:, :100] = 1

        # # Global attention on periods
        # global_attention_mask[(input_ids == self.tokenizer.convert_tokens_to_ids('.'))] = 1

        return global_attention_mask

    def forward(self, input_ids, output_ids):
        """Call LEDForConditionalGeneration.forward"""
        return self.model(input_ids,
                          attention_mask=(input_ids != self.tokenizer.pad_token_id),  # mask padding tokens
                          global_attention_mask=self._set_global_attention_mask(input_ids),  # set global attention
                          labels=output_ids, use_cache=False)

    def training_step(self, batch, batch_nb):
        """Call the forward pass then return loss"""
        outputs = self.forward(*batch)
        return {'loss': outputs.loss}

    def configure_optimizers(self):
        """Configure the optimizer and the learning rate scheduler"""
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr)
        dataset_size = len(self.hf_dataset['train'])
        gpu_count = torch.cuda.device_count()
        num_steps = dataset_size * self.args.epochs / gpu_count / self.args.grad_accum / self.args.batch_size
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=self.args.warmup,
                                                    num_training_steps=num_steps)
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

    def _evaluation_step(self, split, batch, batch_nb):
        """Validaton or Testing - predict output, compare it with gold, compute rouge1, 2, L, and log result"""
        # Generate
        input_ids, output_ids = batch
        generated_ids = self.model.generate(input_ids=input_ids,
                                            attention_mask=(input_ids != self.tokenizer.pad_token_id),
                                            global_attention_mask=self._set_global_attention_mask(input_ids),
                                            use_cache=True, max_length=self.args.max_output_len, num_beams=1)

        # Convert predicted and gold token ids to strings
        predictions = self.tokenizer.batch_decode(generated_ids.tolist(), skip_special_tokens=True)
        references = self.tokenizer.batch_decode(output_ids.tolist(), skip_special_tokens=True)

        # TODO(Yoshi): To make it compatible
        # Compute rouge
        metric_names = ['rouge1', 'rouge2', 'rougeL', 'rougeLsum']
        results = self.rouge.compute(predictions=predictions, references=references)
        for metric_name in metric_names:
            metric_val = input_ids.new_zeros(1) + results[metric_name].mid.fmeasure
            self.log(f'{split}_{metric_name}', metric_val, on_step=False, on_epoch=True, sync_dist=True, prog_bar=True)

    def validation_step(self, batch, batch_nb):
        self._evaluation_step("val", batch, batch_nb)

    def test_step(self, batch, batch_nb):
        self._evaluation_step("test", batch, batch_nb)

    @staticmethod
    def add_model_specific_args(parser):
        # **************** Parameters that we will NOT change during this tutorial **************** #
        parser.add_argument("--seed", type=int, default=1234, help="Seed")
        parser.add_argument("--lr", type=float, default=0.00003, help="Maximum learning rate")
        parser.add_argument("--warmup", type=int, default=1000, help="Number of warmup steps")
        parser.add_argument("--epochs", type=int, default=1, help="Number of epochs")
        parser.add_argument("--num_workers", type=int, default=4, help="Number of data loader workers")
        parser.add_argument("--limit_val_batches", default=0.005, type=float, help='Percent of validation data used')
        parser.add_argument("--limit_test_batches", default=0.005, type=float, help='Percent of test data used')
        parser.add_argument("--limit_train_batches", default=0.002, type=float, help='Percent of training data used')
        parser.add_argument("--max_output_len", type=int, default=256, help="maximum num of wordpieces in the summary")
        parser.add_argument("--output_dir", type=str, default='./saved_models/test', help="Location of output dir")
        parser.add_argument("--val_every", default=0.33, type=float, help='Validation every')

        # **************** Parameters that we will change during this tutorial **************** #
        parser.add_argument("--max_input_len", type=int, default=8192, help="maximum num of wordpieces in the input")
        parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
        parser.add_argument("--grad_accum", type=int, default=1, help="number of gradient accumulation steps")
        parser.add_argument("--fp16", action='store_true', help="Use fp16 ")
        parser.add_argument('--grad_ckpt', action='store_true', help='Enable gradient checkpointing to save memory')
        parser.add_argument("--attention_window", type=int, default=1024, help="Attention window")

        return parser


if __name__ == "__main__":
    # Setup command line args
    main_arg_parser = argparse.ArgumentParser(description="QA summarization")
    main_arg_parser.add_argument("--model_name", default="allenai/led-base-16384")
    parser = Summarizer.add_model_specific_args(main_arg_parser)
    args = parser.parse_args()

    # Init a PL module
    set_seed(args.seed)
    summarizer = Summarizer(args)

    # Load the arXiv dataset from HF datasets
    summarizer.hf_dataset = datasets.load_dataset('scientific_papers', 'arxiv')

    checkpoint_callback = ModelCheckpoint(monitor='val_rouge1',
                                          mode="max",
                                          dirpath=args.output_dir,
                                          save_top_k=3)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Construct a PL trainer
    trainer = pl.Trainer(gpus=1,
                         # Gradient Checkpointing caveat 2:
                         # For gradient checkpointing to work with DistributedDataParallel,
                         # the `find_unused_parameters` should be `False`. Without it,
                         # you get a not-very-helpful error message (PyTorch 1.8.1)
                         max_epochs=args.epochs,
                         num_sanity_val_steps=0,
                         default_root_dir=args.output_dir,
                         limit_val_batches=args.limit_val_batches,
                         limit_train_batches=args.limit_train_batches,
                         limit_test_batches=args.limit_test_batches,
                         precision=16 if args.fp16 else 32,
                         accumulate_grad_batches=args.grad_accum,
                         callbacks=[checkpoint_callback],
                         val_check_interval=args.val_every
                         )

    dm = QASumDataModule(tokenizer=summarizer.tokenizer,
                         batch_sie=args.batch_size)

    # Start training
    trainer.fit(summarizer,
                datamodule=dm)

    # Start testing
    result = trainer.test(datamodule=dm)
    print(result)