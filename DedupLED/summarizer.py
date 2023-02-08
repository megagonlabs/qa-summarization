import argparse
import json
import os
import bert_score
import datasets
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoConfig, AutoModelForSeq2SeqLM, set_seed, get_linear_schedule_with_warmup
import rouge

from dataset import QASumDataModule, QASumPreSplitDataModule


class Summarizer(pl.LightningModule):
    """https://github.com/allenai/naacl2021-longdoc-tutorial/blob/main/summarization.py"""
    """Pytorch Lightning module. It wraps up the model, data loading and training code"""

    def __init__(self, params):
        """Loads the model, the tokenizer and the metric."""
        super().__init__()
        self.args = params

        # Load and update config then load a pretrained LEDForConditionalGeneration
        config = AutoConfig.from_pretrained(self.args.model_name)
        print("config load done")
        config.gradient_checkpointing = self.args.grad_ckpt
        if self.args.model_name in ["allenai/led-base-16384"]:
            config.attention_window = [self.args.attention_window] * len(config.attention_window)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.args.model_name, config=config)
        self.classifier = torch.nn.Sequential(
            torch.nn.Dropout(p=self.args.dropout_rate),
            torch.nn.Linear(self.model.config.d_model, self.model.config.d_model),
            torch.nn.Tanh(),
            torch.nn.Dropout(p=self.args.dropout_rate),
            torch.nn.Linear(self.model.config.d_model, self.args.num_classes),
            )

        # TODO: Hard-coded =================================
        # set generate hyperparameters
        self.model.config.num_beams = 4
        self.model.config.max_length = 512
        self.model.config.min_length = 100
        self.model.config.length_penalty = 2.0
        self.model.config.early_stopping = True
        self.model.config.no_repeat_ngram_size = 3
        # ==================================================

        print("model load done")

        # Load tokenizer and metric
        self.tokenizer = AutoTokenizer.from_pretrained(self.args.model_name) #, use_fast=True)
        print("tokenizer load done")

        print("rouge load done")

        self.rouge_scorer = rouge.Rouge(metrics=["rouge-n", "rouge-l"],
                                        max_n=2,
                                        limit_length=False,
                                        apply_avg=True,
                                        stemming=True, ensure_compatibility=True)
        self.bert_scorer = bert_score.BERTScorer(lang="en", rescale_with_baseline=True)        

        # Load dataset
        print(self.args)
        self.dm = QASumPreSplitDataModule(tokenizer=self.tokenizer,
                                          **vars(self.args))
        self.dm.prepare_data()

    def _set_global_attention_mask(self, input_ids):
        """Configure the global attention pattern based on the task"""

        # Local attention everywhere - no global attention
        global_attention_mask = torch.zeros(input_ids.shape, dtype=torch.long, device=input_ids.device)
        global_attention_mask[:, 0] = 1
        return global_attention_mask

    def forward(self, **batch):
        outputs = self.model(input_ids=batch["input_ids"], 
            attention_mask=batch["attention_mask"], 
            decoder_input_ids=None,
            labels=batch["labels"],
            return_dict=True)

        # calculate classification loss
        clss, clss_mask, clss_label = batch["clss"], batch["clss_mask"], batch["clss_label"]
        bsz = clss.size(0)
        hidden_states = outputs.encoder_last_hidden_state
        hidden_states = hidden_states[torch.arange(bsz).unsqueeze(1), clss] # shape = bsz, num_labels, hidden_dim
        hidden_states = hidden_states * clss_mask.view(bsz, -1, 1).float()
        logits = self.classifier(hidden_states)
        loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
        cls_loss = F.cross_entropy(logits.view(-1, self.args.num_classes), 
                                   clss_label.view(-1), 
                                   weight=torch.tensor([1.0, 100.0]).to(logits.device),
                                   reduction="none")

        cls_loss = cls_loss * clss_mask.view(-1)
        cls_loss = cls_loss.mean()
        # calculate total loss
        return outputs, logits, cls_loss

    def training_step(self, batch, batch_nb):
        """Call the forward pass then return loss"""
        outputs, _, cls_loss = self.forward(**batch)
        loss = self.args.summarizer_weight * outputs.loss + self.args.classifier_weight * cls_loss
        return {"loss": loss}

    def configure_optimizers(self):
        """Configure the optimizer and the learning rate scheduler"""
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr)
        dataset_size = len(self.dm.train_dataset) ## 
        gpu_count = torch.cuda.device_count()
        num_steps = dataset_size * self.args.epochs / gpu_count / self.args.grad_accum / self.args.batch_size
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=self.args.warmup,
                                                    num_training_steps=num_steps)
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

    def train_dataloader(self):
        return DataLoader(self.dm.train_dataset,
                batch_size=self.args.batch_size)

    def val_dataloader(self):
        return DataLoader(self.dm.valid_dataset,
                batch_size=self.args.batch_size)        

    def test_dataloader(self):
        return DataLoader(self.dm.test_dataset,
                batch_size=self.args.batch_size)

    def _evaluation_step(self, split, batch, batch_nb):
        """Validaton or Testing - predict output, compare it with gold, compute rouge1, 2, L, and log result"""
        # calculate f1 for classification
        _, logits, _ = self.forward(**batch)
        cls_label = batch["clss_label"]
        pred_label = logits.argmax(dim=-1)
        num_true_pred = torch.sum(pred_label*cls_label)
        num_pos_pred = torch.sum(pred_label)
        num_true_pos = torch.sum(cls_label)
        precision = float(num_true_pred.item()/num_pos_pred.item()) if num_pos_pred.item() > 0 else 0.0
        recall = float(num_true_pred.item()/num_true_pos.item()) if num_true_pos.item() > 0 else 0.0
        fmeasure = float((2*precision*recall)/(precision+recall)) if precision + recall > 0 else 0.0
        self.log(f"{split}_cls_f1", fmeasure, on_step=False, on_epoch=True, sync_dist=True, prog_bar=True)
        
        # generate text
        output_ids = batch["labels"]
        del batch["labels"]  # Patchy way to not pass labels to generate()
        generated_ids = self.model.generate(input_ids=batch["input_ids"], 
            attention_mask=batch["attention_mask"], use_cache=True)

        # Convert predicted and gold token ids to strings
        predictions = self.tokenizer.batch_decode(generated_ids.tolist(), skip_special_tokens=True)
        references = self.tokenizer.batch_decode(output_ids.tolist(), skip_special_tokens=True)
        # Compute ROUGE/BERT score
        scores = {}

        for metric, vals in self.rouge_scorer.get_scores(predictions, references).items(): # [[r] for r in references]
            for k, v in vals.items():
                scores[f"{metric}_{k}"] = v

        for k, v in zip("prf", self.bert_scorer.score(predictions, references)): # P, R, F
            scores[f"bertscore_{k}"] = v.mean().item()
        
        for metric_name, metric_val in scores.items():
            self.log(f'{split}_{metric_name}', metric_val, on_step=False, on_epoch=True, sync_dist=True, prog_bar=True)

        return predictions

    def validation_step(self, batch, batch_nb):
        return self._evaluation_step("val", batch, batch_nb)

    def test_step(self, batch, batch_nb):
        return self._evaluation_step("test", batch, batch_nb)

    def test_epoch_end(self, output_results):
        """Store generated summaries for test examples"""
        all_generations = [ll for l in output_results for ll in l]
        summary_lists = self.dm.test_data_df["summary_list"]
        assert len(all_generations) == len(summary_lists)
        all_outputs = []
        for pred, ref in zip(all_generations, summary_lists):
            all_outputs.append({"pred": pred,
                                "ref": ref})
        with open(os.path.join(self.args.log_output_subdir, "test_generations.json"), "w") as fout:
            json.dump(all_outputs, fout)

    @staticmethod
    def add_model_specific_args(parser):
        # **************** Parameters that we will NOT change during this tutorial **************** #
        parser.add_argument("--seed", type=int, default=1234, help="Seed")
        parser.add_argument("--lr", type=float, default=0.00003, help="Maximum learning rate")
        parser.add_argument("--warmup", type=int, default=1000, help="Number of warmup steps")
        parser.add_argument("--epochs", type=int, default=1, help="Number of epochs")
        parser.add_argument("--num_workers", type=int, default=4, help="Number of data loader workers")
        parser.add_argument("--max_output_len", type=int, default=256, help="maximum num of wordpieces in the summary")
        parser.add_argument("--num_beams", type=int, default=1, help="Beam width size")
        parser.add_argument("--output_dir", type=str, default='./saved_models/test', help="Location of output dir")
        parser.add_argument("--log_output_basedir", type=str, default='./output', help="Location of log dir")
        parser.add_argument("--val_every", default=0.33, type=float, help='Validation every')

        # **************** Parameters that we will change during this tutorial **************** #
        parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
        parser.add_argument("--grad_accum", type=int, default=1, help="number of gradient accumulation steps")
        parser.add_argument("--fp16", action='store_true', help="Use fp16 ")
        parser.add_argument('--grad_ckpt', action='store_true', help='Enable gradient checkpointing to save memory')
        parser.add_argument("--attention_window", type=int, default=1024, help="Attention window")

        # **************** For Multi-task purpose **************** #
        parser.add_argument("--dropout_rate", type=float, default=0.1, help="Drop out rate for classifier")
        parser.add_argument("--num_classes", type=int, default=2, help="Number of classes")
        parser.add_argument("--classifier_weight", type=float, default=0.5, help="Loss weight for classifier")
        parser.add_argument("--summarizer_weight", type=float, default=0, help="Loss weight for Summarizer")
        
        return parser


if __name__ == "__main__":
    # Setup command line args
    main_arg_parser = argparse.ArgumentParser(description="QA summarization")
    main_arg_parser.add_argument("--model_name", default="allenai/led-base-16384")
    parser = Summarizer.add_model_specific_args(main_arg_parser)
    parser = QASumPreSplitDataModule.add_dataset_specific_args(parser)
    args = parser.parse_args()

    # Create a tagname
    model_pathname = args.model_name.replace("/", "--")
    tagname = f"{model_pathname}__bs={args.batch_size}__ep={args.epochs}__nb={args.num_beams}"
    log_output_dir = os.path.join(args.log_output_basedir, tagname)
    if not os.path.exists(log_output_dir):
        os.makedirs(log_output_dir)

    # Check the largest run ID to allocate a new run ID
    subdirs = [int(x) for x in filter(lambda x: x.isdigit(), os.listdir(log_output_dir))]
    if len(subdirs) == 0:
        runid = 1
    else:
        runid = max(subdirs) + 1
    args.log_output_subdir = os.path.join(log_output_dir, str(runid))
    assert not os.path.exists(args.log_output_subdir), "WARNING: The log output directory already exists."
    os.makedirs(args.log_output_subdir)

    # Save command arguments into a JSON file
    with open(os.path.join(args.log_output_subdir, "params.json"), "w") as fout:
        json.dump(vars(args), fout)

    print("Creating summarizer")

    # Init a PL module
    set_seed(args.seed)
    summarizer = Summarizer(args)
    print("Summarizer done")

    """
    Metric options:
      ['val_rouge-2_f', 'val_rouge-2_p', 'val_rouge-2_r',
       'val_rouge-1_f', 'val_rouge-1_p', 'val_rouge-1_r',
       'val_rouge-l_f', 'val_rouge-l_p', 'val_rouge-l_r',
       'val_bertscore_p', 'val_bertscore_r', 'val_bertscore_f']
    """
    checkpoint_callback = ModelCheckpoint(monitor="val_rouge-1_f",
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
                         precision=16 if args.fp16 else 32,
                         accumulate_grad_batches=args.grad_accum,
                         callbacks=[checkpoint_callback],
                         val_check_interval=args.val_every
                         )
    print("Trainer done")

    # Start training
    trainer.fit(summarizer)

    # Start testing
    result = trainer.test()

    with open(os.path.join(args.log_output_subdir, "scores.json"), "w") as fout:
        json.dump(result, fout)

    print("All done!")
    