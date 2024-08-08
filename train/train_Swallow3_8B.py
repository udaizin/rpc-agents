from typing import Dict, Sequence, Optional
import json

import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from transformers.trainer_pt_utils import LabelSmoother
from torch.utils.data import Dataset
import torch
import wandb


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TARGET_INTERLOCUTOR_ID = 'FL'
IGNORE_TOKEN_ID = LabelSmoother.ignore_index

# モデルの設定
model_name = 'tokyotech-llm/Llama-3-Swallow-8B-Instruct-v0.1'
tokenizer = AutoTokenizer.from_pretrained(model_name, model_max_length=2048)
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

# データセットの準備
def alpaca_tokenize_fn(strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        for text in strings
    ]
    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )

def alpaca_preprocess(
    sources: Sequence[str],
    targets: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    """Preprocess the data by tokenizing."""
    examples = [s + t for s, t in zip(sources, targets)]
    examples_tokenized, sources_tokenized = [alpaca_tokenize_fn(strings, tokenizer) for strings in (examples, sources)]
    input_ids = examples_tokenized["input_ids"]
    input_ids = torch.stack(input_ids, dim=0)
    labels = input_ids.clone()
    attention_mask=input_ids.ne(tokenizer.pad_token_id)
    for label, source_len, total_len in zip(labels, sources_tokenized["input_ids_lens"], examples_tokenized["input_ids_lens"]):
        label[:source_len] = IGNORE_TOKEN_ID
        label[total_len+1:] = IGNORE_TOKEN_ID
    return dict(
        input_ids=input_ids,
        labels=labels,
        attention_mask=attention_mask,
    )

class ProcessedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, raw_data, tokenizer: transformers.PreTrainedTokenizer):
        super(ProcessedDataset, self).__init__()

        sources = [ex['prompt'] for ex in raw_data]
        targets = [ex['output'] for ex in raw_data]
        data_dict = alpaca_preprocess(sources, targets, tokenizer)

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]
        self.attention_mask = data_dict["attention_mask"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(
            input_ids=self.input_ids[i],
            labels=self.labels[i],
            attention_mask=self.attention_mask[i],
        )
    
def make_supervised_data_module(
    tokenizer: transformers.PreTrainedTokenizer, data_path: str
) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    dataset_cls = ProcessedDataset
    # jsonファイルから訓練データの読み込み
    raw_data =[]
    with open(data_path, 'r', encoding='utf-8') as fp:
        for line in fp:
            if line:
                raw_data.append(json.loads(line))
    train_raw_data = raw_data
    train_dataset = dataset_cls(train_raw_data, tokenizer=tokenizer)

    return dict(train_dataset=train_dataset)

# 訓練データセットの読み込み
train_data_module = make_supervised_data_module(tokenizer, f'./RealPersonaChat/data/train_data/train_data_{TARGET_INTERLOCUTOR_ID}.jsonl')

# トレーニングの設定
train_args = TrainingArguments(
    output_dir=f'./models/Swallow3-8B-{TARGET_INTERLOCUTOR_ID}-v2',
    bf16=True,
    per_device_train_batch_size=2,
    save_strategy='epoch',
    save_total_limit=3,
    learning_rate=5e-6,
    num_train_epochs=10,
    report_to='wandb'
)

# トレーナーの設定
trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
    args=train_args,
    train_dataset=train_data_module['train_dataset'],
)

if __name__ == "__main__":
    trainer.train()
    trainer.save_model()