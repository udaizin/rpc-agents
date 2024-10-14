import json
import argparse
from typing import Dict, Optional, Sequence

import torch
import numpy as np
from torch.utils.data import Dataset
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments, Trainer
from transformers.trainer_utils import set_seed
from peft import LoraConfig, TaskType, get_peft_model

# 乱数シードの固定
set_seed(42)

TARGET_INTERLOCUTOR_ID = 'AD'
TRAIN_DATA_PATH = f'./RealPersonaChat/data/train_data/chat_template_{TARGET_INTERLOCUTOR_ID}.jsonl'
MODEL_NAME = 'tokyotech-llm/Llama-3-Swallow-8B-Instruct-v0.1'
QUANTIZATION_CONFIG = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)
LEARNING_RATE = 2e-5
LEARNING_RATE_STR = "{:.0e}".format(LEARNING_RATE)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
IGNORE_TOKEN_ID = transformers.trainer_pt_utils.LabelSmoother.ignore_index

def alpaca_tokenize_fn(strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="max_length",
            max_length=min(tokenizer.model_max_length, 2048),
            truncation=True,
            # TODO: apply_chat_templateの引数tokenize=Falseに対応するためにadd_special_tokens=Falseを追加
            add_special_tokens=False,
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
    print(input_ids.shape, labels.shape)
    print(np.mean(examples_tokenized["input_ids_lens"]), np.max(examples_tokenized["input_ids_lens"]))
    # rank0_print(input_ids[0].tolist())
    # rank0_print(labels[0].tolist())
    return dict(
        input_ids=input_ids,
        labels=labels,
        attention_mask=attention_mask,
    )

class ProcessedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, raw_data, tokenizer: transformers.PreTrainedTokenizer):
        super(ProcessedDataset, self).__init__()

        print("Formatting inputs...")
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
    tokenizer: transformers.PreTrainedTokenizer
) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    dataset_cls = ProcessedDataset
    # Split train/test
    raw_data =[]
    with open(TRAIN_DATA_PATH, 'r', encoding='utf-8') as fp:
        for line in fp:
            if line:
                raw_data.append(json.loads(line))
    
    train_raw_data = raw_data

    print(f"#train {len(train_raw_data)}")

    train_dataset = dataset_cls(train_raw_data, tokenizer=tokenizer)

    return dict(train_dataset=train_dataset)

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--without_inner_monologue', action='store_true')
    args = parser.parse_args()

    if args.without_inner_monologue:
        output_dir = f'./models/Swallow3-8B-{TARGET_INTERLOCUTOR_ID}-v4-divide-QLoRA-{LEARNING_RATE_STR}-wo-im'
    else:
        output_dir = f'./models/Swallow3-8B-{TARGET_INTERLOCUTOR_ID}-v4-divide-QLoRA-{LEARNING_RATE_STR}'

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    data_module = make_supervised_data_module(tokenizer)

    # モデルの読み込み
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, 
        quantization_config=QUANTIZATION_CONFIG,
        torch_dtype=torch.bfloat16,
        use_cache=False, # gradient checkpointingを有効にするために必要
    )
    # LoRAの設定
    peft_config = LoraConfig(
        r=64,  # 差分行列のランク
        lora_alpha=64,  # LoRA層の出力のスケールを調整するハイパーパラメータ
        lora_dropout=0.05,  # LoRA層に適用するドロップアウト
        task_type=TaskType.CAUSAL_LM,  # LLMが解くタスクのタイプを指定
        # LoRAで学習するモジュール
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )
    model.enable_input_require_grads()
    model = get_peft_model(model, peft_config)
    model.to(DEVICE)
    model.print_trainable_parameters() # 学習可能なパラメータの数を表示

    training_args = TrainingArguments(
        output_dir=output_dir,
        bf16=True,
        num_train_epochs=10,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        gradient_checkpointing=True,
        learning_rate=LEARNING_RATE,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        logging_steps=1,
        save_strategy='no',
        eval_strategy='no',
        report_to='wandb',
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=data_module['train_dataset'],
        tokenizer=tokenizer,
    )

    trainer.train()

    # モデルの保存
    trainer.save_model()