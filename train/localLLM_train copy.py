import os
import numpy as np
import math
import pandas as pd
import random
import re

from tqdm import tqdm
import itertools
import collections

import time
import datetime
dt = datetime.datetime.today()  # ローカルな現在の日付と時刻を取得
if dt.hour+9 < 24:
    date = f'{dt.year}-'+f'{dt.month}-'+f'{dt.day}-'+f'{dt.hour+9}-'+f'{dt.minute}'
else:
    date = f'{dt.year}-'+f'{dt.month}-'+f'{dt.day+1}-'+f'{dt.hour-15}-'+f'{dt.minute}'

import datasets
import copy

import wandb
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM, BitsAndBytesConfig
from peft import get_peft_model, LoraConfig, TaskType

from sklearn.metrics import f1_score
import torch.optim as optim
from transformers.modeling_outputs import ModelOutput
from transformers import AutoTokenizer, AutoModel

# 乱数シード設定
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
set_seed(100)

DEVICE = torch.device("cuda:0" if torch.cuda.is_available else "cpu")
DATAPATH = "/work/QG_nursing/QA_dataset_Swallow/"
TRAINING_MODEL_NAME = "tokyotech-llm/Swallow-70b-instruct-hf"
LORA = True
QUANTIZATION = True

BATCH_SIZE = 1
LEARNING_RATE = 1e-4
EPOCH = 10
SEQUENCE = 512
PROJECT_NAME = re.sub(r"^.*\/", "", TRAINING_MODEL_NAME)
if QUANTIZATION:
    PROJECT_NAME += "_Q"
if LORA:
    PROJECT_NAME += "_LoRA"
SAVE_PATH = "/work/QG_nursing/save_localmodel/" + PROJECT_NAME

class CausalLM_Dataset(Dataset):
    def __init__(self, tokenizer, datapath, datatype, mode="training"):
        self.datapath = datapath
        self.datatype = datatype
        self.mode = mode
        self.inputs = []
        self.targets = []
        self.tokenizer = tokenizer
        self._build()

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        source_ids = self.inputs[index]["input_ids"].squeeze()
        target_ids = self.targets[index]["input_ids"].squeeze()
        source_mask = self.inputs[index]["attention_mask"].squeeze()
        target_mask = self.targets[index]["attention_mask"].squeeze()

        return {"input_ids": source_ids, "attention_mask": source_mask,
                "labels": target_ids, "decoder_attention_mask": target_mask}

    def _build(self):
        srcpath, tgtpath = self.datapath + self.datatype + "_src", self.datapath + self.datatype + "_tgt"
        with open(srcpath, "r", encoding="utf-8") as f:
            src_list = f.read().split("\n")
            src_list.pop(-1)
        with open(tgtpath, "r", encoding="utf-8") as f:
            tgt_list = f.read().split("\n")
            tgt_list.pop(-1)

        for i, line in tqdm(enumerate(src_list)):

            src, tgt = src_list[i].replace("\m", "\n"), tgt_list[i].replace("\m", "\n")+self.tokenizer.eos_token #dataset は改行コードを変換しているのでもとに戻す causalLMなのでtargetにはEOSを追加
            source_tokenized = self.tokenizer(src, add_special_tokens=False, padding="longest", max_length=512, return_tensors="pt", return_length=True,)
            source_len = source_tokenized["length"][0]
            if self.mode == "training":
                source_tokenized = self.tokenizer(tgt, add_special_tokens=False, padding="longest", max_length=512, return_tensors="pt")
            targets_tokenized = copy.deepcopy(source_tokenized)
            targets_tokenized["input_ids"][0][:source_len] = -100

            self.inputs.append(source_tokenized)
            self.targets.append(targets_tokenized)

class CausalLM_Trainer():
    
    def train(model_name, batch_size=1, lr=1e-5, epoch=1, lora=False, quantization=False, project_name=None, save_path=None):

        tokenizer = AutoTokenizer.from_pretrained(TRAINING_MODEL_NAME)
        tokenizer.add_special_tokens({'pad_token': '<|padding|>'})

        train_dataset = CausalLM_Dataset(tokenizer, DATAPATH, datatype="train")
        val_dataset = CausalLM_Dataset(tokenizer, DATAPATH, datatype="val")
        test_dataset = CausalLM_Dataset(tokenizer, DATAPATH, datatype="test")

        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, drop_last=True, shuffle=True, num_workers=4,)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, num_workers=4,)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, num_workers=4,)

        device = 'cuda:0'
        wandb.init(project=project_name, name=date)

        ### Quantization
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,  # 量子化の有効化
            bnb_4bit_quant_type="nf4",  # 量子化種別 (fp4 or nf4)
            bnb_4bit_compute_dtype=torch.bfloat16,  # 量子化のdtype (float16 or bfloat16)
            bnb_4bit_use_double_quant=True,  # 二重量子化の有効化
        )

        model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            quantization_config=bnb_config if quantization else None,  # 量子化パラメータ
        )

        ### LoRA
        if lora:
            lora_config = LoraConfig(
                r=8,
                lora_alpha=32,
                target_modules=[
                    "q_proj",
                    # "k_proj",
                    "v_proj",
                    # "o_proj",
                    # "gate_proj",
                    # "up_proj",
                    # "down_proj",
                    # "lm_head",
                ],
                bias="none",
                fan_in_fan_out=False,
                lora_dropout=0.05,
                task_type="CAUSAL_LM",
            )
            model = get_peft_model(model, lora_config)

        loss_fct = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
        # optimizer = torch.optim.Adam(model.parameters(), lr=lr)
                
        model.to(device)

        # 学習＆検証ループ
        for num_epoch in range(epoch):

            train_losses = []
            model.train()
            for batch in tqdm(train_dataloader):
                
                ### origin
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)
                optimizer.zero_grad()
                output = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss, _ = output.loss, output.logits
                loss.backward()
                optimizer.step()

                wandb.log({"train_step_loss": loss.item()})
                train_losses.append(loss.item())

            print(f'Epoch: {num_epoch+1}\tloss: {np.array(train_losses).mean()}')
            wandb.log({"train_loss": np.array(train_losses).mean()})

            val_losses = []
            model.eval()
            for batch in tqdm(val_dataloader):
                with torch.no_grad():

                    input_ids = batch["input_ids"].to(device)
                    attention_mask = batch["attention_mask"].to(device)
                    labels = batch["labels"].to(device)
                    output = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                    loss, _ = output.loss, output.logits
                    
                    wandb.log({"val_step_loss": loss.item()})
                    val_losses.append(loss.item())

            print(f'Epoch: {num_epoch+1}\tloss: {np.array(val_losses).mean()}')
            wandb.log({"val_loss": np.array(val_losses).mean()})

        wandb.finish()
        model.save_pretrained(
            save_path + "_" + date, 
            safe_serialization = False, #.binで保存する，.safetensorsだと何故かうまくいかない
            )


    def generate(load_path, datatype, quantization=False):

        tokenizer = AutoTokenizer.from_pretrained(TRAINING_MODEL_NAME)
        tokenizer.add_special_tokens({'pad_token': '<|padding|>'})

        test_dataset = CausalLM_Dataset(tokenizer, DATAPATH, datatype=datatype, mode="inference")
        test_dataloader = DataLoader(test_dataset, batch_size=1, num_workers=4,)

        ### Quantization
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,  # 量子化の有効化
            bnb_4bit_quant_type="nf4",  # 量子化種別 (fp4 or nf4)
            bnb_4bit_compute_dtype=torch.bfloat16,  # 量子化のdtype (float16 or bfloat16)
            bnb_4bit_use_double_quant=True,  # 二重量子化の有効化
        )

        model = AutoModelForCausalLM.from_pretrained(
            load_path, 
            trust_remote_code=True,        
            torch_dtype=torch.bfloat16,
            device_map = "auto",
            quantization_config=bnb_config if quantization else None,  # 量子化パラメータ
        )

        set_seed(100)
        test_output = []
        model.eval()
        for batch in tqdm(test_dataloader):
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            with torch.no_grad():
                output_ids = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=512, #最大出力トークン
                    temperature=0.99,
                    top_p=0.95,
                    do_sample=True,
                    output_scores=True,
                    return_dict_in_generate=True,
                )
            output = tokenizer.decode(output_ids[0][0], skip_special_tokens=True)
            print(output)
            output = tokenizer.decode(output_ids[0][0][input_ids[0].shape[0]:], skip_special_tokens=True)
            test_output.append(output)

        return test_output

if __name__ == "__main__":
    CausalLM_Trainer.train(
        model_name=TRAINING_MODEL_NAME, 
        batch_size=BATCH_SIZE, 
        lr=LEARNING_RATE, 
        epoch=EPOCH, 
        lora=LORA,
        quantization=QUANTIZATION,
        project_name=PROJECT_NAME, 
        save_path=SAVE_PATH,
        )

    load=TRAINING_MODEL_NAME
    datatype="test"
    test_output = CausalLM_Trainer.generate(
        load_path=load,
        datatype=datatype,
        quantization=True,
        )
    print(test_output)
