import json
import argparse

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from transformers.trainer_utils import set_seed
from trl import DataCollatorForCompletionOnlyLM
from datasets import Dataset

# 乱数シードの固定
set_seed(42)

TARGET_INTERLOCUTOR_ID = 'AD'
TRAIN_DATA_PATH = f'./RealPersonaChat/data/train_data/{TARGET_INTERLOCUTOR_ID}_inner_monologue_train.json'
TEST_DATA_PATH = f'./RealPersonaChat/data/test_data/{TARGET_INTERLOCUTOR_ID}_inner_monologue_test.json'
MODEL_NAME = 'tokyotech-llm/Llama-3-Swallow-8B-Instruct-v0.1'
LEARNING_RATE = 2e-5
LEARNING_RATE_STR = "{:.0e}".format(LEARNING_RATE)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# データセットの作成
def make_tokenized_dataset(tokenizer, dialogues, without_inner_monologue):
    dataset = []
    if without_inner_monologue:
        for dialogue in dialogues:
            messages = []
            for utterance in dialogue['utterances']:
                if utterance['role'] == TARGET_INTERLOCUTOR_ID:
                    messages.append({'role': 'assistant', 'content': utterance['content']})
                else:
                    messages.append({'role': 'user', 'content': utterance['content']})
            dataset.append(messages)
    else:
        for dialogue in dialogues:
            messages = []
            for i in range(len(dialogue['inner_monologue_utterances'])):
                if dialogue['inner_monologue_utterances'][i]['role'] == TARGET_INTERLOCUTOR_ID and dialogue['inner_monologue_utterances'][i]['action'] == '(thinking)':
                    thinking = dialogue['inner_monologue_utterances'][i]['content']
                    speaking = dialogue['inner_monologue_utterances'][i+1]['content']
                    messages.append({'role': 'assistant', 'content': f'<thinking>{thinking}</thinking>{speaking}'})
                elif dialogue['inner_monologue_utterances'][i]['role'] == TARGET_INTERLOCUTOR_ID and dialogue['inner_monologue_utterances'][i]['action'] == '(speaking)':
                    continue
                else:
                    messages.append({'role': 'user', 'content': dialogue['inner_monologue_utterances'][i]['content']})
            dataset.append(messages)

    # chat_template_list = [
    #     tokenizer.apply_chat_template(dialogue, tokenize=False) for dialogue in dataset
    # ]
    # tokenized_list = [tokenizer(dialogue, return_length=True) for dialogue in chat_template_list]
    # tokenized_dataset = Dataset.from_list(tokenized_list)

    tokenized_dataset = [
        tokenizer.apply_chat_template(dialogue) for dialogue in dataset
    ]
    return tokenized_dataset
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--without_inner_monologue', action='store_true')
    args = parser.parse_args()

    if args.without_inner_monologue:
        output_dir = f'./models/Swallow3-8B-{TARGET_INTERLOCUTOR_ID}-v4-{LEARNING_RATE_STR}-change-pad-wo-im'
    else:
        output_dir = f'./models/Swallow3-8B-{TARGET_INTERLOCUTOR_ID}-v4-{LEARNING_RATE_STR}-change-pad'

    with open(TRAIN_DATA_PATH, 'r', encoding='utf-8') as f:
        train_dialogues = json.load(f)
    with open(TEST_DATA_PATH, 'r', encoding='utf-8') as f:
        test_dialogues = json.load(f)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, padding='right')
    # TODO: ここでpad_tokenを変更
    tokenizer.add_special_tokens({'pad_token': '<|reserved_special_token_0|>'})
    tokenized_train_dataset = make_tokenized_dataset(tokenizer, train_dialogues, args.without_inner_monologue)
    tokenized_test_dataset = make_tokenized_dataset(tokenizer, test_dialogues, args.without_inner_monologue)

    # モデルの読み込み
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        use_cache=False, # gradient checkpointingを有効にするために必要
    )
    # TODO: ここでpad_token_idを変更
    model.to(DEVICE)
    model.print_trainable_parameters() # 学習可能なパラメータの数を表示

    training_args = TrainingArguments(
        output_dir=output_dir,
        bf16=True,
        num_train_epochs=10,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=4,
        gradient_checkpointing=True,
        learning_rate=LEARNING_RATE,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        logging_steps=1,
        save_strategy='no',
        eval_strategy='epoch',
        # group_by_length=True,
        # length_column_name='length',
        report_to='wandb',
    )

    data_collator = DataCollatorForCompletionOnlyLM(
        tokenizer=tokenizer,
        instruction_template='<|start_header_id|>user<|end_header_id|>\n\n',
        response_template='<|start_header_id|>assistant<|end_header_id|>\n\n',
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_test_dataset,
        tokenizer=tokenizer,
    )

    trainer.train()

    # モデルの保存
    trainer.save_model()