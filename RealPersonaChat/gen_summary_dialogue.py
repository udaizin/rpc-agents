from api_call_util import decoder_for_openai
from io_utils import read_json, read_gen_data, read_jsonl, load_seed_data_train
import os
import glob
from threading import Thread, Lock
import json
from functools import partial
import re
from tqdm import tqdm
from dotenv import load_dotenv, find_dotenv
from openai import OpenAI
import sys
from config import args

_ = load_dotenv(find_dotenv())
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TARGET_INTERLOCUTOR_IDS = os.getenv("TARGET_INTERLOCUTOR_IDS").split(',')
client = OpenAI(api_key=OPENAI_API_KEY)

for target_interlocutor_id in TARGET_INTERLOCUTOR_IDS:
    # /work/rpc-agents/RealPersonaChat/data/CP_dialogues.jsonを読み込み
    with open(f'./RealPersonaChat/data/change_format/{target_interlocutor_id}_dialogues.json', 'r') as f:
        dialogues = json.load(f)

    utterances_list = []
    for i in range(10):
        utterances = '\n'.join(dialogues[i]['utterances'])
        utterances_list.append(utterances)

    utterances_summary_pair_list = []
    for utterances in utterances_list:
        prompt = f'''対話履歴:
        {utterances}

        上記の対話履歴を完結にまとめて下さい。'''
        completion = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "あなたは要約のプロフェッショナルです。"},
                {"role": "user", "content": prompt}
            ],
        )

        utterances_summary_pair = {
            "utterances": utterances,
            "summary": completion.choices[0].message.content
        }
        utterances_summary_pair_list.append(utterances_summary_pair)

    # json形式で保存
    with open(f'./RealPersonaChat/data/gen_summary/{target_interlocutor_id}_utterances_summary_pair.json', 'w') as f:
        json.dump(utterances_summary_pair_list, f, ensure_ascii=False, indent=2)


        
    

