import os
import json

from dotenv import load_dotenv, find_dotenv



_ = load_dotenv(find_dotenv(), override=True)  
TARGET_INTERLOCUTOR_IDS = os.getenv("TARGET_INTERLOCUTOR_IDS").split(',')

'''
json形式の対話データを読み込み、promptとoutputをjsonl形式に変換する関数
入力例:
[
  {
    "dialogue_id": 941,
    "utterances": [
      {
        "role": "CA",
        "action": "(speaking)",
        "content": "こんにちは。はじめまして。<CA>と申します"
      },
      {
        "role": "CP",
        "action": "(speaking)",
        "content": "こんにちは。<CP>です。よろしくお願いします。"
      },
      […]
    ],
    "inner_monologue_utterances": [
      {
        "role": "CA",
        "action": "(speaking)",
        "content": "こんにちは。はじめまして。<CA>と申します"
      },
      {
        "role": "CP",
        "action": "(thinking)",
        "content": "初めての人と話すのは少し緊張するなあ。でも、楽しみでもある！"
      },
      {
        "role": "CP",
        "action": "(speaking)",
        "content": "こんにちは。<CP>です。よろしくお願いします。"
      },
      […]
    ]
  },
  […]
]

出力例:
{"prompt": あなたはCPです。以下のルールに基づいて、対話履歴にCPの内心を書き加えてください。\n\n1. もとの対話履歴の内容を絶対に書き換えないでください。\n2. 「CP (speaking): ~ 」という行の前に「CP (thinking): ~ 」という行を挿入して考えや感情を表現してください。\n\n対話履歴:\nCA (speaking): こんにちは。はじめまして。<CA>と申します\nCP (speaking): こんにちは。<CP>です。よろしくお願いします。\n[...]\n\n内心描写を付与した対話履歴:", "output": "CA (speaking): こんにちは。はじめまして。<CA>と申します\nCP (thinking): 初めての人と話すのは少し緊張するなあ。でも、楽しみでもある！\nCP (speaking): こんにちは。<CP>です。よろしくお願いします。\n[...]"}
[...]
'''
def convert_inner_monologue_json_into_train_format(target_interlocutor_id: str, common_setting: str):
    # json形式データの読み込み
    with open(f'./RealPersonaChat/data/gen_inner_monologue/{target_interlocutor_id}_inner_monologue.json', 'r', encoding='utf-8') as f:
        dialogues = json.load(f)
    
    with open(f'./RealPersonaChat/data/train_data/train_data_{target_interlocutor_id}.jsonl', 'w', encoding='utf-8') as f:
        for dialogue in dialogues:
            dialogue_id = dialogue['dialogue_id']
            utterances = dialogue['utterances']
            inner_monologue_utterances = dialogue['inner_monologue_utterances']
            dialogue_text = '\n'.join([f"{utterance['role']} {utterance['action']}: {utterance['content']}" for utterance in utterances])
            inner_monologue_text = '\n'.join([f"{utterance['role']} {utterance['action']}: {utterance['content']}" for utterance in inner_monologue_utterances])
            f.write(json.dumps({"prompt": f"{common_setting}\n\n対話履歴:\n{dialogue_text}\n\n内心描写を付与した対話履歴:", "output": f"{inner_monologue_text}"}, ensure_ascii=False))
            f.write('\n')

if __name__ == '__main__':
    for target_interlocutor_id in TARGET_INTERLOCUTOR_IDS:
        common_setting = '\n'.join([
            f'あなたは{target_interlocutor_id}です。以下のルールに基づいて、対話履歴に{target_interlocutor_id}の内心を書き加えてください。',
            '',
            '1. もとの対話履歴の内容を絶対に書き換えないでください。',
            f'2. 「{target_interlocutor_id} (speaking): ~ 」という行の前に「{target_interlocutor_id} (thinking): ~ 」という行を挿入して考えや感情を表現してください。'
        ])
        convert_inner_monologue_json_into_train_format(target_interlocutor_id, common_setting)