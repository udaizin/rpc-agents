import os
import json

from dotenv import load_dotenv, find_dotenv
from transformers import AutoTokenizer



_ = load_dotenv(find_dotenv(), override=True)  
TARGET_INTERLOCUTOR_IDS = os.getenv("TARGET_INTERLOCUTOR_IDS").split(',')

# TODO: モデルによってEOT_IDが異なるため、変更する
tokenizer = AutoTokenizer.from_pretrained('tokyotech-llm/Llama-3-Swallow-8B-Instruct-v0.1')
EOT_ID = '<|eot_id|>'

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
      {
        "role": "CA",
        "action": "(speaking)",
        "content": "こちらは今日も暑くなりはじめてきましたが、まいさんのところのお天気はどんなでしょう？"
      },
      {
        "role": "CP",
        "action": "(speaking)",
        "content": "暑くなりつつあります。今日も晴天です！"
      },
      {
        "role": "CA",
        "action": "(speaking)",
        "content": "いい日になりそうですね！まだまだ夏日だ"
      },
      {
        "role": "CP",
        "action": "(speaking)",
        "content": "ですね。蝉が煩いです。"
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
        "content": "初対面の人との会話は少し緊張するなぁ。"
      },
      {
        "role": "CP",
        "action": "(speaking)",
        "content": "こんにちは。<CP>です。よろしくお願いします。"
      },
      {
        "role": "CA",
        "action": "(speaking)",
        "content": "こちらは今日も暑くなりはじめてきましたが、まいさんのところのお天気はどんなでしょう？"
      },
      {
        "role": "CP",
        "action": "(thinking)",
        "content": "福岡も最近暑くなってきたし、今日も晴れだな。"
      },
      {
        "role": "CP",
        "action": "(speaking)",
        "content": "暑くなりつつあります。今日も晴天です！"
      },
      {
        "role": "CA",
        "action": "(speaking)",
        "content": "いい日になりそうですね！まだまだ夏日だ"
      },
      {
        "role": "CP",
        "action": "(thinking)",
        "content": "夏の日差しが好きだけど、蝉の声はちょっと煩いかな。"
      },
      {
        "role": "CP",
        "action": "(speaking)",
        "content": "ですね。蝉が煩いです。"
      },
      […]
    ]
  },
  […]
]

出力例:
{
  "prompt": "<|begin_of_text|><|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nこんにちは。はじめまして。<CA>と申します<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
  "output": "<thinking>初対面の人との会話は少し緊張するなぁ。</thinking>こんにちは。<CP>です。よろしくお願いします。<|eot_id|>"
}
{
  "prompt": "<|begin_of_text|><|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nこんにちは。<CP>です。よろしくお願いします。<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n<thinking>初対面の人との会話は少し緊張するなぁ。</thinking>こんにちは。<CP>です。よろしくお願いします。<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nこちらは今日も暑くなりはじめてきましたが、まいさんのところのお天気はどんなでしょう？<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
  "output": "<thinking>福岡も最近暑くなってきたし、今日も晴れだな。</thinking>暑くなりつつあります。今日も晴天です！<|eot_id|>"
}
{
  "prompt": "<|begin_of_text|><|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nこんにちは。<CP>です。よろしくお願いします。<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n<thinking>初対面の人との会話は少し緊張するなぁ。</thinking>こんにちは。<CP>です。よろしくお願いします。<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nこちらは今日も暑くなりはじめてきましたが、まいさんのところのお天気はどんなでしょう？<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n<thinking>福岡も最近暑くなってきたし、今日も晴れだな。</thinking>暑くなりつつあります。今日も晴天です！<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nいい日になりそうですね！まだまだ夏日だ<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
  "output": "<thinking>夏の日差しが好きだけど、蝉の声はちょっと煩いかな。</thinking>ですね。蝉が煩いです。<|eot_id|>"
}
[...]
'''

def convert_to_chat_template(target_interlocutor_id: str):
    # json形式データの読み込み
    with open(f'./RealPersonaChat/data/gen_inner_monologue/{target_interlocutor_id}_inner_monologue.json', 'r', encoding='utf-8') as f:
        dialogues = json.load(f)
    
    with open(f'./RealPersonaChat/data/train_data/chat_template_{target_interlocutor_id}.jsonl', 'w', encoding='utf-8') as f:
        for dialogue in dialogues:
            messages = []
            for i in range(len(dialogue['inner_monologue_utterances'])):
                # はじめの発話がtarget_interlocutor_idのものであるときはそれを追加
                if i==0 and dialogue['inner_monologue_utterances'][i]['role'] == target_interlocutor_id:
                    thinking = dialogue['inner_monologue_utterances'][i]['content']
                    speaking = dialogue['inner_monologue_utterances'][i+1]['content']
                    messages.append({'role': 'assistant', 'content': f'<thinking>{thinking}</thinking>{speaking}'})
                    continue
                # target_interlocutor_idの(speaking)のときはスキップ
                elif dialogue['inner_monologue_utterances'][i]['role'] == target_interlocutor_id and dialogue['inner_monologue_utterances'][i]['action'] == '(speaking)':
                    continue
                # target_interlocutor_id出ないものはuserとして追加
                elif dialogue['inner_monologue_utterances'][i]['role'] != target_interlocutor_id:
                    messages.append({'role': 'user', 'content': dialogue['inner_monologue_utterances'][i]['content']})
                # target_interlocutor_idのときはpromptとoutputを作成後、messagesに追加
                else:
                    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                    thinking = dialogue['inner_monologue_utterances'][i]['content']
                    speaking = dialogue['inner_monologue_utterances'][i+1]['content']
                    output = f'<thinking>{thinking}</thinking>{speaking}{EOT_ID}'
                    f.write(json.dumps({"prompt": prompt, "output": output}, ensure_ascii=False))
                    f.write('\n')
                    messages.append({'role': 'assistant', 'content': f'<thinking>{thinking}</thinking>{speaking}'})
    

if __name__ == '__main__':
    for target_interlocutor_id in TARGET_INTERLOCUTOR_IDS:
        convert_to_chat_template(target_interlocutor_id)