import os
import json
from tqdm.contrib import tzip
from dotenv import load_dotenv, find_dotenv
from openai import OpenAI


_ = load_dotenv(find_dotenv())
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TARGET_INTERLOCUTOR_IDS = os.getenv("TARGET_INTERLOCUTOR_IDS").split(',')
client = OpenAI(api_key=OPENAI_API_KEY)

for target_interlocutor_id in TARGET_INTERLOCUTOR_IDS:
    with open(f'./RealPersonaChat/data/change_format/{target_interlocutor_id}_dialogues.json', 'r') as f:
        dialogues = json.load(f)

    utterances_list = ['\n'.join(dialogue['utterances']) for dialogue in dialogues]
    dialogue_id_list = [dialogue['dialogue_id'] for dialogue in dialogues]

    dialogues_summary_list = []
    for utterances, dialogue_id in tzip(utterances_list, dialogue_id_list):
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

        dialogue_summary_dict = {
            "dialogue_id": dialogue_id,
            "utterances": utterances,
            "summary": completion.choices[0].message.content
        }
        dialogues_summary_list.append(dialogue_summary_dict)

    # json形式で保存
    with open(f'./RealPersonaChat/data/gen_summary/{target_interlocutor_id}_summary.json', 'w') as f:
        json.dump(dialogues_summary_list, f, ensure_ascii=False, indent=2)


        
    

