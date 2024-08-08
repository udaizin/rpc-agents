import json
import os
import random

# TODO: 後々.envファイルから取得するようにする。
TARGET_INTERLOCUTOR_ID = 'GN'
OUTPUT_DIR = './JudgeLLM/data/sample20'

def extract_data(path):
    with open(path, 'r') as f:
        dialogues_json = json.load(f)
    
    random_indices_20 = random.sample(range(len(dialogues_json)), 20)

    extract_dialogue_list = []
    for i, dialogue in enumerate(dialogues_json):
        if i in random_indices_20:
            extract_dialogue_list.append(dialogue)

    return extract_dialogue_list

if __name__ == '__main__':
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    dialogue_list = extract_data(f'./RealPersonaChat/data/gen_inner_monologue/{TARGET_INTERLOCUTOR_ID}_inner_monologue.json')
    with open(f'{OUTPUT_DIR}/{TARGET_INTERLOCUTOR_ID}_extracted_data.json', 'w') as f:
        json.dump(dialogue_list, f, indent=2, ensure_ascii=False)