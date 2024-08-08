import json
import os
import random

# TODO: 後々.envファイルから取得するようにする。
TARGET_INTERLOCUTOR_ID = 'DV'
OUTPUT_DIR = './JudgeLLM/data/sample20'

def convert_raw_utterances_into_json_format(utterances: list) -> list:
    utterances_json_list = []
    for utterance in utterances:
        role, action_content = utterance.split(' ', 1)
        action, content = action_content.split(': ', 1)
        utterance_dict = {
            "role": role,
            "action": action,
            "content": content
        }
        utterances_json_list.append(utterance_dict)
    return utterances_json_list

def extract_data_from_change_format(path):
    with open(path, 'r') as f:
        dialogues_json = json.load(f)
    
    random_indices_20 = random.sample(range(len(dialogues_json)), 20)

    extract_dialogue_list = []
    for i, dialogue in enumerate(dialogues_json):
        if i in random_indices_20:
            dialogue_json = {
                'dialogue_id': dialogue['dialogue_id'],
                'utterances': convert_raw_utterances_into_json_format(dialogue['utterances']),
                'inner_monologue_utterances': []
            }
            extract_dialogue_list.append(dialogue_json)

    return extract_dialogue_list

if __name__ == '__main__':
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    dialogue_list = extract_data_from_change_format(f'./RealPersonaChat/data/change_format/{TARGET_INTERLOCUTOR_ID}_dialogues.json')
    with open(f'{OUTPUT_DIR}/{TARGET_INTERLOCUTOR_ID}_extracted_data.json', 'w') as f:
        json.dump(dialogue_list, f, indent=2, ensure_ascii=False)