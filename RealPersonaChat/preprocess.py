from datasets import load_dataset
import json


'''
interlocutor_idsに含まれる話者の対話データセットをフォーマットを変換して保存
[
    {  
        "dialogue_id": 1,  
        "interlocutors": ["AA", "AB"],  
        "utterances": {  
            "utterance_id": [0, 1, 2, 3, 4, 5, ...],  
            "interlocutor_id": ["AA", "AB", "AA", "AB", "AA", "AB", ...],  
            "text": ["よろしくお願いいたします。", "よろしくお願いします！", "今日は涼しいですね", "雨が降って、何か涼しくなりましたね。", "そうですね、明日も涼しいと聞きました", "そうなんですか！でも、ちょっと湿度が高い気がします。", ...],
            "timestamp": [datetime.datetime(2022, 8, 6, 14, 51, 18, 360000), datetime.datetime(2022, 8, 6, 14, 51, 48, 482000), datetime.datetime(2022, 8, 6, 14, 51, 55, 538000), datetime.datetime(2022, 8, 6, 14, 52, 07, 388000), datetime.datetime(2022, 8, 6, 14, 52, 16, 400000), datetime.datetime(2022, 8, 6, 14, 52, 31, 076000), ...]  
        },  
        "evaluations": {  
            "interlocutor_id": ["AA", "AB"],  
            "informativeness": [5, 5],  
            "comprehension": [5, 5],  
            "familiarity": [5, 5],  
            "interest": [5, 5],  
            "proactiveness": [5, 5],  
            "satisfaction": [5, 5]  
        }  
    },
    ...  
]
            |
            |
            V
[
    {
        "dialogue_id": 1,
        "interlocutors": [
            "AA",
            "AB"
        ],
        "utterances": [
            "AA (speaking): よろしくお願いいたします。"
            "AB (speaking): よろしくお願いします！"
            "CA (speaking): 今日は涼しいですね", "雨が降って、何か涼しくなりましたね。"
            ...
        ]
    },
    ...    
]
'''
def convert_dialogues_format(dialogue_dataset, interlocutor_dataset, target_interlocutor_ids):
    for target_interlocutor_id in target_interlocutor_ids:
        interlocutor_dialogues = []
        for dialogue_data in dialogue_dataset['train']:
            # 対象のinterlocutor_idの対話データのみを抽出
            if target_interlocutor_id in dialogue_data['interlocutors']:
                dialogue = {
                    "dialogue_id": dialogue_data['dialogue_id'],
                    "interlocutors": dialogue_data['interlocutors'],
                    "utterances": []
                }
                # utterancesのフォーマット変換
                for interlocutor_id, text in zip(dialogue_data['utterances']['interlocutor_id'], dialogue_data['utterances']['text']):
                    dialogue['utterances'].append(f"{interlocutor_id} (speaking): {text}")
                interlocutor_dialogues.append(dialogue)

        # ファイルの保存
        with open(f"./RealPersonaChat/data/{target_interlocutor_id}_dialogues.json", 'w') as f:
            json.dump(interlocutor_dialogues, f, ensure_ascii=False, indent=2)

if __name__ == '__main__':
    # データセットの読み込み
    dialogue_dataset = load_dataset("nu-dialogue/real-persona-chat", name='dialogue', trust_remote_code=True)
    interlocutor_dataset = load_dataset("nu-dialogue/real-persona-chat", name='interlocutor', trust_remote_code=True)

    # 読み込む対象のinterlocutor_idのリストを指定
    target_interlocutor_ids = ['CP', 'AT', 'FR', 'CA', 'AY', 'DV', 'CE']

    # フォーマット変換
    convert_dialogues_format(dialogue_dataset, interlocutor_dataset, target_interlocutor_ids)