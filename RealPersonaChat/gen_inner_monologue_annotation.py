import os
import json
from pprint import pprint
from tqdm.contrib import tzip
from dotenv import load_dotenv, find_dotenv
from openai import OpenAI
from datasets import load_dataset
import textwrap


_ = load_dotenv(find_dotenv())  
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TARGET_INTERLOCUTOR_IDS = os.getenv("TARGET_INTERLOCUTOR_IDS").split(',')
BIG_FIVE_PERSONALITY_TRAITS_TRANS_DICT = {'BigFive_Openness': '開放性', 'BigFive_Conscientiousness': '誠実性', 
                                          'BigFive_Extraversion': '外向性', 'BigFive_Agreeableness': '協調性', 'BigFive_Neuroticism': '神経症傾向'}
client = OpenAI(api_key=OPENAI_API_KEY)

# ビッグファイブ性格特性の数値(1~7)を(低い, 中程度, 高い)の3段階で評価する関数
def change_bigfive_number_to_level(bigfive_number):
    if bigfive_number <= 3.0:
        return "低い"
    elif bigfive_number <= 5.0:
        return "中程度"
    else:
        return "高い"


if __name__ == '__main__':
    # RealPersonaChatのinterlocutor_datasetを読み込む
    interlocutor_dataset = load_dataset("nu-dialogue/real-persona-chat", name='interlocutor', trust_remote_code=True)

    for target_interlocutor_id in TARGET_INTERLOCUTOR_IDS:
        # interlocutor_datasetからtarget_interlocutor_idに対応するinterlocutorの情報を取得
        interlocutor_data = interlocutor_dataset['train'].filter(lambda x: x['interlocutor_id'] == target_interlocutor_id)
        target_interlocutor_persona = interlocutor_data[0]['persona']
        target_interlocutor_personality = interlocutor_data[0]['personality']
        
        # ペルソナのプロンプト作成
        target_interlocutor_persona_prompt = f'{target_interlocutor_persona}'
        # 性格特性のプロンプト作成
        target_interlocutor_personality_prompt = '['
        for trait_en, trait_jp in BIG_FIVE_PERSONALITY_TRAITS_TRANS_DICT.items():
            bigfive_level = change_bigfive_number_to_level(target_interlocutor_personality[trait_en])
            target_interlocutor_personality_prompt = f'{target_interlocutor_personality_prompt}{trait_jp}: {bigfive_level}, '
        target_interlocutor_personality_prompt = f'{target_interlocutor_personality_prompt[:-2]}]'


        # target_interlocutor_idに対応するdialogues.jsonを読み込む
        with open(f'./RealPersonaChat/data/change_format/{target_interlocutor_id}_dialogues.json', 'r') as f:
            dialogues = json.load(f)
        
        # 実験のため少数の対話データのみを取得
        utterances_list = ['\n'.join(dialogue['utterances']) for dialogue in dialogues[:1]]
        dialogue_id_list = [dialogue['dialogue_id'] for dialogue in dialogues[:1]]
        interlocutors_list = [dialogue['interlocutors'] for dialogue in dialogues[:1]]
        partner_interlocutor_id_list = [interlocutors[0] if interlocutors[1] == target_interlocutor_id else interlocutors[1] for interlocutors in interlocutors_list]

        dialogues_first_person_list = []
        for utterances, dialogue_id, partner_interlocutor_id in tzip(utterances_list, dialogue_id_list, partner_interlocutor_id_list):
            user_prompt = '\n'.join([f'{target_interlocutor_id}の基本情報は次のとおりです。',
                            f'ペルソナ: {target_interlocutor_persona_prompt}',
                            f'性格特性: {target_interlocutor_personality_prompt}',
                            '',          
                           '以下のルールに従って、対話履歴を1人称の視点で再構成してください。',
                            f'1. {target_interlocutor_id}には感情と思考能力があります。{target_interlocutor_id}が何を感じ、何を考えているのか慎重に考えてください。',
                            f'2. あなたは今、ペルソナや性格特性に基づいて、対話履歴を{target_interlocutor_id}の一人称視点に変換することを目的としています。',
                            f'3. 主人公は{target_interlocutor_id}です。対話履歴の中に、{target_interlocutor_id}が何かを感じたり考えたりしたと思うところに、(thinking)のラベルを用いて{target_interlocutor_id}の気持ちや考えを挿入してください。',
                            f'4. (speaking)のラベルの内容は「絶対に」書き換えないでください。ただし、{target_interlocutor_id}の思考があったと思うときに(speaking)のラベルの前か後ろに、(thinking)のラベルを挿入することができます。',
                            f'5. Big Five性格特性やペルソナを考慮して、{target_interlocutor_id}の感情や思考を表現してください。',
                            f'6. 以下はフォーマットの例です。(speaking)や(thinking)のラベルは各行にどちらか1つしか含めないでください。',
                            f'例:',
                            f'{target_interlocutor_id} (speaking): こんにちは。今日はいい天気ですね。',
                            f'{target_interlocutor_id} (thinking): 今日は天気がいいので、散歩に行こうかな。',
                            f'{partner_interlocutor_id} (speaking): こんにちは。そうですね。',
                            'or',
                            f'{target_interlocutor_id} (thinking): 今日は天気がよくて、気持ちがいいな。',
                            f'{target_interlocutor_id} (speaking): こんにちは。今日はいい天気ですね。',
                            f'{partner_interlocutor_id} (speaking): こんにちは。そうですね。',
                            '',       
                            '対話履歴:',
                            f'{utterances}'])

            print(user_prompt)

            completion = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": f"あなたは{target_interlocutor_id}です。ルールに基づいて対話履歴を再構成してください。"},
                    {"role": "user", "content": user_prompt}
                ],
            )

            dialogue_first_person_dict = {
                "dialogue_id": dialogue_id,
                "utterances": utterances,
                "first_person_dialogue": completion.choices[0].message.content
            }
            dialogues_first_person_list.append(dialogue_first_person_dict)
            print('First Person Dialogue:')
            print(completion.choices[0].message.content)

        # json形式で保存
        with open(f'./RealPersonaChat/data/gen_inner_monologue/{target_interlocutor_id}_inner_monologue.json', 'w') as f:
            json.dump(dialogues_first_person_list, f, ensure_ascii=False, indent=2)