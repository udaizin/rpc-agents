import os
import json
import re
from tqdm.contrib import tzip
from dotenv import load_dotenv, find_dotenv
from openai import OpenAI
from datasets import load_dataset


_ = load_dotenv(find_dotenv())  
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TARGET_INTERLOCUTOR_IDS = os.getenv("TARGET_INTERLOCUTOR_IDS").split(',')
BIG_FIVE_PERSONALITY_TRAITS_TRANS_DICT = {'BigFive_Openness': '開放性', 'BigFive_Conscientiousness': '誠実性', 
                                          'BigFive_Extraversion': '外向性', 'BigFive_Agreeableness': '協調性', 'BigFive_Neuroticism': '神経症傾向'}
GENDER_TRANS_DICT = {0: '男性', 1: '女性', 2: 'その他'}
client = OpenAI(api_key=OPENAI_API_KEY)

# ビッグファイブ性格特性の数値(1~7)を(低い, 中程度, 高い)の3段階で評価する関数
def change_bigfive_number_to_level(bigfive_number: float) -> str:
    if bigfive_number <= 3.0:
        return "低い"
    elif bigfive_number <= 5.0:
        return "中程度"
    else:
        return "高い"

# GPTによって生成された内心描写付き対話データの形式が正しいのかチェックする関数
def validate_monologue_format(original_dialogue: str, inner_monologue_dialogue: str, target_interlocutor_id: str, partner_interlocutor_id: str) -> bool:
    # inner_monologue_dialogueのすべての行が{target_interlocutor_id} (speaking): 〜 または {target_interlocutor_id} (thinking): 〜 または {partner_interlocutor_id} (speaking): 〜 で始まっているかチェック
    inner_monologue_lines = inner_monologue_dialogue.split('\n')
    for inner_monologue_line in inner_monologue_lines:
        if not (inner_monologue_line.startswith(f'{target_interlocutor_id} (speaking):') or inner_monologue_line.startswith(f'{target_interlocutor_id} (thinking):') or inner_monologue_line.startswith(f'{partner_interlocutor_id} (speaking):')):
            print(f'inner_monologue_line: {inner_monologue_line}')
            print('inner_monologue_dialogueのすべての行が{target_interlocutor_id} (speaking): 〜 または {target_interlocutor_id} (thinking): 〜 または {partner_interlocutor_id} (speaking): 〜 で始まっていません。')
            return False

    original_dialogue_lines = original_dialogue.split('\n')
    inner_monologue_speaking_lines = [line for line in inner_monologue_dialogue.split('\n') if line.startswith(f'{target_interlocutor_id} (speaking):') or line.startswith(f'{partner_interlocutor_id} (speaking):')]
    inner_monologue_thinking_lines = [line for line in inner_monologue_dialogue.split('\n') if line.startswith(f'{target_interlocutor_id} (thinking):')]

    # original_dialogue_linesの内容がinner_monologue_speaking_linesで書き換わってないかチェック
    # ここで{target_interlocutor_id} (speaking): 〜 (thinking) などという行があるかどうかの確認も兼ねている
    for original_dialogue_line, inner_monologue_speaking_line in zip(original_dialogue_lines, inner_monologue_speaking_lines):
        if original_dialogue_line != inner_monologue_speaking_line:
            print(f'original_dialogue_line: {original_dialogue_line}')
            print(f'inner_monologue_speaking_line: {inner_monologue_speaking_line}')
            print('original_dialogue_lineの内容がinner_monologue_speaking_lineで書き換わっています。')
            return False

    return True

def postprocess_inner_monologue(inner_monologue_dialogue: str, target_interlocutor_id) -> str:
    # {target_interlocutor_id} (thinking): ~ の ~ が()で囲まれている場合、()を削除
    # 例: {target_interlocutor_id} (thinking): (あいうえお) -> {target_interlocutor_id} (thinking): あいうえお
    inner_monologue_dialogue = re.sub(rf'{target_interlocutor_id} \(thinking\): \((.*?)\)', rf'{target_interlocutor_id} (thinking): \1', inner_monologue_dialogue)
    return inner_monologue_dialogue


def create_inner_monologue_annotation(utterances: str, target_interlocutor_id: str, partner_interlocutor_id: str) -> str:
    # インデントなどを整えるため、'\n'でjoin
    utterances_example = '\n'.join([
        f'{partner_interlocutor_id} (speaking): ...',
        f'{target_interlocutor_id} (speaking): ...',
        f'{partner_interlocutor_id} (speaking): ...',
        f'{target_interlocutor_id} (speaking): ...',
        f'{partner_interlocutor_id} (speaking): ...',
        f'{target_interlocutor_id} (speaking): ...',
        f'{partner_interlocutor_id} (speaking): ...',
        f'{target_interlocutor_id} (speaking): ...'
    ])
    inner_monologue_example = '\n'.join([
        f'{partner_interlocutor_id} (speaking): ...',
        f'{target_interlocutor_id} (speaking): ...',
        f'{partner_interlocutor_id} (speaking): ...',
        f'{target_interlocutor_id} (thinking): (具体的な感情や考え)',
        f'{target_interlocutor_id} (speaking): ...',
        f'{target_interlocutor_id} (thinking): (具体的な感情や考え)',
        f'{partner_interlocutor_id} (speaking): ...',
        f'{target_interlocutor_id} (speaking): ...',
        f'{target_interlocutor_id} (thinking): (具体的な感情や考え)',
        f'{partner_interlocutor_id} (speaking): ...',
        f'{target_interlocutor_id} (thinking): (具体的な感情や考え)',
        f'{target_interlocutor_id} (speaking): ...'
    ]) 
    system_prompt = '\n'.join([
        f'あなたは{target_interlocutor_id}です。あなたの性別やペルソナ、性格特性は次のようになっています。',
        f'性別: {target_interlocutor_gender_jp}',
        f'ペルソナ: {target_interlocutor_persona_prompt}',
        f'性格特性: {target_interlocutor_personality_prompt}',
        '',          
        'ルール:',
        f'1. {target_interlocutor_id}には感情と思考能力があります。{target_interlocutor_id}が何を感じ、何を考えているのか慎重に考えてください。',
        f'2. あなたは今、ペルソナや性格特性に基づいて、対話履歴に{target_interlocutor_id}の感情や考えを追加することを目的としています。',
        f'3. 主人公は{target_interlocutor_id}です。対話履歴の中に、{target_interlocutor_id}が何かを感じたり考えたりしたと思うところに、(thinking)のラベルを用いて{target_interlocutor_id}の気持ちや考えを挿入してください。',
        f'4. (speaking)のラベルの内容は「絶対に」書き換えないでそのまま残してください。',
        f'5. 「{target_interlocutor_id} (thinking): 〜 」という形式必ず従って、{target_interlocutor_id}の内心描写を追加してください。{partner_interlocutor_id} (thinking): 〜 という行は絶対に作らないでください。',
        f'6. 各行に(speaking)や(thinking)のラベルは1つしか含まれないようにしてください。',
        f'7. 基本的に(thinking)ラベルは{partner_interlocutor_id}の発言の後または{target_interlocutor_id}の発言の前に挿入してください。ただし、{target_interlocutor_id}が発言後に感情や考えを持った場合は、その発言の後に(thinking)ラベルを挿入してください。',
        '',
        '次に、もとの対話履歴と出力のフォーマットの例を示します。',
        'もとの対話履歴の例:',
        f'{utterances_example}',
        '出力の例:',
        f'{inner_monologue_example}',
        '回答を生成する際には、「出力:」を含めないでください。また、出力には空行を含めないでください'
    ])

    user_first_prompt = '\n'.join([
        '以下の対話履歴を読んで、{target_interlocutor_id}の思考や感情が動いた場所に行を適宜挿入してください。',
        '行を挿入する際は、(thinking): 〜 という形式で{target_interlocutor_id}の内心描写を追加してください。',
        '',
        f'対話履歴:',
        f'{utterances}'
    ])

    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_first_prompt},
        ],
        temperature=0.7,
    )

    return completion.choices[0].message.content

if __name__ == '__main__':
    # RealPersonaChatのinterlocutor_datasetを読み込む
    interlocutor_dataset = load_dataset("nu-dialogue/real-persona-chat", name='interlocutor', trust_remote_code=True)

    for target_interlocutor_id in TARGET_INTERLOCUTOR_IDS:
        # interlocutor_datasetからtarget_interlocutor_idに対応するinterlocutorの情報を取得
        interlocutor_data = interlocutor_dataset['train'].filter(lambda x: x['interlocutor_id'] == target_interlocutor_id)
        target_interlocutor_persona = interlocutor_data[0]['persona']
        target_interlocutor_personality = interlocutor_data[0]['personality']
        target_interlocutor_gender = interlocutor_data[0]['demographic_information']['gender']
        target_interlocutor_gender_jp = GENDER_TRANS_DICT.get(target_interlocutor_gender, 'その他')
        
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
        utterances_list = ['\n'.join(dialogue['utterances']) for dialogue in dialogues[:10]]
        dialogue_id_list = [dialogue['dialogue_id'] for dialogue in dialogues[:10]]
        interlocutors_list = [dialogue['interlocutors'] for dialogue in dialogues[:10]]
        partner_interlocutor_id_list = [interlocutors[0] if interlocutors[1] == target_interlocutor_id else interlocutors[1] for interlocutors in interlocutors_list]

        dialogues_first_person_list = []
        for utterances, dialogue_id, partner_interlocutor_id in tzip(utterances_list, dialogue_id_list, partner_interlocutor_id_list):
            # 正しいフォーマットの内心描写付き対話データが生成されるまで繰り返す
            while True:
                inner_monologue_utterances = create_inner_monologue_annotation(utterances, target_interlocutor_id, partner_interlocutor_id)
                inner_monologue_utterances = postprocess_inner_monologue(inner_monologue_utterances, target_interlocutor_id)
                if validate_monologue_format(utterances, inner_monologue_utterances, target_interlocutor_id, partner_interlocutor_id):
                    break
                else:
                    print('inner_monologue_utterancesの形式が正しくありません。再度生成します。')

            dialogue_first_person_dict = {
                "dialogue_id": dialogue_id,
                "utterances": utterances,
                "first_person_dialogue": inner_monologue_utterances
            }
            dialogues_first_person_list.append(dialogue_first_person_dict)

        # json形式で保存
        with open(f'./RealPersonaChat/data/tmp/{target_interlocutor_id}_inner_monologue.json', 'w') as f:
            json.dump(dialogues_first_person_list, f, ensure_ascii=False, indent=2)
