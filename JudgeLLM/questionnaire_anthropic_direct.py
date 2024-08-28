import os
import json

from dotenv import load_dotenv, find_dotenv
from tqdm import tqdm
from datasets import load_dataset
from anthropic import Anthropic


_ = load_dotenv(find_dotenv(), override=True)
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
client = Anthropic(api_key=ANTHROPIC_API_KEY)
TARGET_INTERLOCUTOR_IDS = os.getenv("TARGET_INTERLOCUTOR_IDS").split(',')

# TODO: evalかvalidかはデータの種類によって違うので、適宜変更する
OUTPUT_DIR = f'./JudgeLLM/result/eval/anthropic_direct'

# TODO: 内心描写も評価に含める場合は以下のプロンプトを編集する
normal_setting_prompt = '''
あなたは心理学者として、{target_interlocutor_id}の対話履歴を分析し、Big Five性格特性を予測する任務を担当しています。
'''

questionnaire_prompt = '''
あなたは対話履歴に基づいて、{target_interlocutor_id}のビッグファイブ性格特性を推測する役割を担っています。ビッグファイブ性格特性は以下の5つの項目から構成されています：

1. 外向性（Extraversion）
2. 神経症傾向（Neuroticism）
3. 開放性（Openness）
4. 誠実性（Conscientiousness）
5. 協調性（Agreeableness）

以下に提供される対話履歴を注意深く分析してください：

<dialogue_history>
{dialogues}
</dialogue_history>

この対話履歴を基に、上記の5つの性格特性それぞれについて、1から7までの数値（小数を含む）で評価してください。1は最も低い傾向を、7は最も高い傾向を示します。

各特性について、以下の手順で評価を行ってください：

1. 対話履歴から、その特性に関連する行動や発言を特定する
2. 特定した情報を基に、その特性の強さを分析する
4. 1から7までの数値で評価する

すべての特性の評価が終わったら、以下のフォーマットで結果のみを出力してください：

<result>
外向性: [数値]
神経症傾向: [数値]
開放性: [数値]
誠実性: [数値]
協調性: [数値]
</result>

それでは、対話履歴の分析と性格特性の評価を始めてください。
'''

def load_dialogue_list(path):
    with open(path, 'r') as f:
        dialogues_json = json.load(f)

    dialogue_list = []
    inner_monologue_list = []
    # TODO: テストのために20個の対話のみを取得
    for dialogue in dialogues_json:
        utterances = dialogue['utterances']
        inner_monologue_utterances = dialogue['inner_monologue_utterances']
        dialogue_text = '\n'.join([f"{utterance['role']} {utterance['action']}: {utterance['content']}" for utterance in utterances])
        inner_monologue_text = '\n'.join([f"{utterance['role']} {utterance['action']}: {utterance['content']}" for utterance in inner_monologue_utterances])
        dialogue_list.append(dialogue_text)
        inner_monologue_list.append(inner_monologue_text)

    return dialogue_list, inner_monologue_list

def get_answer(dialogue_list, target_interlocutor_id):
    dialogues = '\n\n'.join(dialogue_list)
    # プロンプトを作成
    # TODO: 普通の対話か内心描写付きかでnormal_setting_promptかinner_monologue_setting_promptを使い分ける
    system_prompt = normal_setting_prompt.format(target_interlocutor_id=target_interlocutor_id)
    user_prompt = questionnaire_prompt.format(target_interlocutor_id=target_interlocutor_id, dialogues=dialogues)

    message = client.messages.create(
        model='claude-3-5-sonnet-20240620',
        max_tokens=1024,
        temperature=0,
        system=system_prompt,
        messages=[{'role': 'user', 'content': user_prompt}]
    )

    return message.content[0].text

def calculate_error(scores, target_interlocutor_id):
    # 対象人物のBFIスコアを読み込む
    interlocutor_dataset = load_dataset("nu-dialogue/real-persona-chat", name='interlocutor', trust_remote_code=True)
    target_interlocutor_data = interlocutor_dataset['train'].filter(lambda x: x['interlocutor_id'] == target_interlocutor_id)
    target_interlocutor_BFI = target_interlocutor_data[0]['personality']
    target_interlocutor_BFI = {
        '外向性': target_interlocutor_BFI['BigFive_Extraversion'], 
        '神経症傾向': target_interlocutor_BFI['BigFive_Neuroticism'], 
        '開放性': target_interlocutor_BFI['BigFive_Openness'], 
        '誠実性': target_interlocutor_BFI['BigFive_Conscientiousness'], 
        '協調性': target_interlocutor_BFI['BigFive_Agreeableness']
    }

    # scoresとtarget_interlocutor_BFIの差を計算する
    error = {}
    for key in scores.keys():
        error[key] = scores[key] - target_interlocutor_BFI[key]
    
    # 平均二乗誤差を計算
    MSE = sum([error[key] ** 2 for key in error.keys()]) / 5

    return target_interlocutor_BFI, error, MSE


if __name__ == '__main__':
    # 出力ディレクトリの作成
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for target_interlocutor_id in tqdm(TARGET_INTERLOCUTOR_IDS):
        # TODO: 全部の対話データセットを取得するか、一部のみを取得するかはpathを変更することで対応可能
        dialogue_list, inner_monologue_list = load_dialogue_list(f'./RealPersonaChat/data/gen_inner_monologue/{target_interlocutor_id}_inner_monologue.json')
        # 対話データをAnthropicに投げて回答を取得
        answer = get_answer(dialogue_list, target_interlocutor_id)
        # 性格特性ごとにスコアを集計
        scores = {'外向性': 0, '神経症傾向': 0, '開放性': 0, '誠実性': 0, '協調性': 0}
        # answerを解析してscoresに代入
        answer = answer.split('<result>')[1].split('</result>')[0].split('\n')
        answer = answer[1:-1]

        for ans in answer:
            key, value = ans.split(': ')
            scores[key] = float(value)
        
        # 誤差と平均二乗誤差を計算
        true_scores, error, MSE = calculate_error(scores, target_interlocutor_id)
        # 結果をjsonファイルに保存
        results = []
        with open(f'{OUTPUT_DIR}/{target_interlocutor_id}_BFI_test.json', 'w') as f:
            results.append({'scores': scores})
            results.append({'true_scores': true_scores})
            results.append({'error': error})
            results.append({'MSE': MSE})
            json.dump(results, f, ensure_ascii=False, indent=2)