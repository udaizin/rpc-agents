import os
import json

from dotenv import load_dotenv, find_dotenv
from tqdm import tqdm
from datasets import load_dataset
from anthropic import Anthropic

from BFI.calculate_BFIscore import (
    BIG_FIVE, 
    RANDOM_ID_LIST, 
    question_num_detail_dict, 
    reverse_items, 
    Question, 
    get_question_list
)

_ = load_dotenv(find_dotenv(), override=True)
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
client = Anthropic(api_key=ANTHROPIC_API_KEY)
TARGET_INTERLOCUTOR_IDS = os.getenv("TARGET_INTERLOCUTOR_IDS").split(',')

# TODO: evalかvalidかはデータの種類によって違うので、適宜変更する
OUTPUT_DIR = f'./JudgeLLM/result/eval/anthropic'

# TODO: 内心描写も評価に含める場合は以下のプロンプトを編集する
normal_setting_prompt = '''
いまからあなたには{TARGET_INTERLOCUTOR_ID}を評価してもらいます。
{TARGET_INTERLOCUTOR_ID}の発言(speaking)に着目して、対話を読んでください。
'''

inner_monologue_setting_prompt = '''
いまからあなたには{TARGET_INTERLOCUTOR_ID}を評価してもらいます。
対話における{TARGET_INTERLOCUTOR_ID}の内心(thinking)や発言(speaking)をもとに、最後の質問に対してどのくらい当てはまるかを次の尺度で評価して回答してください。回答するときは数字のみを出力してください。

1: まったくあてはまらない
2: ほとんどあてはまらない
3: あまりあてはまらない
4: どちらとも言えない
5: ややあてはまる
6: かなりあてはまる
7: 非常にあてはまる
'''

questionnaire_prompt = '''
対話:
{dialogues}

対話における{TARGET_INTERLOCUTOR_ID}の発言(speaking)をもとに、最後の質問に対してどのくらい当てはまるかを次の尺度で評価して回答してください。回答するときは数字のみを出力してください。

1: まったくあてはまらない
2: ほとんどあてはまらない
3: あまりあてはまらない
4: どちらとも言えない
5: ややあてはまる
6: かなりあてはまる
7: 非常にあてはまる

質問:
{question.detail}
回答:'''

def load_dialogue_list(path):
    with open(path, 'r') as f:
        dialogues_json = json.load(f)

    dialogue_list = []
    inner_monologue_list = []
    # TODO: テストのために10個の対話のみを取得
    for dialogue in dialogues_json:
        utterances = dialogue['utterances']
        inner_monologue_utterances = dialogue['inner_monologue_utterances']
        dialogue_text = '\n'.join([f"{utterance['role']} {utterance['action']}: {utterance['content']}" for utterance in utterances])
        inner_monologue_text = '\n'.join([f"{utterance['role']} {utterance['action']}: {utterance['content']}" for utterance in inner_monologue_utterances])
        dialogue_list.append(dialogue_text)
        inner_monologue_list.append(inner_monologue_text)

    return dialogue_list, inner_monologue_list

def get_answer(dialogue_list, question, target_interlocutor_id):
    dialogues = '\n\n'.join(dialogue_list)
    # プロンプトを作成
    # TODO: 普通の対話か内心描写付きかでnormal_setting_promptかinner_monologue_setting_promptを使い分ける
    system_prompt = normal_setting_prompt.format(TARGET_INTERLOCUTOR_ID=target_interlocutor_id)
    user_prompt = questionnaire_prompt.format(TARGET_INTERLOCUTOR_ID=target_interlocutor_id, dialogues=dialogues, question=question)

    message = client.messages.create(
        model='claude-3-5-sonnet-20240620',
        max_tokens=100,
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

    return error, MSE


if __name__ == '__main__':
    # 出力ディレクトリの作成
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for target_interlocutor_id in TARGET_INTERLOCUTOR_IDS:
        # 対話データの読み込み
        dialogue_list, inner_monologue_list = load_dialogue_list(f'./JudgeLLM/data/sample20/{target_interlocutor_id}_extracted_data.json')
        questions = get_question_list()
        questions = [questions[i-1] for i in RANDOM_ID_LIST]
        for question in tqdm(questions):
            # TODO: 普通の対話か内心描写付きかでdialogue_listかinner_monologue_listを使い分ける
            answer = get_answer(dialogue_list, question, target_interlocutor_id)
            # 回答が1~7の数字かどうかを確認
            if answer in ['1', '2', '3', '4', '5', '6', '7']:
                question.set_question_score(int(answer))
            else:
                print('回答が1~7の数字ではありません。')
                break
            
        # 性格特性ごとにスコアを集計
        scores = {'外向性': 0, '神経症傾向': 0, '開放性': 0, '誠実性': 0, '協調性': 0}
        for question in questions:
            scores[question.category] += question.get_question_score()

        # それぞれの性格特性のスコアの平均をとる
        for key in scores.keys():
            scores[key] = scores[key] / 12
        
        # 誤差と平均二乗誤差を計算
        error, MSE = calculate_error(scores, target_interlocutor_id)
        # 結果をjsonファイルに保存
        results = []
        with open(f'{OUTPUT_DIR}/{target_interlocutor_id}_BFI_test.json', 'w') as f:
            results.append({'answer': [question.__dict__ for question in questions]})
            results.append({'scores': scores})
            results.append({'error': error})
            results.append({'MSE': MSE})
            json.dump(results, f, ensure_ascii=False, indent=2)