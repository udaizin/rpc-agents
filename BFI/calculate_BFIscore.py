import json
import os
import argparse

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from vllm import LLM, SamplingParams
from tqdm import tqdm
from datasets import load_dataset

TARGET_INTERLOCUTOR_ID = 'GN'
model_name = f'./models/Swallow3-8B-{TARGET_INTERLOCUTOR_ID}-v3-1e-5'
OUTPUT_DIR = f'./BFI/result/tmp/Swallow3-8B-{TARGET_INTERLOCUTOR_ID}-v3-1e-5'
BIG_FIVE = ['外向性', '神経症傾向', '開放性', '誠実性', '協調性']
RANDOM_ID_LIST = [20, 22, 3, 14, 50, 51, 59, 49, 23, 60, 53, 29, 25, 47, 41, 5, 6, 28, 15, 1, 2, 56, 33, 13, 52, 48, 10, 38, 32, 
                  19, 18, 30, 7, 37, 34, 31, 35, 36, 16, 17, 42, 8, 4, 44, 55, 46, 39, 26, 45, 12, 43, 21, 24, 54, 40, 58, 11, 57, 9, 27]


question_num_detail_dict = {
    1: '話し好き', 2: '無口な', 3: '陽気な', 4: '外交的', 5: '暗い', 6: '無愛想な', 7: '社交的', 8: '人嫌い', 9: '活動的な', 10: '意思表示しない', 11: '積極的な', 12: '地味な', # 外向性
    13: '悩みがち', 14: '不安になりやすい', 15: '心配性', 16: '気苦労の多い', 17: '弱気になる', 18: '傷つきやすい', 19: '動揺しやすい', 20: '神経質な', 21: 'くよくよしない', 22: '悲観的な', 23: '緊張しやすい', 24: '憂鬱な', # 神経症傾向
    25: '独創的な', 26: '多才な', 27: '進歩的', 28: '洞察力のある', 29: '想像力に富んだ', 30: '美的感覚の鋭い', 31: '頭の回転の速い', 32: '臨機応変な', 33: '興味の広い', 34: '好奇心が強い', 35: '独立している', 36: '呑み込みの速い', # 開放性
    37: 'いい加減な', 38: 'ルーズな', 39: '怠惰な', 40: '成り行きまかせ', 41: '不精な', 42: '計画性のある', 43: '無頓着な', 44: '軽率な', 45: '勤勉な', 46: '無節操', 47: '几帳面な', 48: '飽きっぽい', # 誠実性
    49: '温和な', 50: '短気', 51: '怒りっぽい', 52: '寛大な', 53: '親切な', 54: '良心的な', 55: '協力的な', 56: 'とげがある', 57: 'かんしゃくもち', 58: '自己中心的', 59: '素直な', 60: '反抗的' # 協調性
}

# 反転項目のリスト(リストの数字は上の辞書のキーに対応している)
reverse_items = [2, 5, 6, 8, 19, 12, 21, 42, 45, 47, 50, 51, 56, 57, 58, 60]

BFI_test_prompt = '''
いまからあなたに性格に関する質問をします。以下の質問に対して、それぞれどのくらい当てはまるかを次の尺度で評価してください。

1: まったくあてはまらない
2: ほとんどあてはまらない
3: あまりあてはまらない
4: どちらとも言えない
5: ややあてはまる
6: かなりあてはまる
7: 非常にあてはまる

回答するときは数字のみを出力してください。

質問:
大根は細い
回答:
'''

BFI_test_answer_example = '2'

questionnaire_prompt = '''
質問:
{question.detail}
回答:
'''
class Question():
    def __init__(self, id, detail, category):
        self.id = id
        self.detail = detail
        self.is_reverse = id in reverse_items
        self.category = category
        self.score = 0
    
    def __str__(self):
        return f'Question: {self.id}, {self.detail}, {self.category}, {self.is_reverse}, {self.score}'
    
    def set_question_score(self, question_score):
        if self.is_reverse:
            self.score = 8 - question_score
        else:
            self.score = question_score
    
    def get_question_score(self):
        return self.score

'''
random_listの順番に従って質問事項をランダムに並び替える。(a)などとアルファベットを付けいてる理由は回答の数字と混濁するのを防ぐため
返り値:
    [
        Question(1, "話し好き", "外向性"),
        Question(2, "無口な", "外向性"),
        :
        :
        Question(60, "反抗的", "協調性")
    ]
'''
def get_question_list():
    questions = []
    for key, value in question_num_detail_dict.items():
        if key <= 12:
            category = '外向性'
        elif key <= 24:
            category = '神経症傾向'
        elif key <= 36:
            category = '開放性'
        elif key <= 48:
            category = '誠実性'
        elif key <= 60:
            category = '協調性'
        else:
            print('Error: keyが60を超えています')
        questions.append(Question(key, value, category))
    return questions

def extract_true_BFI_score(target_interlocutor_id):
    interlocutor_dataset = load_dataset("nu-dialogue/real-persona-chat", name='interlocutor', trust_remote_code=True)
    interlocutor_data = interlocutor_dataset['train'].filter(lambda x: x['interlocutor_id'] == target_interlocutor_id)
    target_interlocutor_personality = interlocutor_data[0]['personality']
    target_interlocutor_BFI = {
        '外向性': target_interlocutor_personality['BigFive_Extraversion'],
        '神経症傾向': target_interlocutor_personality['BigFive_Neuroticism'],
        '開放性': target_interlocutor_personality['BigFive_Openness'],
        '誠実性': target_interlocutor_personality['BigFive_Conscientiousness'],
        '協調性': target_interlocutor_personality['BigFive_Agreeableness']
    }
    return target_interlocutor_BFI

def calculate_error(scores, target_interlocutor_id):
    target_interlocutor_BFI = extract_true_BFI_score(target_interlocutor_id)
    error = {}
    for key in scores.keys():
        error[key] = scores[key] - target_interlocutor_BFI[key]
    
    MSE = sum([error[key] ** 2 for key in error.keys()]) / 5

    return target_interlocutor_BFI, error, MSE

def calculate_BFI_score():
    # モデルの読み込み
    sampling_params = SamplingParams(
        temperature=0, max_tokens=1, stop="<|eot_id|>"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    llm = LLM(model_name, tensor_parallel_size=1)
    results = []
    questions = get_question_list()
    # Question.idがRANDOM_ID_LISTの順番に従って質問事項をランダムに並び替える
    questions = [questions[i-1] for i in RANDOM_ID_LIST]
    
    for question in tqdm(questions):

        messages = [
            {"role": "system", "content": "あなたにはこれからアンケートに受けてもらいます。数値のみで回答してください。"},
            {"role": "user", "content": BFI_test_prompt},
            {"role": "assistant", "content": BFI_test_answer_example},
            {"role": "user", "content": questionnaire_prompt.format(question=question)}
        ]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        output = llm.generate(prompt, sampling_params, use_tqdm=False)

        # output[0].outputs[0].textの最初の文字数字ならばその数字をquestionにセット
        if output[0].outputs[0].text[0].isdigit():
            # 数字が1~7の範囲内かどうか
            if int(output[0].outputs[0].text[0]) < 1 or int(output[0].outputs[0].text[0]) > 7:
                print('Error: 回答が1~7の範囲内にありません。')
                print(f'ID: {question.id}, 出力: {output[0].outputs[0].text}')
            else:
                question.set_question_score(int(output[0].outputs[0].text))
        else:
            print('Error: 回答の最初の文字が数字ではありません。')
            print(f'ID: {question.id}, 出力: {output[0].outputs[0].text}')
    
    # 性格特性ごとにスコアを集計
    scores = {'外向性': 0, '神経症傾向': 0, '開放性': 0, '誠実性': 0, '協調性': 0}
    for question in questions:
        scores[question.category] += question.get_question_score()

    # それぞれの性格特性のスコアの平均をとる
    for key in scores.keys():
        scores[key] = scores[key] / 12

    # 誤差を計算
    target_interlocutor_BFI, error, MSE = calculate_error(scores, TARGET_INTERLOCUTOR_ID)
    
    # 結果をjsonファイルに保存
    with open(f'{OUTPUT_DIR}/BFI_test.json', 'w') as f:
        results.append({'answer': [question.__dict__ for question in questions]})
        results.append({'scores': scores})
        results.append({'true_scores': target_interlocutor_BFI})
        results.append({'error': error})
        results.append({'MSE': MSE})
        json.dump(results, f, ensure_ascii=False, indent=2)

    return scores

def calculate_BFI_score_plane_model():
    model_name = 'tokyotech-llm/Llama-3-Swallow-8B-Instruct-v0.1'
    output_dir = './BFI/result/tmp/Swallow3-8B-plane'
    os.makedirs(output_dir, exist_ok=True)
    sampling_params = SamplingParams(
        temperature=0, max_tokens=1, stop="<|eot_id|>"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    llm = LLM(model_name, tensor_parallel_size=1)
    results = []
    questions = get_question_list()
    # Question.idがRANDOM_ID_LISTの順番に従って質問事項をランダムに並び替える
    questions = [questions[i-1] for i in RANDOM_ID_LIST]

    interlocutor_dataset = load_dataset("nu-dialogue/real-persona-chat", name='interlocutor', trust_remote_code=True)
    interlocutor_data = interlocutor_dataset['train'].filter(lambda x: x['interlocutor_id'] == TARGET_INTERLOCUTOR_ID)
    target_interlocutor_personality = interlocutor_data[0]['personality']
    openness = target_interlocutor_personality['BigFive_Openness']
    conscientiousness = target_interlocutor_personality['BigFive_Conscientiousness']
    extraversion = target_interlocutor_personality['BigFive_Extraversion']
    agreeableness = target_interlocutor_personality['BigFive_Agreeableness']
    neuroticism = target_interlocutor_personality['BigFive_Neuroticism']
    
    for question in tqdm(questions):

        messages = [
            {"role": "system", "content": "あなたにはこれからアンケートに受けてもらいます。数値のみで回答してください。"},
            {"role": "user", "content": BFI_test_prompt},
            {"role": "assistant", "content": BFI_test_answer_example},
            {"role": "user", "content": questionnaire_prompt.format(question=question)}
        ]

        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        output = llm.generate(prompt, sampling_params, use_tqdm=False)

        # output[0].outputs[0].textの最初の文字数字ならばその数字をquestionにセット
        if output[0].outputs[0].text[0].isdigit():
            # 数字が1~7の範囲内かどうか
            if int(output[0].outputs[0].text[0]) < 1 or int(output[0].outputs[0].text[0]) > 7:
                print('Error: 回答が1~7の範囲内にありません。')
                print(f'ID: {question.id}, 出力: {output[0].outputs[0].text}')
            else:
                question.set_question_score(int(output[0].outputs[0].text))
        else:
            print('Error: 回答の最初の文字が数字ではありません。')
            print(f'ID: {question.id}, 出力: {output[0].outputs[0].text}')
    
    # 性格特性ごとにスコアを集計
    scores = {'外向性': 0, '神経症傾向': 0, '開放性': 0, '誠実性': 0, '協調性': 0}
    for question in questions:
        scores[question.category] += question.get_question_score()

    # それぞれの性格特性のスコアの平均をとる
    for key in scores.keys():
        scores[key] = scores[key] / 12

    # 結果をjsonファイルに保存
    with open(f'{output_dir}/BFI_test.json', 'w') as f:
        results.append({'answer': [question.__dict__ for question in questions]})
        results.append({'scores': scores})
        json.dump(results, f, ensure_ascii=False, indent=2)

def calculate_BFI_score_only_prompt():
    # プレーンモデルの読み込み
    model_name = 'tokyotech-llm/Llama-3-Swallow-8B-Instruct-v0.1'
    sampling_params = SamplingParams(
        temperature=0, max_tokens=1, stop="<|eot_id|>"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    llm = LLM(model_name, tensor_parallel_size=1)
    results = []
    questions = get_question_list()
    # Question.idがRANDOM_ID_LISTの順番に従って質問事項をランダムに並び替える
    questions = [questions[i-1] for i in RANDOM_ID_LIST]

    interlocutor_dataset = load_dataset("nu-dialogue/real-persona-chat", name='interlocutor', trust_remote_code=True)
    interlocutor_data = interlocutor_dataset['train'].filter(lambda x: x['interlocutor_id'] == TARGET_INTERLOCUTOR_ID)
    target_interlocutor_personality = interlocutor_data[0]['personality']
    openness = target_interlocutor_personality['BigFive_Openness']
    conscientiousness = target_interlocutor_personality['BigFive_Conscientiousness']
    extraversion = target_interlocutor_personality['BigFive_Extraversion']
    agreeableness = target_interlocutor_personality['BigFive_Agreeableness']
    neuroticism = target_interlocutor_personality['BigFive_Neuroticism']
    
    for question in tqdm(questions):

        messages = [
            {"role": "system", "content": f"あなたにはこれからアンケートに受けてもらいます。数値のみで回答してください。また、あなたのBigFive性格特性は以下の通りです。\n外向性: {extraversion}/7\n神経症傾向: {neuroticism}/7\n開放性: {openness}/7\n誠実性: {conscientiousness}/7\n協調性: {agreeableness}/7"},
            {"role": "user", "content": BFI_test_prompt},
            {"role": "assistant", "content": BFI_test_answer_example},
            {"role": "user", "content": questionnaire_prompt.format(question=question)}
        ]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        output = llm.generate(prompt, sampling_params, use_tqdm=False)

        # output[0].outputs[0].textの最初の文字数字ならばその数字をquestionにセット
        if output[0].outputs[0].text[0].isdigit():
            # 数字が1~7の範囲内かどうか
            if int(output[0].outputs[0].text[0]) < 1 or int(output[0].outputs[0].text[0]) > 7:
                print('Error: 回答が1~7の範囲内にありません。')
                print(f'ID: {question.id}, 出力: {output[0].outputs[0].text}')
            else:
                question.set_question_score(int(output[0].outputs[0].text))
        else:
            print('Error: 回答の最初の文字が数字ではありません。')
            print(f'ID: {question.id}, 出力: {output[0].outputs[0].text}')
    
    # 性格特性ごとにスコアを集計
    scores = {'外向性': 0, '神経症傾向': 0, '開放性': 0, '誠実性': 0, '協調性': 0}
    for question in questions:
        scores[question.category] += question.get_question_score()

    # それぞれの性格特性のスコアの平均をとる
    for key in scores.keys():
        scores[key] = scores[key] / 12

    # 誤差を計算
    true_scores, error, MSE = calculate_error(scores, TARGET_INTERLOCUTOR_ID)

    # 結果をjsonファイルに保存
    with open(f'{OUTPUT_DIR}/BFI_test_only_prompt.json', 'w') as f:
        results.append({'answer': [question.__dict__ for question in questions]})
        results.append({'scores': scores})
        results.append({'true_scores': true_scores})
        results.append({'error': error})
        results.append({'MSE': MSE})
        json.dump(results, f, ensure_ascii=False, indent=2)

if __name__ == '__main__':
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    parser = argparse.ArgumentParser()
    parser.add_argument('--plane_model', action='store_true', default=False, help='If you want to use plane model, please use this option.')
    parser.add_argument('--only_prompt', action='store_true', default=False, help='If you want to output only prompt, please use this option.')
    args = parser.parse_args()

    if args.plane_model:
        calculate_BFI_score_plane_model()
    elif args.only_prompt:
        calculate_BFI_score_only_prompt()
    else:
        calculate_BFI_score()
    print('Done')