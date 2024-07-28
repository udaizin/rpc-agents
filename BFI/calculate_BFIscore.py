from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from vllm import LLM, SamplingParams
from tqdm import tqdm


model_name = './models/Swallow3-8B-FL-lr-5e-6'
sampling_params = SamplingParams(
    temperature=0, max_tokens=5, stop="<|eot_id|>"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)
llm = LLM(model_name, tensor_parallel_size=1)
OUTPUT_DIR = './BFI/result/Swallow3-8B-FL-lr-5e-6'

question_num_detail_dict = {
    1: '話し好き', 2: '無口な', 3: '陽気な', 4: '外交的', 5: '暗い', 6: '無愛想な', 7: '社交的', 8: '人嫌い', 9: '活動的な', 10: '意思表示しない', 11: '積極的な', 12: '地味な', # 外向性
    13: '悩みがち', 14: '不安になりやすい', 15: '心配性', 16: '気苦労の多い', 17: '弱気になる', 18: '傷つきやすい', 19: '動揺しやすい', 20: '神経質な', 21: 'くよくよしない', 22: '悲観的な', 23: '緊張しやすい', 24: '憂鬱な', # 神経症傾向
    25: '独創的な', 26: '多才の', 27: '進歩的', 28: '洞察力のある', 29: '想像力に富んだ', 30: '美的感覚の鋭い', 31: '頭の回転の速い', 32: '臨機応変な', 33: '興味の広い', 34: '好奇心が強い', 35: '独立した', 36: '呑み込みの速い', # 開放性
    37: 'いい加減な', 38: 'ルーズな', 39: '怠惰な', 40: '成り行きまかせ', 41: '不精な', 42: '計画性のある', 43: '無頓着な', 44: '軽率な', 45: '勤勉な', 46: '無節操', 47: '几帳面な', 48: '飽きっぽい', # 誠実性
    49: '温和な', 50: '短期', 51: '怒りっぽい', 52: '寛大な', 53: '親切な', 54: '良心的な', 55: '協力的な', 56: 'とげがある', 57: 'かんしゃくもち', 58: '自己中心的', 59: '素直な', 60: '反抗的' # 協調性
}

# 反転項目のリスト(リストの数字は上の辞書のキーに対応している)
reverse_items = [2, 5, 6, 8, 19, 12, 21, 42, 45, 47, 50, 51, 56, 57, 58, 60]

class Question():
    def __init__(self, id, detail):
        self.id = id
        self.detail = detail
        self.is_reverse = id in reverse_items
        self.score = None
    
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
        Question(1, "(a) 話し好き")
        Question(2, "(b) 無口な")
        :
        :
        Question(60, "(bh) 反抗的")
    ]
'''
def get_question_list():
    questions = []
    for key, value in question_num_detail_dict.items():
        if key <= 26:
            alphabet = chr(96+key)
        elif key <= 52:
            alphabet = f"a{chr(96+key-26)}"
        else:
            alphabet = f"b{chr(96+key-52)}"
        questions.append(Question(key, f"({alphabet}) {value}"))
    return questions


def calculate_BFI_score():
    results = []
    questions = get_question_list()
    for question in tqdm(questions):

        BFI_test_prompt = '\n'.join(['いまからあなたに性格に関する質問をします。以下の質問に対して、それぞれどのくらい当てはまるかを次の尺度で評価してください。回答するときは問題番号とその横に該当する数字のみを入力してください。',
            '',
            '1: まったくあてはまらない',
            '2: ほとんどあてはまらない',
            '3: あまりあてはまらない',
            '4: どちらとも言えない',
            '5: ややあてはまる',
            '6: かなりあてはまる',
            '7: 非常にあてはまる',
            '',
            '質問:',
            '(A) りんごは赤い',
            '回答:',
            '(A) 7',
            '',
            '質問:',
            f'{question.detail}',
            '回答:'
        ])

        message = [
            {"role": "system", "content": "あなたにはこれからアンケートに受けてもらいます。問題番号と数値のみで回答してください。"},
            {
                "role": "user",
                "content": BFI_test_prompt,
            },
        ]
        prompt = tokenizer.apply_chat_template(
            message, tokenize=False, add_generation_prompt=True
        )

        output = llm.generate(prompt, sampling_params)

        # output[0].outputs[0].textの最初の行をresultsに追加
        results.append(output[0].outputs[0].text.split('\n')[0])

        # output[0].outputs[0].textの最初の行の最後のもじが数字ならその数字をquestionにセット
        if output[0].outputs[0].text.split('\n')[0][-1].isdigit():
            question.set_question_score(int(output[0].outputs[0].text.split('\n')[0][-1]))
        else:
            print('Error: output[0].outputs[0].textの最初の行の最後のもじが数字ではありません')
        
    # resultsの結果を要素ごとに改行して./BFI/result/sample.txtに保存
    with open(f'{OUTPUT_DIR}/questionnaire.txt', 'w') as f:
        for result in results:
            f.write(result+'\n')
    
    # 性格特性ごとにスコアを集計
    scores = {'外向性': 0, '神経症傾向': 0, '開放性': 0, '誠実性': 0, '協調性': 0}
    for question in questions:
        if question.id <= 12:
            scores['外向性'] += question.get_question_score()
        elif question.id <= 24:
            scores['神経症傾向'] += question.get_question_score()
        elif question.id <= 36:
            scores['開放性'] += question.get_question_score()
        elif question.id <= 48:
            scores['誠実性'] += question.get_question_score()
        else:
            scores['協調性'] += question.get_question_score()

    # それぞれの性格特性のスコアの平均をとる
    for key in scores.keys():
        scores[key] = scores[key] / 12

    # スコアをOUTPUT_DIR/scores.txtに保存
    with open(f'{OUTPUT_DIR}/scores.txt', 'w') as f:
        for key, value in scores.items():
            f.write(f'{key}: {value}\n')

    return scores

if __name__ == '__main__':
    calculate_BFI_score()
    print('Done')