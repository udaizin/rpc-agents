import json

from datasets import load_dataset


PLANE_MODEL_RESULT_PATH = './BFI/result/Swallow3-8B-plane/BFI_test.json'
BIG_FIVE_JA_EN = {'外向性': 'Extraversion', '神経症傾向': 'Neuroticism', '開放性': 'Openness', '誠実性': 'Conscientiousness', '協調性': 'Agreeableness'}

def calculate_MSE(my_model_result_path, target_interlocutor_id):
    # planeモデルの結果を読み込む
    with open(PLANE_MODEL_RESULT_PATH, 'r', encoding='utf-8') as f:
        plane_result = json.load(f)
    
    # 独自モデルの結果を読み込む
    with open(my_model_result_path, 'r', encoding='utf-8') as f:
        my_model_result = json.load(f)

    # プロンプトのみを指定した場合の結果を読み込む
    with open(f'{my_model_result_path[:-5]}_only_prompt.json', 'r', encoding='utf-8') as f:
        only_prompt_result = json.load(f)
    
    # 対象人物のBFIスコアを読み込む
    interlocutor_dataset = load_dataset("nu-dialogue/real-persona-chat", name='interlocutor', trust_remote_code=True)
    target_interlocutor_data = interlocutor_dataset['train'].filter(lambda x: x['interlocutor_id'] == target_interlocutor_id)
    target_interlocutor_BFI = target_interlocutor_data[0]['personality']

    # planeモデルと対象人物のMSEを計算
    plane_MSE = 0
    for key in BIG_FIVE_JA_EN.keys():
        interlocutor_key = f'BigFive_{BIG_FIVE_JA_EN[key]}'
        plane_MSE += (plane_result[1]['scores'][key] - target_interlocutor_BFI[interlocutor_key]) ** 2
    plane_MSE /= 5

    # 独自モデルと対象人物のMSEを計算
    my_model_MSE = 0
    for key in BIG_FIVE_JA_EN.keys():
        my_model_key = f'BigFive_{BIG_FIVE_JA_EN[key]}'
        my_model_MSE += (my_model_result[1]['scores'][key] - target_interlocutor_BFI[my_model_key]) ** 2
    my_model_MSE /= 5

    # プロンプトのみと対象人物のMSEを計算
    only_prompt_MSE = 0
    for key in BIG_FIVE_JA_EN.keys():
        my_model_key = f'BigFive_{BIG_FIVE_JA_EN[key]}'
        only_prompt_MSE += (only_prompt_result[1]['scores'][key] - target_interlocutor_BFI[my_model_key]) ** 2

    return plane_MSE, my_model_MSE, only_prompt_MSE

if __name__ == '__main__':
    target_interlocutor_id = 'FL'
    my_model_result_path = f'./BFI/result/Swallow3-8B-{target_interlocutor_id}-v2-4e-6/BFI_test.json'
    plane_MSE, my_model_MSE, only_prompt_MSE = calculate_MSE(my_model_result_path, target_interlocutor_id)
    print(f'PlaneモデルとのMSE: {plane_MSE}')
    print(f'独自モデルとのMSE: {my_model_MSE}')
    print(f'プロンプトのみとのMSE: {only_prompt_MSE}')


