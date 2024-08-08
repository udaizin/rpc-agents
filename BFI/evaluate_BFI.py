import json

import pandas as pd
from datasets import load_dataset
import japanize_matplotlib
from pycirclize import Circos


TARGET_INTERLOCUTOR_ID = 'CP'
# TODO: マルチターンのfew-shotかどうかで代わる。
PLANE_MODEL_RESULT_PATH = './BFI/result/tmp/Swallow3-8B-plane/BFI_test.json'
BIG_FIVE_JA_EN = {'外向性': 'Extraversion', '神経症傾向': 'Neuroticism', '開放性': 'Openness', '誠実性': 'Conscientiousness', '協調性': 'Agreeableness'}


def plot_radar_chart(my_model_result, only_prompt_result, plane_result, target_interlocutor_BFI):
    target_interlocutor_BFI = {
        '外向性': target_interlocutor_BFI['BigFive_Extraversion'], 
        '神経症傾向': target_interlocutor_BFI['BigFive_Neuroticism'], 
        '開放性': target_interlocutor_BFI['BigFive_Openness'], 
        '誠実性': target_interlocutor_BFI['BigFive_Conscientiousness'], 
        '協調性': target_interlocutor_BFI['BigFive_Agreeableness']
    }
    # 1つのpd.DataFrameにまとめる
    BFI_df = pd.DataFrame({
        '外向性': [my_model_result[1]['scores']['外向性'], only_prompt_result[1]['scores']['外向性'], plane_result[1]['scores']['外向性'], target_interlocutor_BFI['外向性']],
        '神経症傾向': [my_model_result[1]['scores']['神経症傾向'], only_prompt_result[1]['scores']['神経症傾向'], plane_result[1]['scores']['神経症傾向'], target_interlocutor_BFI['神経症傾向']],
        '開放性': [my_model_result[1]['scores']['開放性'], only_prompt_result[1]['scores']['開放性'], plane_result[1]['scores']['開放性'], target_interlocutor_BFI['開放性']],
        '誠実性': [my_model_result[1]['scores']['誠実性'], only_prompt_result[1]['scores']['誠実性'], plane_result[1]['scores']['誠実性'], target_interlocutor_BFI['誠実性']],
        '協調性': [my_model_result[1]['scores']['協調性'], only_prompt_result[1]['scores']['協調性'], plane_result[1]['scores']['協調性'], target_interlocutor_BFI['協調性']],
    }, index=['SFT済みモデル', '性格特性プロンプト', 'プレーンモデル', '対象人物の真のスコア'])

    circos = Circos.radar_chart(
        BFI_df,
        vmax=7,
        marker_size=6,
        circular=True,
        cmap="Set2",
        grid_interval_ratio=0.25,
    )
    # Plot figure & set legend on upper right
    fig = circos.plotfig()
    _ = circos.ax.legend(loc="upper right")
    fig.savefig(f"./BFI/radar_chart/{TARGET_INTERLOCUTOR_ID}-v2-2e-5.png")



def calculate_MSE(my_model_result_path, target_interlocutor_id):
    # planeモデルの結果を読み込む
    with open(PLANE_MODEL_RESULT_PATH, 'r', encoding='utf-8') as f:
        plane_result = json.load(f)
    
    # 独自モデルの結果を読み込む
    with open(my_model_result_path, 'r', encoding='utf-8') as f:
        my_model_result = json.load(f)
    
    print(my_model_result[1])

    # プロンプトのみを指定した場合の結果を読み込む
    with open(f'{my_model_result_path[:-5]}_only_prompt.json', 'r', encoding='utf-8') as f:
        only_prompt_result = json.load(f)
    
    # 対象人物のBFIスコアを読み込む
    interlocutor_dataset = load_dataset("nu-dialogue/real-persona-chat", name='interlocutor', trust_remote_code=True)
    target_interlocutor_data = interlocutor_dataset['train'].filter(lambda x: x['interlocutor_id'] == target_interlocutor_id)
    target_interlocutor_BFI = target_interlocutor_data[0]['personality']


    # レーダーチャートをプロット
    plot_radar_chart(my_model_result, only_prompt_result, plane_result, target_interlocutor_BFI)

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
    only_prompt_MSE /= 5

    return plane_MSE, my_model_MSE, only_prompt_MSE


    

if __name__ == '__main__':
    my_model_result_path = f'./BFI/result/tmp/Swallow3-8B-{TARGET_INTERLOCUTOR_ID}-v2-2e-5/BFI_test.json'
    plane_MSE, my_model_MSE, only_prompt_MSE = calculate_MSE(my_model_result_path, TARGET_INTERLOCUTOR_ID)
    print(f'PlaneモデルとのMSE: {plane_MSE}')
    print(f'独自モデルとのMSE: {my_model_MSE}')
    print(f'プロンプトのみとのMSE: {only_prompt_MSE}')


