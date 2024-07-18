import os
import json
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
        utterances_list = ['\n'.join(dialogue['utterances']) for dialogue in dialogues]
        dialogue_id_list = [dialogue['dialogue_id'] for dialogue in dialogues]
        interlocutors_list = [dialogue['interlocutors'] for dialogue in dialogues]
        partner_interlocutor_id_list = [interlocutors[0] if interlocutors[1] == target_interlocutor_id else interlocutors[1] for interlocutors in interlocutors_list]

        dialogues_first_person_list = []
        for utterances, dialogue_id, partner_interlocutor_id in tzip(utterances_list, dialogue_id_list, partner_interlocutor_id_list):
            # インデントなどを整えるため、'\n'でjoin
            system_prompt = '\n'.join([
                f'あなたは{target_interlocutor_id}です。あなたの性別やペルソナ、性格特性は次のようになっています。',
                f'性別: {target_interlocutor_gender_jp}',
                f'ペルソナ: {target_interlocutor_persona_prompt}',
                f'性格特性: {target_interlocutor_personality_prompt}',
                '',          
                'ルール:',
                f'1. {target_interlocutor_id}には感情と思考能力があります。{target_interlocutor_id}が何を感じ、何を考えているのか慎重に考えてください。',
                f'2. あなたは今、ペルソナや性格特性に基づいて、対話履歴を{target_interlocutor_id}の内心描写を追加することを目的としています。',
                f'3. 主人公は{target_interlocutor_id}です。対話履歴の中に、{target_interlocutor_id}が何かを感じたり考えたりしたと思うところに、(thinking)のラベルを用いて{target_interlocutor_id}の気持ちや考えを挿入してください。',
                f'4. (speaking)のラベルの内容は「絶対に」書き換えないでそのまま残してください。ただし、{target_interlocutor_id}の思考があったと思うときに(speaking)のラベルの前か後ろに、(thinking)のラベルを挿入することができます。',
                f'5. Big Five性格特性やペルソナを考慮して、{target_interlocutor_id}の感情や思考を表現してください。'
                f'6. {target_interlocutor_id} (thinking): 〜 という形式に従って、{target_interlocutor_id}の内心描写を追加してください。',
            ])
            utterances_example = '\n'.join([
                f'{partner_interlocutor_id} (speaking): こんにちは。',
                f'{target_interlocutor_id} (speaking): こんにちは。最近暑くなってきましたよね。',
                f'{partner_interlocutor_id} (speaking): はい、そうですね。でも、夏が好きなので、暑いのは嬉しいです。',
                f'{target_interlocutor_id} (speaking): そうなんですね。私は暑いのはあまり得意じゃないんです。',
                f'{partner_interlocutor_id} (speaking): そうなんですか。じゃあ、夏はあまり好きじゃないんですね。',
                f'{target_interlocutor_id} (speaking): 夏はあまり好きじゃないんですけど、冬も寒いからどっちもどっちかな笑',
                f'{partner_interlocutor_id} (speaking): まあ、春とか秋の方が気温的にはすごしやすいですね。',
                f'{target_interlocutor_id} (speaking): お花見とか好きなので、春が待ち遠しいです。',
                f'{partner_interlocutor_id} (speaking): お花見いいですね。私も大好きです',
                f'{target_interlocutor_id} (speaking): もし良かったら来年一緒にお花見しませんか？',
                f'{partner_interlocutor_id} (speaking): いいですね〜 来年いきましょう',
                f'{target_interlocutor_id} (speaking): 楽しみにしてますね！'
            ])                       
            user_first_prompt = '\n'.join([
                f'対話履歴が以下の通り与えられるとき、対話履歴を読んで{target_interlocutor_id}の内心描写を追加してください。',
                f'対話履歴:',
                f'{utterances_example}'
            ])
            assistant_prompt = '\n'.join([
                f'{partner_interlocutor_id} (speaking): こんにちは。',
                f'{target_interlocutor_id} (speaking): こんにちは。最近暑くなってきましたよね。',
                f'{partner_interlocutor_id} (speaking): はい、そうですね。でも、夏が好きなので、暑いのは嬉しいです。',
                f'{target_interlocutor_id} (thinking): 暑いのが好きなんだ。私はあんまり好きじゃないな〜',
                f'{target_interlocutor_id} (speaking): そうなんですね。私は暑いのはあまり得意じゃないんです。',
                f'{partner_interlocutor_id} (speaking): そうなんですか。じゃあ、夏はあまり好きじゃないんですね。',
                f'{target_interlocutor_id} (speaking): 夏はあまり好きじゃないんですけど、冬も寒いからどっちもどっちかな笑',
                f'{partner_interlocutor_id} (speaking): まあ、春とか秋の方が気温的にはすごしやすいですね。',
                f'{target_interlocutor_id} (thinking): 春は特に好きだな。桜とかも見れるし',
                f'{target_interlocutor_id} (speaking): お花見とか好きなので、春が待ち遠しいです。',
                f'{target_interlocutor_id} (thinking): <{partner_interlocutor_id}>さんお花見とか好きなのかな？好きだったら来年一緒に行きたいな。',
                f'{partner_interlocutor_id} (speaking): お花見いいですね。私も大好きです',
                f'{target_interlocutor_id} (speaking): もし良かったら来年一緒にお花見しませんか？',
                f'{target_interlocutor_id} (thinking): OKしてくれるかな〜',
                f'{partner_interlocutor_id} (speaking): いいですね〜 来年いきましょう',
                f'{target_interlocutor_id} (speaking): 楽しみにしてますね！',
                f'{target_interlocutor_id} (thinking): OKもらえて良かった。'
            ])
            user_second_prompt = '\n'.join([
                f'対話履歴が以下の通り与えられるとき、「性格特性」と「ペルソナ」を忠実に反映して{target_interlocutor_id}の内心描写を追加してください。',
                f'対話履歴:',
                f'{utterances}'
            ])    

            completion = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_first_prompt},
                    {"role": "assistant", "content": assistant_prompt},
                    {"role": "user", "content": user_second_prompt}
                ],
            )

            dialogue_first_person_dict = {
                "dialogue_id": dialogue_id,
                "utterances": utterances,
                "first_person_dialogue": completion.choices[0].message.content
            }
            dialogues_first_person_list.append(dialogue_first_person_dict)

        # json形式で保存
        with open(f'./RealPersonaChat/data/gen_inner_monologue/{target_interlocutor_id}_inner_monologue.json', 'w') as f:
            json.dump(dialogues_first_person_list, f, ensure_ascii=False, indent=2)