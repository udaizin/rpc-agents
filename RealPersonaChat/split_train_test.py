
import os
import json
import random

from dotenv import load_dotenv, find_dotenv

# 乱数のシードを固定
random.seed(42)

_ = load_dotenv(find_dotenv(), override=True)
TARGET_INTERLOCUTOR_IDS = os.getenv("TARGET_INTERLOCUTOR_IDS").split(',')

def split_train_test(data, target_interlocutor_id, test_size=0.1):
    n_data = len(data)
    n_test = int(n_data * test_size)
    n_train = n_data - n_test
    # dataをランダムにシャッフル
    random.shuffle(data)
    train_data = data[:n_train]
    test_data = data[n_train:]
    with open(f'./RealPersonaChat/data/train_data/{target_interlocutor_id}_inner_monologue_train.json', 'w') as f_train:
        json.dump(train_data, f_train, ensure_ascii=False, indent=2)
    with open(f'./RealPersonaChat/data/test_data/{target_interlocutor_id}_inner_monologue_test.json', 'w') as f_test:
        json.dump(test_data, f_test, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    for target_interlocutor_id in TARGET_INTERLOCUTOR_IDS:
        with open(f'./RealPersonaChat/data/gen_inner_monologue/{target_interlocutor_id}_inner_monologue.json', 'r', encoding='utf-8') as f:
            dialogues = json.load(f)
            split_train_test(dialogues, target_interlocutor_id)