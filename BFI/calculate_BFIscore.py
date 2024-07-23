import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from BFI_prompt import BFI_test_prompt


device = "cuda" if torch.cuda.is_available() else "cpu"

model_name = 'tokyotech-llm/Llama-3-Swallow-8B-Instruct-v0.1'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

def test():
    inputs = tokenizer(BFI_test_prompt, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    # max_lengthを4137にしないと余計なトークンが生成される
    output = model.generate(input_ids, max_length=4137, attention_mask=attention_mask, pad_token_id=tokenizer.eos_token_id, temperature=0.7, do_sample=True)
    # output = model.generate(input_ids, max_length=4137, attention_mask=attention_mask, pad_token_id=tokenizer.eos_token_id)

    # プロンプトに対応している部分を除いてデコード
    prompt_length = input_ids.shape[1]
    generated_text = tokenizer.decode(output[0][prompt_length:], skip_special_tokens=True)
    
    return generated_text

print(test())