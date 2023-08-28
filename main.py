import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
tokenizer = AutoTokenizer.from_pretrained("NumbersStation/nsql-llama-2-7B")
model = AutoModelForCausalLM.from_pretrained("NumbersStation/nsql-llama-2-7B", torch_dtype=torch.bfloat16).cuda()

text = "Generate a Python function to calculate the factorial of a number."

input_ids = tokenizer(text, return_tensors="pt").input_ids.cuda()

generated_ids = model.generate(input_ids, max_length=500, do_sample=False,temperature=1.0)
print(tokenizer.decode(generated_ids[0], skip_special_tokens=True))


