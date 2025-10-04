# pip install transformers
# ! pip install torch
# ! pip install hf_xet

from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

inputs = tokenizer("Create a poem on an adult boy trying to learn piano", return_tensors="pt")

outputs = model.generate(
    **inputs,
    max_new_tokens=100,   # allow a longer poem
    do_sample=True,       # enable sampling
    temperature=0.8,      # creativity (higher = more random, 0.7–1.0 is good)
    top_p=0.9             # nucleus sampling, keeps the most likely 90% of probability mass
)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))
