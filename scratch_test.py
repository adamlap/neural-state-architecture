import sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_id = "Qwen/Qwen2.5-0.5B-Instruct" if len(sys.argv) < 2 else sys.argv[1]
print(f"Loading {model_id} via HuggingFace on CPU...")

tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float32,
    trust_remote_code=True,
)
model.eval()
print(f"Model {model_id} loaded successfully!")

prompt = "You are a cloud DevOps agent. The cluster has high latency. Respond with a JSON object specifying your reasoning and proposed diagnostic tool."
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=64, temperature=0.2, do_sample=True)
gen_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("--- MODEL OUTPUT ---")
print(gen_text)
print("--------------------")
