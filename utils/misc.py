import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from optimum.bettertransformer import BetterTransformer

device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("microsoft/phi-2", torch_dtype=torch_dtype, device_map=device, trust_remote_code=True)

model.to(device)
# model = model.to_bettertransformer() not available for phi

pipe = pipeline(
    'text-generation',
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=512,
    do_sample=True,
    temperature=0.3
)

print("STARTING INFERENCE")
pipe("Who are you? Are you working?")