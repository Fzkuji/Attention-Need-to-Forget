# 加载模型

from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "Qwen/Qwen3-4B"

# load the tokenizer and the model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)

# prepare the model input - directly input text and tokens
import torch

text = "Give me a short introduction to large language model."
print(f"Input text: {text}")

# tokenize the text to get token IDs
token_ids = tokenizer.encode(text, add_special_tokens=False)
print(f"Token IDs: {token_ids}")

# prepare model inputs directly from token IDs
input_ids = torch.tensor([token_ids], dtype=torch.long).to(model.device)
model_inputs = {"input_ids": input_ids}

# get model output without generation
with torch.no_grad():
    outputs = model(
        **model_inputs,
        output_attentions=True,
        output_hidden_states=True,
    )

# get the output logits (probability distribution)
logits = outputs.logits  # shape: [batch_size, sequence_length, vocab_size]
attentions = outputs.attentions  # tuple of attention weights for each layer
hidden_states = outputs.hidden_states  # tuple of hidden states for each layer
