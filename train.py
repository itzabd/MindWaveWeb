# deepseek_inference.py
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig

def generate_text(prompt, max_new_tokens=100, temperature=0.7, top_k=50):
    # Load model and tokenizer
    model_name = "deepseek-ai/deepseek-llm-7b-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,  # Use torch.float16 if bfloat16 is unsupported
        device_map="auto",
        # attn_implementation="flash_attention_2"  # Uncomment if flash-attn is installed
    )

    # Configure generation
    generation_config = GenerationConfig.from_pretrained(model_name)
    generation_config.pad_token_id = generation_config.eos_token_id
    generation_config.temperature = temperature
    generation_config.top_k = top_k
    generation_config.do_sample = True  # Enable sampling for creativity

    # Tokenize input and generate
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)

    # Decode only the generated text (exclude input prompt)
    generated_text = tokenizer.decode(
        outputs[0][inputs.input_ids.shape[-1]:],
        skip_special_tokens=True
    )
    return generated_text

if __name__ == "__main__":
    # Example usage
    prompt = "An attention function can be described as mapping a query and a set of key-value pairs to an output, where the query, keys, values, and output are all vectors. The output is"
    generated_text = generate_text(prompt)
    print("Generated Text:")
    print(generated_text)