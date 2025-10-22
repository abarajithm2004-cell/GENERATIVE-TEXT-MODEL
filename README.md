from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load pre-trained GPT-2 model and tokenizer
model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

def generate_text(prompt, max_length=120):
    inputs = tokenizer.encode(prompt, return_tensors='pt')
    outputs = model.generate(
        inputs,
        max_length=max_length,
        num_return_sequences=1,
        no_repeat_ngram_size=2,
        temperature=0.8,
        top_p=0.9,
        do_sample=True,
        top_k=50
    )
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return text

# Example prompt
if __name__ == "__main__":
    topic = input("Enter a topic to generate text about: ")
    print("\n🧠 Generating text...\n")
    generated = generate_text(f"Write a paragraph about {topic}:")
    print(generated)
