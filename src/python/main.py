import qa_benchmark as qa
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

def load_model(model_name="gpt2"):
    print(f"Loading model '{model_name}'...")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    print("Moving model to device...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    print("Model loaded successfully!")
    return tokenizer, model, device

tokenizer, model, device = load_model("TinyLlama/TinyLlama-1.1B-Chat-v1.0")

benchmark = qa.QABenchmark("strasbourg_qa.db")
results = benchmark.evaluate_all(tokenizer, model, device, batch_size=10)

for q, expected, generated in results:
    print(f"Q: {q}")
    print(f"Expected: {expected}")
    print(f"Generated: {generated}")
    print("-" * 50)