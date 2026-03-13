from datasets import load_dataset
from transformers import AutoTokenizer

# Load a slice of the test split
dataset = load_dataset("dany0407/eli5_category", split="test[100:200]")

tokenizer = AutoTokenizer.from_pretrained("gpt2")

print(f"{'Index':<6} {'Title tokens':<14} {'Selftext tokens':<16} {'Title (truncated)'}")
print("-" * 80)

for i, example in enumerate(dataset):
    title = example["title"] or ""
    selftext = example["selftext"] or ""

    title_tokens = tokenizer(title, add_special_tokens=False)["input_ids"]
    selftext_tokens = tokenizer(selftext, add_special_tokens=False)["input_ids"]

    print(f"{i:<6} {len(title_tokens):<14} {len(selftext_tokens):<16} {title[:50]!r}")