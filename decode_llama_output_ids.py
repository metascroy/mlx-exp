#!/usr/bin/env python3
import os
from transformers import AutoTokenizer

MODEL_ID = os.getenv("MODEL_ID", "unsloth/Llama-3.2-1B-Instruct")
OUTPUT_IDS = os.getenv("OUTPUT_IDS", "output_ids.txt")

def read_token_ids(path: str):
    with open(path, "r") as f:
        text = f.read().strip()
    if not text:
      raise ValueError(f"{path} is empty")
    return [int(tok) for tok in text.split()]

def main():
    ids = read_token_ids(OUTPUT_IDS)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    text = tokenizer.decode(ids, skip_special_tokens=False)
    print(text)

if __name__ == "__main__":
    main()
