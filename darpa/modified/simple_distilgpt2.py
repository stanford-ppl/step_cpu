#!/usr/bin/env python
# --------------------------------------------------------------
#  Benchmark – fine‑tune a tiny GPT‑2 model on the ELI5 dataset
#
#  Required: torch, transformers, datasets
#  Optional: psutil (for CPU % in table)
#            tabulate (for pretty table formatting)
# --------------------------------------------------------------

import argparse
import os
import math
import time
import warnings
import threading
from collections import namedtuple
from typing import List, Dict

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer,
)


# --------------------------------------------------------------
#  Silence irrelevant warnings
# --------------------------------------------------------------
warnings.filterwarnings(
    "ignore",
    message="loss_type=None was set in the config but it is unrecognised.*",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    module="torch.cuda",
)


def main(prompt="Why is the sky blue?"):
    # --------------------------------------------------------------
    # Overall wall‑clock start
    # --------------------------------------------------------------
    overall_start = time.perf_counter()

    # --------------------------------------------------------------
    # System info
    # --------------------------------------------------------------
    print("\n=== System resources ===")
    print(f"Available CPUs : {os.cpu_count()}")
    # --------------------------------------------------------------
    # Inference mode
    # --------------------------------------------------------------

    model_id = "distilbert/distilgpt2"

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_id)
    # GPT2LMHeadModel(GPT2PreTrainedModel, GenerationMixin)
    device = torch.device("cpu")
    model.to(device)

    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    # inputs: {
    #   'input_ids': tensor([[5195,  318,  262, 6766, 4171,   30]]),
    #   'attention_mask': tensor([[1, 1, 1, 1, 1, 1]])}
    # }

    model.eval()
    with torch.no_grad():
        # inputs["input_ids"] = tensor([[5195,  318,  262, 6766, 4171,   30]])
        # shape = [1, 6]
        output_ids = model.generate(
            inputs["input_ids"],
            max_new_tokens=20,
            do_sample=False,
        )
    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    # print(f"\n=== Generated Text ===\n{generated_text}\n")

    overall_end = time.perf_counter()
    total_elapsed = overall_end - overall_start

    print("\n=== Total Runtime ===")
    print(f"{total_elapsed:.2f} seconds")


if __name__ == "__main__":
    main()
