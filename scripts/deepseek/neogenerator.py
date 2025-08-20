# -*- coding: utf-8 -*-
"""
DeepSeek-R1-Distill-Llama-8B — soft-cap <think> tokens + safe saves
Transformers >= 4.53.0
"""

from pathlib import Path
import json
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    LogitsProcessor,
    LogitsProcessorList,
    NoBadWordsLogitsProcessor,
    BitsAndBytesConfig,
)

# =========================
# Config knobs (edit here)
# =========================
MODEL_ID = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
INPUT_JSON = "deepseek-trial.json"
OUTPUT_JSON = "datasets/CodeForce/inference/DeepSeek8B/trial.json"

MAX_NEW_TOKENS = 2048       # total generation budget
TEMP = 0.2                  # a touch higher helps accept the closing tag
TOP_P = 0.9

# Soft-cap params
MAX_THINK_TOKENS = 128      # how many tokens you're willing to spend inside <think>
END_BOOST = 12.0            # push toward the first token of </think>
OTHER_PENALTY = 3.0         # gently penalize "keep thinking" tokens after cap

BAN_REOPEN = True           # prevent starting a second <think> after it closes


# =========================
# Logits Processor (Soft Cap)
# =========================
class CapThink(LogitsProcessor):
    """
    Soft-cap <think>...</think>. After MAX_THINK_TOKENS inside the block, we:
      1) Boost the first token of </think>
      2) Penalize all other tokens (lightly) to encourage closing
    Works even if <think> / </think> are multi-token sequences (we match on ids).
    """
    def __init__(self, start_ids, end_ids, max_think_tokens=128,
                 end_boost=12.0, other_penalty=3.0):
        self.start_ids = start_ids or []
        self.end_ids = end_ids or []
        self.max = max_think_tokens
        self.end_boost = end_boost
        self.other_pen = other_penalty
        self.in_think = False
        self.k = 0

    @staticmethod
    def _endswith(seq, suffix):
        return len(seq) >= len(suffix) and seq[-len(suffix):] == suffix

    def __call__(self, input_ids, scores):
        # single-batch assumed; extend to multi-batch if needed
        ids = input_ids[0].tolist()

        # Track entering/exiting the think block
        if self.start_ids and self._endswith(ids, self.start_ids):
            self.in_think, self.k = True, 0
        if self.end_ids and self._endswith(ids, self.end_ids):
            self.in_think = False

        if self.in_think and self.k >= self.max and self.end_ids:
            first_end = self.end_ids[0]
            # 1) push toward closing token
            scores[:, first_end] = scores[:, first_end] + self.end_boost
            # 2) gently penalize everything else so it picks </think> sooner
            mask = torch.ones_like(scores, dtype=torch.bool)
            mask[:, first_end] = False
            scores[mask] -= self.other_pen

        if self.in_think:
            self.k += 1
        return scores


def main():
    # -------------------------
    # Load data
    # -------------------------
    with open(INPUT_JSON, "r") as f:
        data = json.load(f)

    # -------------------------
    # Model / tokenizer
    # -------------------------
    quant = BitsAndBytesConfig(load_in_8bit=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        device_map="auto",
        quantization_config=quant,
    )

    # Prevent hidden 1024 caps
    # (Let per-call generate() control MAX_NEW_TOKENS)
    gen_cfg = model.generation_config
    gen_cfg.max_length = 32000
    gen_cfg.max_new_tokens = None
    model.generation_config = gen_cfg

    # Map tags to token-id sequences (robust if they’re multi-token)
    think_start_ids = tokenizer.encode("<think>", add_special_tokens=False)
    think_end_ids = tokenizer.encode("</think>", add_special_tokens=False)

    cap_proc = CapThink(
        start_ids=think_start_ids,
        end_ids=think_end_ids,
        max_think_tokens=MAX_THINK_TOKENS,
        end_boost=END_BOOST,
        other_penalty=OTHER_PENALTY,
    )

    procs = [cap_proc]
    if BAN_REOPEN and think_start_ids:
        procs.append(
            NoBadWordsLogitsProcessor(
                bad_words_ids=[think_start_ids],
                eos_token_id=tokenizer.eos_token_id,
            )
        )
    logits_processors = LogitsProcessorList(procs)

    # Ensure output dir exists
    out_path = Path(OUTPUT_JSON)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # -------------------------
    # Inference loop
    # -------------------------
    for idx, item in enumerate(data):
        if item.get("outputs"):
            print(f"{idx} already processed. Skipping.")
            continue

        outputs = []

        for jdx, prompt_text in enumerate(item["problem_statements"]):
            chat = [
                {
                    "role": "user",
                    "content": (
                        "You are very knowledgeable. Think briefly inside <think> tags "
                        f"(max ~30 words), then provide the final answer outside the tags.\n\n{prompt_text}"
                    ),
                }
            ]

            prompt = tokenizer.apply_chat_template(
                chat, tokenize=False, add_generation_prompt=True
            )
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

            gen = model.generate(
                **inputs,
                do_sample=True,
                temperature=TEMP,
                top_p=TOP_P,
                max_new_tokens=MAX_NEW_TOKENS,
                logits_processor=logits_processors,
                pad_token_id=tokenizer.eos_token_id,
                return_dict_in_generate=True,
            )

            seq = gen.sequences[0]
            generated_text = tokenizer.decode(
                seq[inputs.input_ids.shape[-1]:], skip_special_tokens=True
            )
            outputs.append(generated_text)

            tokens_generated = seq.shape[-1] - inputs.input_ids.shape[-1]
            print(f"Problem {idx}.{jdx+1}: Generated {tokens_generated} tokens")

            if "</think>" not in generated_text and think_end_ids:
                print(
                    "  Warning: reasoning block may not have closed — "
                    "consider raising END_BOOST, raising OTHER_PENALTY, or lowering TEMP."
                )

        item["outputs"] = outputs

        # Save after each item to be safe
        with out_path.open("w") as f:
            json.dump(data, f, indent=4)
        print(f"Saved at index {idx}.")

    # Final save
    with out_path.open("w") as f:
        json.dump(data, f, indent=4)
    print("Final save complete.")


if __name__ == "__main__":
    main()
