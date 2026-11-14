"""QLoRA fine-tuning of Qwen2.5-Coder on the Cube tool-calling data (data/sft/sft_tools_*.jsonl).

The loss covers only the assistant turns (the tool calls), not the system prompt, the question or
the tool results. Prompts are rendered with the model's own chat template and the same tool
definitions the harness sends, so training matches what the model sees at run time.

    pip install -r requirements-train.txt
    python scripts/train_lora.py --base Qwen/Qwen2.5-Coder-7B-Instruct --out lora/qwen7b --dry-run
    python scripts/train_lora.py --base Qwen/Qwen2.5-Coder-7B-Instruct --out lora/qwen7b
    python scripts/train_lora.py --base Qwen/Qwen2.5-Coder-14B-Instruct --out lora/qwen14b

Then scripts/export_ollama.py turns the adapter into an Ollama model for harness.evaluate.
"""
import argparse
import json
import math
import random
from pathlib import Path


def load_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in open(path, encoding="utf-8")]


def as_template_messages(messages: list[dict]) -> list[dict]:
    """Qwen's chat template serialises tool-call arguments itself, so pass them as objects."""
    out = []
    for m in messages:
        m = dict(m)
        if m.get("tool_calls"):
            m["tool_calls"] = [{**tc, "function": {**tc["function"],
                                                   "arguments": json.loads(tc["function"]["arguments"])}}
                               for tc in m["tool_calls"]]
        out.append(m)
    return out


def encode(tok, row: dict, max_len: int) -> dict | None:
    """input_ids with labels set only on assistant tokens; None if the example is too long."""
    msgs = as_template_messages(row["messages"])
    tools = row.get("tools")
    full = tok.apply_chat_template(msgs, tools=tools, tokenize=False)
    ids = tok(full, add_special_tokens=False)["input_ids"]
    if len(ids) > max_len:
        return None
    labels = [-100] * len(ids)
    for i, m in enumerate(msgs):
        if m["role"] != "assistant":
            continue
        # tokens from "<|im_start|>assistant\n" up to and including this turn's "<|im_end|>"
        start_text = tok.apply_chat_template(msgs[:i], tools=tools, tokenize=False, add_generation_prompt=True)
        end_text = tok.apply_chat_template(msgs[: i + 1], tools=tools, tokenize=False)
        assert full.startswith(start_text) and full.startswith(end_text), "chat template is not prefix-stable"
        a = len(tok(start_text, add_special_tokens=False)["input_ids"])
        b = len(tok(end_text, add_special_tokens=False)["input_ids"])
        for j in range(a, min(b, len(ids))):
            labels[j] = ids[j]
    return {"input_ids": ids, "attention_mask": [1] * len(ids), "labels": labels}


def labelled_loss(model, inputs):
    """Cross-entropy over the labelled (assistant) positions only.

    The usual path builds logits for all positions x 152k vocabulary in fp32, about 2.7 GB per
    1k tokens, although only ~200 tokens per example carry a label. Applying lm_head to the
    labelled positions alone gives the same loss for a fraction of the memory."""
    import torch.nn.functional as F
    base = model.get_base_model()
    mask = inputs["attention_mask"]
    mask = None if bool(mask.all()) else mask  # an all-ones mask makes SDPA build the full L x L matrix
    hidden = base.model(input_ids=inputs["input_ids"], attention_mask=mask).last_hidden_state
    labels = inputs["labels"][:, 1:]
    keep = labels != -100
    logits = base.lm_head(hidden[:, :-1][keep]).float()
    return F.cross_entropy(logits, labels[keep])


def make_trainer_class():
    from transformers import Trainer

    class LabelledLossTrainer(Trainer):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            # compute_loss returns a per-micro-batch mean, so let Trainer divide by grad accumulation
            # (otherwise it assumes the loss is already normalised by num_items_in_batch).
            self.model_accepts_loss_kwargs = False

        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            loss = labelled_loss(model, inputs)
            return (loss, {"loss": loss}) if return_outputs else loss  # Trainer.prediction_step indexes outputs

    return LabelledLossTrainer


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen2.5-Coder-7B-Instruct")
    ap.add_argument("--data", type=Path, default=Path("data/sft"))
    ap.add_argument("--mode", choices=["tools", "single"], default="tools")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max-len", type=int, default=8192, help="longer examples are dropped; 8192 fits the 7B on 16 GB")
    ap.add_argument("--epochs", type=float, default=2)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=16)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--resume", action="store_true", help="continue from the last checkpoint in --out")
    ap.add_argument("--dry-run", action="store_true", help="tokenise, show one masked example, and stop")
    args = ap.parse_args()

    import torch
    from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
                              DataCollatorForSeq2Seq, TrainingArguments)
    LabelledLossTrainer = make_trainer_class()

    random.seed(args.seed)
    tok = AutoTokenizer.from_pretrained(args.base)
    data = {}
    for split in ("train", "val"):
        rows = load_rows(args.data / f"sft_{args.mode}_{split}.jsonl")
        enc = [e for e in (encode(tok, r, args.max_len) for r in rows) if e]
        data[split] = enc
        print(f"{split}: {len(enc)}/{len(rows)} examples fit {args.max_len} tokens")

    ex = data["train"][0]
    shown = tok.decode([t for t, l in zip(ex["input_ids"], ex["labels"]) if l != -100])
    print("trained-on text of example 0:\n" + shown)
    if args.dry_run:
        return

    model = AutoModelForCausalLM.from_pretrained(
        args.base, device_map={"": 0}, torch_dtype=torch.bfloat16, attn_implementation="sdpa",
        quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                               bnb_4bit_compute_dtype=torch.bfloat16,
                                               bnb_4bit_use_double_quant=True))
    from peft import LoraConfig, get_peft_model
    # Not prepare_model_for_kbit_training: it upcasts embeddings and lm_head to fp32 (+2 GB on 7B).
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model.config.use_cache = False
    model = get_peft_model(model, LoraConfig(
        r=args.rank, lora_alpha=args.alpha, lora_dropout=0.05, task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
    model.print_trainable_parameters()

    steps_per_epoch = math.ceil(len(data["train"]) / (args.batch * args.grad_accum))
    targs = TrainingArguments(
        output_dir=str(args.out), num_train_epochs=args.epochs, learning_rate=args.lr,
        per_device_train_batch_size=args.batch, per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.grad_accum, lr_scheduler_type="cosine", warmup_ratio=0.03,
        bf16=True, logging_steps=10, eval_strategy="steps", eval_steps=max(10, steps_per_epoch // 2),
        save_strategy="steps", save_steps=max(10, steps_per_epoch // 2), save_total_limit=2,
        load_best_model_at_end=True, metric_for_best_model="eval_loss",
        optim="paged_adamw_8bit", group_by_length=True, report_to="none", seed=args.seed,
        prediction_loss_only=True)
    trainer = LabelledLossTrainer(model=model, args=targs, train_dataset=data["train"], eval_dataset=data["val"],
                      data_collator=DataCollatorForSeq2Seq(tok, padding=True, label_pad_token_id=-100))
    if not args.resume:
        # Fail in minutes, not hours, if evaluation or saving is broken; also gives the step-0 val loss.
        print("eval before training:", trainer.evaluate())
        trainer.save_model(str(args.out / "smoke"))
    trainer.train(resume_from_checkpoint=True if args.resume else None)
    final = args.out / "final"
    trainer.model.save_pretrained(final)
    tok.save_pretrained(final)
    json.dump({**{k: str(v) for k, v in vars(args).items()},
               "train_examples": len(data["train"]), "val_examples": len(data["val"]),
               "log": trainer.state.log_history}, open(args.out / "train_info.json", "w"), indent=1)
    print(f"adapter saved to {final}")


if __name__ == "__main__":
    main()
