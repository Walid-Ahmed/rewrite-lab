"""Train a Qwen3-1.7B rewrite adapter from humanized-paragraphs.json."""
import argparse
import json
import random
from pathlib import Path

SYSTEM = (
    "Rewrite the supplied paragraph in natural, clear English. Preserve its meaning, "
    "facts, technical identifiers, paths, commands, numbers, and references. "
    "Return only the rewritten paragraph, without commentary."
)


def load_pairs(path, reviewed_only=False, master=None):
    rows = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError("Expected a JSON list.")
    from training_eligibility import eligible_pairs, digest
    # Legacy rows carry no mode/view, so any eligible slot with the exact pair qualifies.
    allowed = {(source, output) for source, output, _, _ in eligible_pairs(master)}
    selected, seen = [], set()
    for row in rows:
        if row.get("training_eligible") is False:
            continue
        before, after = row.get("before"), row.get("after")
        if not isinstance(before, str) or not isinstance(after, str):
            continue
        if (digest(before), digest(after)) not in allowed:
            continue
        before, after = before.strip(), after.strip()
        if not before or not after or before == after:
            continue
        if row.get("status", "").startswith("preserved"):
            continue
        if "Subscribe to Unlock" in row.get("note", ""):
            continue
        if reviewed_only and row.get("reviewed") is not True:
            continue
        # Keep each source paragraph in only one split, including repeated inputs.
        key = " ".join(before.split()).casefold()
        if key in seen:
            continue
        seen.add(key)
        selected.append({**row, "before": before, "after": after})
    if len(selected) < 10:
        raise ValueError("Need at least 10 usable pairs. Check filters/reviewed flags.")
    return selected


def split_pairs(rows, seed):
    # Batch grouping is useful here because neighboring paragraphs were rewritten together.
    groups = {}
    for row in rows:
        group = (row.get("document", "unknown"), row.get("batch", "unknown"))
        groups.setdefault(group, []).append(row)
    rng = random.Random(seed)
    if len(groups) >= 2:
        keys = sorted(groups, key=str)
        rng.shuffle(keys)
        valid_keys = set(keys[:max(1, round(len(keys) * 0.2))])
        train = [r for k in keys if k not in valid_keys for r in groups[k]]
        valid = [r for k in keys if k in valid_keys for r in groups[k]]
    else:
        shuffled = list(rows)
        rng.shuffle(shuffled)
        n = max(1, round(len(shuffled) * 0.15))
        valid, train = shuffled[:n], shuffled[n:]
    return train, valid


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="humanized-paragraphs.json")
    parser.add_argument("--output", default="qwen3-humanizer")
    parser.add_argument("--epochs", type=float, default=2)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--reviewed-only", action="store_true")
    parser.add_argument("--master", help="Canonical master whose quality eligibility gates pairs (default: this project's humanizer-master.json)")
    parser.add_argument("--dry-run", action="store_true", help="Check data without ML dependencies or GPU")
    parser.add_argument("--sample", default="")
    args = parser.parse_args()
    rows = load_pairs(args.data, args.reviewed_only, args.master)
    train, valid = split_pairs(rows, args.seed)
    print(f"Usable pairs: {len(rows)}; train: {len(train)}; validation: {len(valid)}")
    print("Validation uses the same chapter; it does not measure generalization to other books.")
    if args.dry_run:
        return

    # Import Unsloth before transformers/TRL so its patches are applied first.
    from unsloth import FastLanguageModel, is_bfloat16_supported
    import torch
    from datasets import Dataset
    from trl import SFTConfig, SFTTrainer

    if not torch.cuda.is_available():
        raise RuntimeError("This 4-bit sample requires an NVIDIA CUDA GPU. Use a GPU Colab/Linux runtime.")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/Qwen3-1.7B-unsloth-bnb-4bit",
        max_seq_length=args.max_length,
        dtype=None,
        load_in_4bit=True,
    )
    model = FastLanguageModel.get_peft_model(
        model, r=16, lora_alpha=16, lora_dropout=0,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none", use_gradient_checkpointing="unsloth",
        random_state=args.seed, use_rslora=False,
    )

    def make_prompt(text):
        return tokenizer.apply_chat_template(
            [{"role": "system", "content": SYSTEM}, {"role": "user", "content": text}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )

    def prepare(items):
        records = []
        for row in items:
            prompt = make_prompt(row["before"])
            completion = row["after"] + tokenizer.eos_token
            length = len(tokenizer(prompt + completion, add_special_tokens=False)["input_ids"])
            if length > args.max_length:
                raise ValueError(f"Paragraph {row.get('paragraph_index')} needs {length} tokens; increase --max-length.")
            records.append({"prompt": prompt, "completion": completion})
        return Dataset.from_list(records)

    train_ds, valid_ds = prepare(train), prepare(valid)
    trainer = SFTTrainer(
        model=model, processing_class=tokenizer,
        train_dataset=train_ds, eval_dataset=valid_ds,
        args=SFTConfig(
            output_dir=str(Path(args.output) / "checkpoints"),
            max_length=args.max_length, packing=False,
            completion_only_loss=True,
            per_device_train_batch_size=1, per_device_eval_batch_size=1,
            gradient_accumulation_steps=8, num_train_epochs=args.epochs,
            learning_rate=1e-4, warmup_ratio=0.1, weight_decay=0.01,
            lr_scheduler_type="linear", optim="adamw_8bit",
            bf16=is_bfloat16_supported(), fp16=not is_bfloat16_supported(),
            logging_steps=5, eval_strategy="epoch", save_strategy="epoch",
            load_best_model_at_end=True, metric_for_best_model="eval_loss",
            greater_is_better=False, save_total_limit=2,
            seed=args.seed, report_to="none",
        ),
    )
    trainer.train()
    metrics = trainer.evaluate()
    adapter_dir = Path(args.output) / "adapter"
    model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    Path(args.output).mkdir(parents=True, exist_ok=True)
    manifest = {
        "base_model": "unsloth/Qwen3-1.7B-unsloth-bnb-4bit",
        "system_prompt": SYSTEM, "seed": args.seed,
        "train_paragraphs": [r.get("paragraph_index") for r in train],
        "validation_paragraphs": [r.get("paragraph_index") for r in valid],
        "metrics": metrics,
    }
    (Path(args.output) / "training-manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Saved LoRA adapter and tokenizer to {adapter_dir}")
    if args.sample:
        FastLanguageModel.for_inference(model)
        inputs = tokenizer(make_prompt(args.sample), return_tensors="pt", add_special_tokens=False).to(model.device)
        with torch.inference_mode():
            result = model.generate(**inputs, max_new_tokens=512, do_sample=False,
                                    pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
        print(tokenizer.decode(result[0, inputs.input_ids.shape[1]:], skip_special_tokens=True))


if __name__ == "__main__":
    main()
