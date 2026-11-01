"""Fine-tuning Whisper dengan prompt konteks (docs/TRAINING_DUAL_GPU.md, docs/CONTEXT_ENGINEERING.md).

Satu GPU (smoke test):
    CUDA_VISIBLE_DEVICES=0 python src/train.py --max_train_samples 200 --max_steps 50 --output_dir outputs/smoke
Dua GPU (DDP):
    torchrun --nproc_per_node=2 src/train.py --output_dir outputs/run-001

Evaluasi di dalam Trainer hanya eval_loss (prompt per sampel berbeda, generate() HF hanya menerima
satu prompt per batch). WER dan metrik entitas dihitung per checkpoint dengan src/evaluate.py.
"""
from __future__ import annotations

import argparse
import logging
import random
import zlib
from dataclasses import dataclass
from pathlib import Path

import soundfile as sf
import torch
from datasets import concatenate_datasets, load_dataset
from transformers import (Seq2SeqTrainer, Seq2SeqTrainingArguments, WhisperForConditionalGeneration,
                          WhisperProcessor, set_seed)

from context_builder import (Tok, build_decoder_sequence, build_prompt_ids, load_domain_prompt,
                             sample_prompt)

log = logging.getLogger("train")
SR = 16000


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="openai/whisper-large-v3")
    ap.add_argument("--train", default="dataset/train.jsonl")
    ap.add_argument("--eval", default="dataset/validation.jsonl")
    ap.add_argument("--domain_prompt", default="configs/domain_prompt.txt")
    ap.add_argument("--output_dir", default="outputs/run-001")
    ap.add_argument("--epochs", type=float, default=3)
    ap.add_argument("--max_steps", type=int, default=-1)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--warmup_steps", type=int, default=500)
    ap.add_argument("--batch_size", type=int, default=8, help="per GPU")
    ap.add_argument("--grad_accum", type=int, default=2)
    ap.add_argument("--workers", type=int, default=6, help="dataloader worker per GPU")
    ap.add_argument("--eval_steps", type=int, default=500)
    ap.add_argument("--save_total_limit", type=int, default=2)
    ap.add_argument("--pii_oversample", type=int, default=2, help="segmen PII muncul N kali di train")
    ap.add_argument("--freeze_encoder", action="store_true")
    ap.add_argument("--spec_augment", action="store_true")
    ap.add_argument("--optim", default="adamw_torch_fused", help="adamw_bnb_8bit bila OOM")
    ap.add_argument("--max_train_samples", type=int, default=None)
    ap.add_argument("--max_eval_samples", type=int, default=None)
    ap.add_argument("--resume_from_checkpoint", default=None)
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args(argv)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def load_split(path: str, max_samples: int | None, pii_oversample: int = 1, is_eval: bool = False):
    ds = load_dataset("json", data_files=path, split="train")
    if max_samples:
        ds = ds.select(range(min(max_samples, len(ds))))
    if pii_oversample > 1:
        pii = ds.filter(lambda r: bool(r["pii_types"]))
        ds = concatenate_datasets([ds] + [pii] * (pii_oversample - 1))
    return ds.map(lambda _: {"_eval": is_eval})


@dataclass
class ContextCollator:
    """Baca audio → log-mel; pilih variasi prompt; susun decoder_input_ids + labels."""
    processor: WhisperProcessor
    domain: str

    def __post_init__(self):
        self.tok = Tok(self.processor.tokenizer)
        self.pad_id = self.tok.special("eot")

    def __call__(self, features: list[dict]) -> dict:
        feats, dec, labs = [], [], []
        for ex in features:
            audio, sr = sf.read(ex["audio_filepath"], dtype="float32")
            if sr != SR:
                raise ValueError(f"{ex['audio_filepath']}: sample rate {sr}, harus {SR}")
            feats.append(self.processor.feature_extractor(
                audio, sampling_rate=SR, return_tensors="pt").input_features[0])
            # Train: acak per epoch (torch men-seed `random` di tiap worker).
            # Eval: deterministik per id agar eval_loss antar-checkpoint sebanding.
            rng = random.Random(zlib.crc32(ex["id"].encode())) if ex.get("_eval") else random
            choice = sample_prompt(ex, self.domain, rng)
            prompt = build_prompt_ids(self.tok, choice.domain, choice.crm, choice.history)
            d, l = build_decoder_sequence(self.tok, prompt, ex["text"])
            dec.append(d)
            labs.append(l)
        n = max(len(d) for d in dec)
        dec_t = torch.full((len(dec), n), self.pad_id, dtype=torch.long)
        lab_t = torch.full((len(dec), n), -100, dtype=torch.long)
        mask = torch.zeros((len(dec), n), dtype=torch.long)
        for i, (d, l) in enumerate(zip(dec, labs)):
            dec_t[i, :len(d)] = torch.tensor(d)
            lab_t[i, :len(l)] = torch.tensor(l)
            mask[i, :len(d)] = 1
        return {
            "input_features": torch.stack(feats),
            "decoder_input_ids": dec_t,
            "decoder_attention_mask": mask,
            "labels": lab_t,
        }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv=None):
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    set_seed(args.seed)

    processor = WhisperProcessor.from_pretrained(args.model, language="indonesian", task="transcribe")
    model = WhisperForConditionalGeneration.from_pretrained(args.model)
    model.config.forced_decoder_ids = None
    model.config.use_cache = False
    model.generation_config.language = "id"
    model.generation_config.task = "transcribe"
    model.generation_config.forced_decoder_ids = None
    if args.spec_augment:
        model.config.apply_spec_augment = True
        model.config.mask_time_prob = 0.05
    if args.freeze_encoder:
        model.model.encoder.requires_grad_(False)

    train_ds = load_split(args.train, args.max_train_samples, args.pii_oversample)
    eval_ds = load_split(args.eval, args.max_eval_samples, is_eval=True) if Path(args.eval).exists() else None
    if eval_ds is not None and len(eval_ds) == 0:
        eval_ds = None
    log.info("train %d sampel (setelah oversampling PII x%d), eval %s",
             len(train_ds), args.pii_oversample, len(eval_ds) if eval_ds else "-")

    targs = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="linear",
        warmup_steps=args.warmup_steps,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        bf16=True,
        tf32=True,
        optim=args.optim,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        ddp_find_unused_parameters=False,
        dataloader_num_workers=args.workers,
        dataloader_pin_memory=True,
        dataloader_persistent_workers=args.workers > 0,
        remove_unused_columns=False,
        label_names=["labels"],
        eval_strategy="steps" if eval_ds is not None else "no",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.eval_steps,
        save_total_limit=args.save_total_limit,
        logging_steps=25,
        report_to=["tensorboard"],   # lokal saja: data berisi PII
        seed=args.seed,
    )
    trainer = Seq2SeqTrainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=ContextCollator(processor, load_domain_prompt(args.domain_prompt)),
        processing_class=processor,   # tokenizer + feature extractor ikut tersimpan di setiap checkpoint
    )
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    peak = torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else 0.0
    log.info("rank %s puncak memori GPU: %.1f GiB", targs.process_index, peak)
    if trainer.is_world_process_zero():
        final = Path(args.output_dir) / "final"
        trainer.model.config.use_cache = True   # dimatikan hanya untuk gradient checkpointing
        trainer.save_model(str(final))
        processor.save_pretrained(str(final))
        log.info("model + processor disimpan di %s", final)


if __name__ == "__main__":
    main()
