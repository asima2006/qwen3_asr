#!/usr/bin/env python3
"""
qwen3_asr_sft.py — Fine-tune Qwen3-ASR from JSONL audio-text pairs.
Supports single-GPU (python) and multi-GPU (torchrun --nproc_per_node=N).
"""

import argparse, json, math, os, time, shutil
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from transformers import (
    AutoProcessor,
    AutoModelForSpeechSeq2Seq,
    get_cosine_schedule_with_warmup,
)

TARGET_SR = 16_000

# ── Argument parsing ───────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path",         default="Qwen/Qwen3-ASR-1.7B")
    p.add_argument("--train_file",         required=True)
    p.add_argument("--eval_file",          default=None)
    p.add_argument("--output_dir",         default="./qwen3-asr-finetuning-out")
    p.add_argument("--batch_size",         type=int,   default=4)
    p.add_argument("--grad_acc",           type=int,   default=8)
    p.add_argument("--lr",                 type=float, default=2e-5)
    p.add_argument("--epochs",             type=int,   default=1)
    p.add_argument("--warmup_ratio",       type=float, default=0.05)
    p.add_argument("--max_grad_norm",      type=float, default=1.0)
    p.add_argument("--save_strategy",      default="steps", choices=["steps","epoch"])
    p.add_argument("--save_steps",         type=int,   default=200)
    p.add_argument("--save_total_limit",   type=int,   default=5)
    p.add_argument("--resume_from",        default=None)
    p.add_argument("--resume",             type=int,   default=0)
    p.add_argument("--log_steps",          type=int,   default=10)
    p.add_argument("--num_workers",        type=int,   default=2)
    p.add_argument("--pin_memory",         type=int,   default=1)
    p.add_argument("--persistent_workers", type=int,   default=1)
    p.add_argument("--prefetch_factor",    type=int,   default=2)
    p.add_argument("--dtype",              default="bfloat16",
                                           choices=["float16","bfloat16","float32"])
    p.add_argument("--seed",               type=int,   default=42)
    return p.parse_args()


# ── Dataset ────────────────────────────────────────────────────
class ASRDataset(Dataset):
    def __init__(self, jsonl_path):
        self.records = []
        with open(jsonl_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.records.append(json.loads(line))

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec  = self.records[idx]
        arr, sr = sf.read(rec["audio"], dtype="float32", always_2d=False)
        if arr.ndim == 2:
            arr = arr.mean(axis=1)
        if sr != TARGET_SR:
            import librosa
            arr = librosa.resample(arr, orig_sr=sr, target_sr=TARGET_SR)
        return {"array": arr, "text": rec["text"]}


def collate_fn(batch, processor):
    arrays = [b["array"] for b in batch]
    texts  = [b["text"]  for b in batch]

    inputs = processor(
        audios        = arrays,
        sampling_rate = TARGET_SR,
        return_tensors= "pt",
        padding       = True,
    )
    with processor.as_target_processor():
        labels = processor(texts, return_tensors="pt", padding=True).input_ids
    labels[labels == processor.tokenizer.pad_token_id] = -100
    inputs["labels"] = labels
    return inputs


# ── Checkpoint helpers ─────────────────────────────────────────
def latest_checkpoint(output_dir):
    ckpts = sorted(
        (d for d in Path(output_dir).glob("checkpoint-*") if d.is_dir()),
        key=lambda p: int(p.name.split("-")[-1]),
    )
    return str(ckpts[-1]) if ckpts else None


def save_checkpoint(model, processor, optimizer, scheduler,
                    global_step, output_dir, save_total_limit):
    ckpt_dir = os.path.join(output_dir, f"checkpoint-{global_step}")
    os.makedirs(ckpt_dir, exist_ok=True)
    raw = model.module if hasattr(model, "module") else model
    raw.save_pretrained(ckpt_dir)
    processor.save_pretrained(ckpt_dir)
    torch.save(
        {"optimizer": optimizer.state_dict(),
         "scheduler": scheduler.state_dict(),
         "global_step": global_step},
        os.path.join(ckpt_dir, "trainer_state.pt"),
    )
    print(f"[ckpt] saved → {ckpt_dir}")

    if save_total_limit and save_total_limit > 0:
        all_ckpts = sorted(
            (d for d in Path(output_dir).glob("checkpoint-*") if d.is_dir()),
            key=lambda p: int(p.name.split("-")[-1]),
        )
        while len(all_ckpts) > save_total_limit:
            old = all_ckpts.pop(0)
            shutil.rmtree(str(old))
            print(f"[ckpt] pruned → {old}")


# ── Main ───────────────────────────────────────────────────────
def main():
    args       = parse_args()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    is_main    = (local_rank == 0)

    if world_size > 1:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)

    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed + local_rank)

    dtype_map   = {"float16": torch.float16,
                   "bfloat16": torch.bfloat16,
                   "float32": torch.float32}
    torch_dtype = dtype_map[args.dtype]

    if is_main:
        os.makedirs(args.output_dir, exist_ok=True)
        print(f"\n{'='*60}")
        print(f"  model  : {args.model_path}")
        print(f"  train  : {args.train_file}")
        print(f"  eval   : {args.eval_file}")
        print(f"  output : {args.output_dir}")
        print(f"  GPUs   : {world_size}  |  dtype: {args.dtype}")
        print(f"{'='*60}\n")

    # ── Resolve resume path ────────────────────────────────────
    resume_path = args.resume_from
    if not resume_path and args.resume:
        resume_path = latest_checkpoint(args.output_dir)
        if resume_path and is_main:
            print(f"[resume] auto-resuming from {resume_path}")

    load_path = resume_path if resume_path else args.model_path

    # ── Load processor & model ─────────────────────────────────
    processor = AutoProcessor.from_pretrained(load_path)
    model     = AutoModelForSpeechSeq2Seq.from_pretrained(
        load_path,
        torch_dtype         = torch_dtype,
        attn_implementation = "flash_attention_2",
        use_cache           = False,
    )
    model.gradient_checkpointing_enable()
    model = model.to(device)

    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)

    # ── DataLoaders ────────────────────────────────────────────
    train_ds      = ASRDataset(args.train_file)
    train_sampler = DistributedSampler(train_ds, shuffle=True) if world_size > 1 else None
    train_loader  = DataLoader(
        train_ds,
        batch_size         = args.batch_size,
        sampler            = train_sampler,
        shuffle            = (train_sampler is None),
        collate_fn         = lambda b: collate_fn(b, processor),
        num_workers        = args.num_workers,
        pin_memory         = bool(args.pin_memory),
        persistent_workers = bool(args.persistent_workers) and args.num_workers > 0,
        prefetch_factor    = args.prefetch_factor if args.num_workers > 0 else None,
        drop_last          = True,
    )

    eval_loader = None
    if args.eval_file:
        eval_ds     = ASRDataset(args.eval_file)
        eval_loader = DataLoader(
            eval_ds,
            batch_size = args.batch_size,
            shuffle    = False,
            collate_fn = lambda b: collate_fn(b, processor),
            num_workers= args.num_workers,
        )

    # ── Optimiser & scheduler ──────────────────────────────────
    optimizer         = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                          weight_decay=0.01, eps=1e-8)
    steps_per_epoch   = math.ceil(len(train_ds) /
                                  (args.batch_size * world_size * args.grad_acc))
    total_steps       = steps_per_epoch * args.epochs
    warmup_steps      = max(1, int(total_steps * args.warmup_ratio))
    scheduler         = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    # ── Restore optimiser state if resuming ───────────────────
    global_step = 0
    if resume_path:
        state_file = os.path.join(resume_path, "trainer_state.pt")
        if os.path.isfile(state_file):
            state = torch.load(state_file, map_location="cpu")
            optimizer.load_state_dict(state["optimizer"])
            scheduler.load_state_dict(state["scheduler"])
            global_step = state["global_step"]
            if is_main:
                print(f"[resume] restored at step {global_step}")

    if is_main:
        print(f"Total steps : {total_steps}  |  Warmup : {warmup_steps}")
        print(f"Starting at : step {global_step}\n")

    # ── Training loop ──────────────────────────────────────────
    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == "float16"))

    for epoch in range(args.epochs):
        model.train()
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        optimizer.zero_grad()
        running_loss = 0.0
        t0 = time.time()

        for step, batch in enumerate(train_loader):
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}

            with torch.cuda.amp.autocast(enabled=(args.dtype != "float32"),
                                         dtype=torch_dtype):
                loss = model(**batch).loss / args.grad_acc

            scaler.scale(loss).backward()
            running_loss += loss.item() * args.grad_acc

            if (step + 1) % args.grad_acc == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if is_main and global_step % args.log_steps == 0:
                    avg  = running_loss / args.log_steps / args.grad_acc
                    lr_  = scheduler.get_last_lr()[0]
                    print(f"Epoch {epoch+1} | step {global_step}/{total_steps} "
                          f"| loss {avg:.4f} | lr {lr_:.2e} | {time.time()-t0:.1f}s")
                    running_loss = 0.0
                    t0 = time.time()

                if (is_main and args.save_strategy == "steps"
                        and global_step % args.save_steps == 0):
                    save_checkpoint(model, processor, optimizer, scheduler,
                                    global_step, args.output_dir, args.save_total_limit)

        if is_main and args.save_strategy == "epoch":
            save_checkpoint(model, processor, optimizer, scheduler,
                            global_step, args.output_dir, args.save_total_limit)

        # ── Eval ───────────────────────────────────────────────
        if eval_loader is not None and is_main:
            model.eval()
            eval_loss = 0.0
            with torch.no_grad():
                for batch in eval_loader:
                    batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                             for k, v in batch.items()}
                    with torch.cuda.amp.autocast(enabled=(args.dtype != "float32"),
                                                 dtype=torch_dtype):
                        eval_loss += model(**batch).loss.item()
            print(f"\n[eval] epoch {epoch+1} | eval_loss = {eval_loss/len(eval_loader):.4f}\n")
            model.train()

    if is_main:
        save_checkpoint(model, processor, optimizer, scheduler,
                        global_step, args.output_dir, args.save_total_limit)
        print("\n🎉 Training complete!")

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()