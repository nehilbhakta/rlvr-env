"""Train the tiny transformer on RLVR task variants (supervised fine-tuning).

Usage:
    python train.py [--epochs N] [--math-per-task N] [--code-copies N]
                   [--batch N] [--lr F] [--d-model N] [--n-layer N]
                   [--dev-every N] [--smoke]

Loss is only on the target (solution) tokens. Checkpoints + training curve
are written to --out (default: checkpoints/ next to this script).
"""

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "rlvr-env"))

from data import ORIGINAL, build_heldout_set, build_train_set  # noqa: E402
from model import BOS, EOS, PAD, SEP, TinyGPT, count_params  # noqa: E402
from rlvr_env.rewards import verify  # noqa: E402

HERE = Path(__file__).resolve().parent


def build_vocab(examples):
    chars = set()
    for e in examples:
        chars.update(e["prompt"])
        chars.update(e["target"])
    itos = [PAD, BOS, SEP, EOS] + sorted(chars)
    stoi = {c: i for i, c in enumerate(itos)}
    return itos, stoi


def encode(example, stoi, max_len):
    ids = [stoi[BOS]]
    ids += [stoi[c] for c in example["prompt"]]
    prompt_len = len(ids)
    ids += [stoi[SEP]]
    ids += [stoi[c] for c in example["target"]]
    ids += [stoi[EOS]]
    ids = ids[:max_len]
    # loss mask: 1 for target chars + EOS, 0 elsewhere
    mask = [0] * len(ids)
    for i in range(prompt_len + 1, len(ids)):
        mask[i] = 1
    return ids, mask


def make_batches(encoded, batch_size, rng):
    # Length-grouped batching: sort by length, chunk, shuffle chunk order.
    order = sorted(range(len(encoded)), key=lambda i: len(encoded[i][0]))
    chunks = [order[i : i + batch_size] for i in range(0, len(order), batch_size)]
    rng.shuffle(chunks)
    return chunks


def collate(idxs, encoded, pad_id):
    seqs = [encoded[i] for i in idxs]
    maxlen = max(len(s[0]) for s in seqs)
    B = len(seqs)
    x = torch.full((B, maxlen), pad_id, dtype=torch.long)
    mask = torch.zeros((B, maxlen), dtype=torch.long)
    for r, (ids, m) in enumerate(seqs):
        x[r, : len(ids)] = torch.tensor(ids)
        mask[r, : len(m)] = torch.tensor(m)
    return x, mask


@torch.no_grad()
def greedy_answer(model, prompt, stoi, itos, max_new, device):
    ids = [stoi[BOS]] + [stoi[c] for c in prompt] + [stoi[SEP]]
    idx = torch.tensor([ids], dtype=torch.long, device=device)
    out = model.generate(idx, max_new, eos_id=stoi[EOS])
    gen = out[0, len(ids) :].tolist()
    text = "".join(itos[i] for i in gen if itos[i] not in (EOS, PAD))
    return text


def dev_eval(model, dev_items, stoi, itos, device):
    """Greedy pass rate on a small held-out slice. Returns (passed, total, details)."""
    model.eval()
    passed = 0
    details = []
    for item in dev_items:
        max_new = 16 if item["category"] == "math" else 420
        ans = greedy_answer(model, item["prompt"], stoi, itos, max_new, device)
        task = dict(ORIGINAL[item["task_id"]])
        if item["category"] == "math":
            task["answer"] = item["answer"]
        ok, detail = verify(task, ans)
        passed += int(ok)
        details.append((item["task_id"], ok, ans[:80], detail[:100]))
    return passed, len(dev_items), details


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--math-per-task", type=int, default=400)
    ap.add_argument("--code-copies", type=int, default=150)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--n-layer", type=int, default=3)
    ap.add_argument("--n-head", type=int, default=4)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--dev-every", type=int, default=5)
    ap.add_argument("--save-every", type=int, default=0,
                    help="save an intermediate checkpoint every N epochs (0 = only at end)")
    ap.add_argument("--out", type=str, default=str(HERE / "checkpoints"))
    ap.add_argument("--resume", type=str, default=None,
                    help="checkpoint to continue training from")
    ap.add_argument("--smoke", action="store_true", help="tiny run to verify the pipeline")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    device = torch.device("cpu")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "train.log"

    def log(msg):
        print(msg, flush=True)
        with open(log_path, "a") as f:
            f.write(msg + "\n")

    if args.smoke:
        args.epochs, args.math_per_task, args.code_copies = 1, 20, 5

    t0 = time.time()
    log(f"building data: math_per_task={args.math_per_task} code_copies={args.code_copies}")
    train_examples = build_train_set(
        math_per_task=args.math_per_task, code_copies=args.code_copies, seed=args.seed
    )
    heldout = build_heldout_set(math_per_task=20, seed=987654)
    dev_items = [e for e in heldout if e["category"] == "math"][:5] + [
        e for e in heldout if e["category"] == "code"
    ][:5]
    log(f"train examples: {len(train_examples)} | held-out pool: {len(heldout)}")

    itos, stoi = build_vocab(train_examples)
    log(f"vocab size: {len(itos)}")
    encoded = [encode(e, stoi, max_len=640) for e in train_examples]
    max_len = max(len(ids) for ids, _ in encoded)
    log(f"max sequence length: {max_len}")

    resume_ckpt = None
    if args.resume:
        resume_ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        itos, stoi = resume_ckpt["itos"], resume_ckpt["stoi"]
        args.d_model = resume_ckpt["config"]["d_model"]
        args.n_layer = resume_ckpt["config"]["n_layer"]
        args.n_head = resume_ckpt["config"]["n_head"]
        # Re-encode with the checkpoint's vocab (identical data, but be safe).
        encoded = [encode(e, stoi, max_len=640) for e in train_examples]
        log(f"resuming from {args.resume}")

    model = TinyGPT(
        len(itos), d_model=args.d_model, n_layer=args.n_layer, n_head=args.n_head, max_len=640
    ).to(device)
    if resume_ckpt is not None:
        model.load_state_dict(resume_ckpt["state_dict"])
    log(f"model params: {count_params(model):,}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01, betas=(0.9, 0.95))
    rng = random.Random(args.seed)
    steps_per_epoch = math.ceil(len(encoded) / args.batch)
    total_steps = steps_per_epoch * args.epochs
    warmup = min(100, total_steps // 10)

    def lr_at(step):
        if step < warmup:
            return args.lr * step / max(1, warmup)
        p = (step - warmup) / max(1, total_steps - warmup)
        return args.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * p)))

    curve = []
    epoch_offset = 0
    curve_path = out_dir / "training_curve.json"
    if resume_ckpt is not None and curve_path.exists():
        try:
            curve = json.load(open(curve_path))["curve"]
            epoch_offset = len(curve)
            log(f"continuing curve from epoch {epoch_offset}")
        except Exception:
            curve, epoch_offset = [], 0
    step = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        ep_loss, ep_tok, ep_start = 0.0, 0, time.time()
        for chunk in make_batches(encoded, args.batch, rng):
            for g in opt.param_groups:
                g["lr"] = lr_at(step)
            x, mask = collate(chunk, encoded, stoi[PAD])
            _, loss = model(x, targets=x, loss_mask=mask)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            ep_loss += loss.item() * int(mask.sum())
            ep_tok += int(mask.sum())
            step += 1
        avg_loss = ep_loss / max(1, ep_tok)
        msg = (
            f"epoch {epoch + epoch_offset}/{args.epochs + epoch_offset} "
            f"loss {avg_loss:.4f} ({time.time() - ep_start:.0f}s)"
        )
        dev_pass = None
        if args.dev_every and epoch % args.dev_every == 0:
            p, n, details = dev_eval(model, dev_items, stoi, itos, device)
            dev_pass = p / n
            msg += f" | dev pass {p}/{n}"
            for tid, ok, ans, det in details:
                log(f"    dev {tid} {'PASS' if ok else 'FAIL'} ans={ans!r} :: {det}")
        curve.append(
            {"epoch": epoch + epoch_offset, "loss": avg_loss, "dev_pass_rate": dev_pass}
        )
        log(msg)
        with open(out_dir / "training_curve.json", "w") as f:
            json.dump({"curve": curve, "config": vars(args)}, f, indent=2)
        if args.save_every and epoch % args.save_every == 0:
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "config": {
                        "vocab_size": len(itos),
                        "d_model": args.d_model,
                        "n_layer": args.n_layer,
                        "n_head": args.n_head,
                        "max_len": 640,
                    },
                    "itos": itos,
                    "stoi": stoi,
                },
                out_dir / f"model-epoch{epoch + epoch_offset}.pt",
            )
            log(f"saved intermediate checkpoint epoch {epoch + epoch_offset}")

    ckpt = {
        "state_dict": model.state_dict(),
        "config": {
            "vocab_size": len(itos),
            "d_model": args.d_model,
            "n_layer": args.n_layer,
            "n_head": args.n_head,
            "max_len": 640,
        },
        "itos": itos,
        "stoi": stoi,
    }
    torch.save(ckpt, out_dir / "model.pt")
    log(f"saved checkpoint to {out_dir / 'model.pt'}")
    log(f"total training time: {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
