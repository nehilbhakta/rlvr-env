"""GRPO reinforcement-learning pass on the tiny RLVR transformer.

Starts from the SFT checkpoint (checkpoints/model.pt) and improves it with
group-relative policy optimization against the env's own verifiable
rewards (rlvr_env.rewards.verify): reward is 1.0 iff the sampled answer
passes the verifier, else 0.0. No partial credit, no learned reward model.

Per step:
  * sample a batch of prompts — fresh math variants from the TRAIN split
    (disjoint from the held-out eval ranges) + the 10 fixed code prompts.
  * sample G completions per prompt at temperature 1.0 (batched per prompt).
  * group-relative advantage: A = (r - group_mean) / (group_std + 1e-8);
    groups where every sample gets the same reward give A = 0 (no signal).
  * one clipped-surrogate policy-gradient update (eps=0.2) with a KL
    penalty (k3 estimator) against the frozen SFT reference.

The SFT checkpoint is NEVER overwritten: output is checkpoints/model-rl.pt.

Usage:
    python train_grpo.py [--steps N] [--prompts-per-step N] [--group-size G]
                         [--lr F] [--beta-kl F] [--temp F] [--seed N]
                         [--smoke] [--out PATH]
"""

import argparse
import copy
import json
import math
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "rlvr-env"))

from data import MATH_GENERATORS, ORIGINAL, build_math_variants  # noqa: E402
from model import BOS, EOS, PAD, SEP, TinyGPT  # noqa: E402
from rlvr_env.rewards import verify  # noqa: E402
from rlvr_env.tasks import CODE_TASKS  # noqa: E402

HERE = Path(__file__).resolve().parent


@torch.no_grad()
def sample_group(model, prompt_ids, group_size, max_new, temperature, eos_id, device):
    """Sample G completions for one prompt (batched). Returns list of
    (token_ids, old_logprobs) — token_ids exclude the prompt, include EOS
    if generated; old_logprobs are the log-probs under the sampling policy.
    """
    model.eval()
    idx = torch.tensor([prompt_ids] * group_size, dtype=torch.long, device=device)
    finished = torch.zeros(group_size, dtype=torch.bool, device=device)
    gen_ids = [[] for _ in range(group_size)]
    gen_logps = [[] for _ in range(group_size)]
    for _ in range(max_new):
        logits, _ = model(idx)
        next_logits = logits[:, -1, :].float()
        logp = F.log_softmax(next_logits / temperature, dim=-1)
        nxt = torch.multinomial(torch.exp(logp), 1).squeeze(1)  # (G,)
        # Log-prob of the sampled token (under the tempered distribution).
        tok_logp = logp.gather(1, nxt.unsqueeze(1)).squeeze(1)
        for g in range(group_size):
            if not finished[g]:
                gen_ids[g].append(int(nxt[g]))
                gen_logps[g].append(float(tok_logp[g]))
                if int(nxt[g]) == eos_id:
                    finished[g] = True
        # Keep finished rows on PAD so the batch stays rectangular.
        nxt = torch.where(finished, torch.tensor(0, device=device), nxt)
        idx = torch.cat([idx, nxt.unsqueeze(1)], dim=1)
        if bool(finished.all()):
            break
    return list(zip(gen_ids, gen_logps))


def completion_text(gen_ids, itos):
    return "".join(itos[i] for i in gen_ids if itos[i] not in (EOS, PAD))


def token_logprobs(model, seqs, prompt_lens, device, pad_id):
    """Per-token log-probs of the completion part of each sequence.

    seqs: list of full id lists (prompt + completion). Returns a padded
    (B, T) tensor of log-probs aligned with seqs[:, 1:] and a (B, T) mask
    that is 1 exactly on completion tokens (excluding prompt and PAD).
    """
    maxlen = max(len(s) for s in seqs)
    B = len(seqs)
    x = torch.full((B, maxlen), pad_id, dtype=torch.long, device=device)
    mask = torch.zeros((B, maxlen), dtype=torch.long, device=device)
    for r, (s, plen) in enumerate(zip(seqs, prompt_lens)):
        x[r, : len(s)] = torch.tensor(s, device=device)
        for i in range(plen, len(s)):
            mask[r, i] = 1
    logits, _ = model(x)
    logp = F.log_softmax(logits[:, :-1, :].float(), dim=-1)
    tgt = x[:, 1:]
    tok_lp = logp.gather(2, tgt.unsqueeze(-1)).squeeze(-1)
    # tok_lp[:, i] is the log-prob of x[:, i+1]; completion mask shifts too.
    return tok_lp, mask[:, 1:].float()


@torch.no_grad()
def greedy_answer(model, prompt, stoi, itos, max_new, device):
    ids = [stoi[BOS]] + [stoi[c] for c in prompt] + [stoi[SEP]]
    idx = torch.tensor([ids], dtype=torch.long, device=device)
    out = model.generate(idx, max_new, eos_id=stoi[EOS])
    gen = out[0, len(ids):].tolist()
    return "".join(itos[i] for i in gen if itos[i] not in (EOS, PAD))


def dev_eval(model, dev_items, stoi, itos, device):
    model.eval()
    passed = 0
    for item in dev_items:
        max_new = 16 if item["category"] == "math" else 420
        ans = greedy_answer(model, item["prompt"], stoi, itos, max_new, device)
        task = dict(ORIGINAL[item["task_id"]])
        if item["category"] == "math":
            task["answer"] = item["answer"]
        ok, _ = verify(task, ans)
        passed += int(ok)
    return passed, len(dev_items)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", type=str, default=str(HERE / "checkpoints" / "model.pt"))
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--prompts-per-step", type=int, default=16)
    ap.add_argument("--math-per-step", type=int, default=12)
    ap.add_argument("--group-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--beta-kl", type=float, default=0.04)
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--eps", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=20261106)
    ap.add_argument("--dev-every", type=int, default=10)
    ap.add_argument("--save-every", type=int, default=25)
    ap.add_argument("--out", type=str, default=str(HERE / "checkpoints" / "model-rl.pt"))
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    if args.smoke:
        args.steps, args.prompts_per_step, args.math_per_step = 2, 4, 2
        args.group_size = 4
        args.dev_every, args.save_every = 0, 0

    torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    device = torch.device("cpu")
    rng = random.Random(args.seed)
    t0 = time.time()

    ckpt = torch.load(args.init, map_location=device, weights_only=False)
    itos, stoi = ckpt["itos"], ckpt["stoi"]
    cfg = ckpt["config"]

    policy = TinyGPT(**cfg).to(device)
    policy.load_state_dict(ckpt["state_dict"])
    ref = copy.deepcopy(policy).eval()
    for p in ref.parameters():
        p.requires_grad_(False)
    print(f"policy params: {sum(p.numel() for p in policy.parameters()):,} | "
          f"init: {args.init}", flush=True)

    # Prompt pool: fresh TRAIN-split math variants (never the held-out ranges)
    # + the 10 fixed code prompts (same prompts SFT saw).
    math_pool = []
    for task_id in MATH_GENERATORS:
        math_pool.extend(
            build_math_variants(task_id, 200, seed=args.seed + 1, split="train")
        )
    code_pool = [
        {"task_id": t["id"], "category": "code", "prompt": t["prompt"], "answer": None}
        for t in CODE_TASKS
    ]
    print(f"prompt pool: {len(math_pool)} math (train split) + {len(code_pool)} code",
          flush=True)

    dev_items = []
    for task_id in ["math-01", "math-03", "math-05", "math-08", "math-10"]:
        dev_items.extend(build_math_variants(task_id, 1, seed=987654, split="test"))
    dev_items.extend(code_pool[:5])

    opt = torch.optim.AdamW(policy.parameters(), lr=args.lr, weight_decay=0.01,
                            betas=(0.9, 0.95))
    total_steps = args.steps

    def lr_at(step):
        warmup = min(5, total_steps // 10)
        if step < warmup:
            return args.lr * (step + 1) / warmup
        p = (step - warmup) / max(1, total_steps - warmup)
        return args.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * p)))

    curve = []
    log_path = Path(args.out).parent / "grpo.log"

    def log(msg):
        print(msg, flush=True)
        with open(log_path, "a") as f:
            f.write(msg + "\n")

    def save_ckpt(path):
        torch.save(
            {"state_dict": policy.state_dict(), "config": cfg,
             "itos": itos, "stoi": stoi,
             "grpo": {"steps": args.steps, "group_size": args.group_size,
                      "lr": args.lr, "beta_kl": args.beta_kl, "temp": args.temp,
                      "seed": args.seed, "init": args.init}},
            path,
        )

    for step in range(1, args.steps + 1):
        step_t0 = time.time()
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        # ---- sample prompts: stratified math + code -------------------------
        math_prompts = rng.sample(math_pool, args.math_per_step)
        code_prompts = rng.sample(code_pool, args.prompts_per_step - args.math_per_step)
        batch = math_prompts + code_prompts
        rng.shuffle(batch)

        # ---- rollout --------------------------------------------------------
        groups = []  # per prompt: {task, rewards, completions}
        for item in batch:
            prompt_ids = [stoi[BOS]] + [stoi[c] for c in item["prompt"]] + [stoi[SEP]]
            max_new = 16 if item["category"] == "math" else 200
            samples = sample_group(policy, prompt_ids, args.group_size, max_new,
                                   args.temp, stoi[EOS], device)
            task = dict(ORIGINAL[item["task_id"]])
            if item["category"] == "math":
                task["answer"] = item["answer"]
            rewards, completions = [], []
            for gen_ids, old_lp in samples:
                text = completion_text(gen_ids, itos)
                ok, _ = verify(task, text)
                rewards.append(1.0 if ok else 0.0)
                completions.append((gen_ids, old_lp))
            groups.append({"item": item, "prompt_ids": prompt_ids,
                           "rewards": rewards, "completions": completions})

        # ---- advantages -----------------------------------------------------
        n_mixed = 0
        flat = []  # (seq_ids, prompt_len, old_lp_list, advantage)
        for grp in groups:
            r = torch.tensor(grp["rewards"])
            mean, std = float(r.mean()), float(r.std())
            if std > 1e-8:
                n_mixed += 1
                adv = (r - mean) / std
            else:
                adv = torch.zeros_like(r)
            for (gen_ids, old_lp), a in zip(grp["completions"], adv):
                if len(gen_ids) == 0:
                    continue
                flat.append((grp["prompt_ids"] + gen_ids, len(grp["prompt_ids"]),
                             old_lp, float(a)))

        # ---- update ---------------------------------------------------------
        policy.train()
        if flat:
            seqs = [f[0] for f in flat]
            plens = [f[1] for f in flat]
            old_lp = torch.zeros(len(flat), max(len(f[2]) for f in flat))
            adv_t = torch.tensor([f[3] for f in flat])
            for i, f in enumerate(flat):
                old_lp[i, : len(f[2])] = torch.tensor(f[2])
            new_lp, mask = token_logprobs(policy, seqs, plens, device, stoi[PAD])
            with torch.no_grad():
                ref_lp, _ = token_logprobs(ref, seqs, plens, device, stoi[PAD])
            old_lp = old_lp.to(device)
            adv_t = adv_t.to(device)
            # Align old_lp to the same (B, T) layout as new_lp: token at
            # position i of the completion is predicted at x-index plen+i-1.
            old_full = torch.zeros_like(new_lp)
            for i, f in enumerate(flat):
                n = len(f[2])
                old_full[i, f[1] - 1: f[1] - 1 + n] = old_lp[i, :n]
            ratio = torch.exp(new_lp - old_full)
            surr1 = ratio * adv_t.unsqueeze(1)
            surr2 = torch.clamp(ratio, 1 - args.eps, 1 + args.eps) * adv_t.unsqueeze(1)
            pg_loss = -(torch.min(surr1, surr2) * mask).sum() / mask.sum().clamp(min=1)
            # KL penalty, k3 estimator (always >= 0).
            kl_token = torch.exp(ref_lp - new_lp) - (ref_lp - new_lp) - 1
            kl = (kl_token * mask).sum() / mask.sum().clamp(min=1)
            loss = pg_loss + args.beta_kl * kl
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            opt.step()
            loss_v, kl_v, pg_v = float(loss.detach()), float(kl.detach()), float(pg_loss.detach())
        else:
            loss_v, kl_v, pg_v = float("nan"), 0.0, float("nan")

        all_r = [r for grp in groups for r in grp["rewards"]]
        math_r = [r for grp in groups if grp["item"]["category"] == "math"
                  for r in grp["rewards"]]
        code_r = [r for grp in groups if grp["item"]["category"] == "code"
                  for r in grp["rewards"]]
        rec = {
            "step": step,
            "reward_mean": sum(all_r) / len(all_r),
            "reward_math": sum(math_r) / max(1, len(math_r)),
            "reward_code": sum(code_r) / max(1, len(code_r)),
            "mixed_groups": n_mixed,
            "groups": len(groups),
            "loss": loss_v, "kl": kl_v, "pg_loss": pg_v,
            "lr": lr_at(step),
            "seconds": round(time.time() - step_t0, 1),
        }
        if args.dev_every and step % args.dev_every == 0:
            p, n = dev_eval(policy, dev_items, stoi, itos, device)
            rec["dev_pass"] = f"{p}/{n}"
        curve.append(rec)
        log(f"step {step}/{args.steps} reward {rec['reward_mean']:.3f} "
            f"(math {rec['reward_math']:.3f} code {rec['reward_code']:.3f}) "
            f"mixed {n_mixed}/{len(groups)} loss {loss_v:.4f} kl {kl_v:.4f} "
            f"({rec['seconds']}s)"
            + (f" | dev {rec['dev_pass']}" if "dev_pass" in rec else ""))
        with open(Path(args.out).parent / "grpo_curve.json", "w") as f:
            json.dump({"curve": curve, "config": vars(args)}, f, indent=2)
        if args.save_every and step % args.save_every == 0:
            save_ckpt(Path(args.out).parent / f"model-rl-step{step}.pt")
            log(f"  saved intermediate {Path(args.out).parent / f'model-rl-step{step}.pt'}")

    save_ckpt(args.out)
    log(f"saved RL checkpoint to {args.out}")
    log(f"total GRPO time: {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
