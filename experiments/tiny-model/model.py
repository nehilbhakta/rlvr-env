"""Tiny decoder-only transformer, trained from scratch (char-level).

Architecture: GPT-style decoder with SDPA attention, ~1M params, CPU-friendly.
Input format:  <bos> PROMPT <sep> TARGET <eos>
Loss is computed only on the target tokens (prompt is context).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

BOS, SEP, EOS, PAD = "<bos>", "<sep>", "<eos>", "<pad>"


class TinyGPT(nn.Module):
    def __init__(self, vocab_size, d_model=128, n_layer=3, n_head=4, max_len=640, dropout=0.0):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [Block(d_model, n_head, dropout) for _ in range(n_layer)]
        )
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.apply(self._init)

    def _init(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=0.02)

    def forward(self, idx, targets=None, loss_mask=None):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.tok_emb(idx) + self.pos_emb(pos)[None, :, :]
        x = self.drop(x)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.head(x)
        loss = None
        if targets is not None:
            logp = F.log_softmax(logits[:, :-1, :].float(), dim=-1)
            tgt = targets[:, 1:]
            nll = -logp.gather(2, tgt.unsqueeze(-1)).squeeze(-1)
            if loss_mask is not None:
                m = loss_mask[:, 1:].float()
                loss = (nll * m).sum() / m.sum().clamp(min=1)
            else:
                loss = nll.mean()
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, eos_id, temperature=0.0):
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.max_len :]
            logits, _ = self.forward(idx_cond)
            next_logits = logits[:, -1, :]
            if temperature > 0:
                probs = F.softmax(next_logits.float() / temperature, dim=-1)
                nxt = torch.multinomial(probs, 1)
            else:
                nxt = torch.argmax(next_logits, dim=-1, keepdim=True)
            idx = torch.cat([idx, nxt], dim=1)
            if int(nxt[0, 0]) == eos_id:
                break
        return idx


class Block(nn.Module):
    def __init__(self, d_model, n_head, dropout):
        super().__init__()
        assert d_model % n_head == 0
        self.n_head = n_head
        self.head_dim = d_model // n_head
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        B, T, C = x.shape
        h = self.ln1(x)
        qkv = self.qkv(h).view(B, T, 3, self.n_head, self.head_dim)
        q, k, v = qkv[:, :, 0].transpose(1, 2), qkv[:, :, 1].transpose(1, 2), qkv[:, :, 2].transpose(1, 2)
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        a = a.transpose(1, 2).contiguous().view(B, T, C)
        x = x + self.proj(a)
        x = x + self.mlp(self.ln2(x))
        return x


def count_params(model):
    return sum(p.numel() for p in model.parameters())
