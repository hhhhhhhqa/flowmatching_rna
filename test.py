

import os, math, json, random, argparse, time
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from accelerate import Accelerator
from accelerate.utils import set_seed
from transformers import AutoTokenizer, PreTrainedTokenizerFast

# ========================
# Dataset
# ========================

class ProtRnaDataset(Dataset):
    """
    jsonl 行格式: {"input": <protein_aa_seq>, "output": <rna_seq>}
    """
    def __init__(self, path, rna_tok: PreTrainedTokenizerFast, max_len=1024):
        self.samples = [json.loads(l) for l in open(path)]
        self.rna_tok, self.max_len = rna_tok, max_len

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        prot = self.samples[idx]["input"]
        rna  = self.samples[idx]["output"]
        enc  = self.rna_tok(
            rna, max_length=self.max_len, padding="max_length",
            truncation=True, return_tensors="pt"
        )
        return {"prot": prot, "rna_ids": enc["input_ids"].squeeze(0)}

def collate_fn(batch: List[Dict]):
    return {
        "rna_ids": torch.stack([b["rna_ids"] for b in batch]),
        "prot":    [b["prot"] for b in batch]
    }

# ========================
# Encoder-MLP
# ========================

def timestep_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """sin/cos 时间嵌入: (B,) -> (B, dim)"""
    half = dim // 2
    freqs = torch.exp(torch.linspace(math.log(1.0), math.log(10000.0), steps=half, device=t.device))
    args  = t[:, None] * freqs[None, :]
    emb   = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if dim % 2 == 1:
        emb = torch.nn.functional.pad(emb, (0,1))
    return emb

class ProteinEncoder(nn.Module):
    def __init__(self, tokenizer_path, embed_dim: int, hidden_size: int):
        super().__init__()
        self.tok   = PreTrainedTokenizerFast.from_pretrained(tokenizer_path)
        self.embed = nn.Embedding(self.tok.vocab_size, embed_dim)
        self.mlp   = nn.Sequential(
            nn.Linear(embed_dim, 512), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(512, hidden_size)
        )

    def forward(self, seq_list):
        ids = [torch.tensor(self.tok.encode(s), dtype=torch.long) for s in seq_list]
        Lm  = max(x.size(0) for x in ids)
        pad = torch.stack([torch.nn.functional.pad(x, (0, Lm - x.size(0))) for x in ids])  # (B,Lp)
        x   = self.embed(pad.to(self.embed.weight.device))  # (B,Lp,E)
        return self.mlp(x)                                   # (B,Lp,H)

# ========================
# Flow Decoder (Transformer + Cross-Attn)
# ========================

class CrossAttnBlock(nn.Module):
    def __init__(self, hidden: int, n_heads: int, ff_mult=4, p=0.1):
        super().__init__()
        self.self_attn  = nn.MultiheadAttention(hidden, n_heads, batch_first=True, dropout=p)
        self.cross_attn = nn.MultiheadAttention(hidden, n_heads, batch_first=True, dropout=p)
        self.ff   = nn.Sequential(
            nn.Linear(hidden, hidden*ff_mult),
            nn.GELU(),
            nn.Dropout(p),
            nn.Linear(hidden*ff_mult, hidden)
        )
        self.ln1, self.ln2, self.ln3 = nn.LayerNorm(hidden), nn.LayerNorm(hidden), nn.LayerNorm(hidden)
        self.drop = nn.Dropout(p)

    def forward(self, x, mem, src_kpm=None, mem_kpm=None):
        h = self.ln1(x)
        s,_ = self.self_attn(h, h, h, key_padding_mask=src_kpm, need_weights=False)
        x = x + self.drop(s)
        h = self.ln2(x)
        c,_ = self.cross_attn(h, mem, mem, key_padding_mask=mem_kpm, need_weights=False)
        x = x + self.drop(c)
        h = self.ln3(x)
        x = x + self.drop(self.ff(h))
        return x

class FlowDecoder(nn.Module):
    """
    输入: x_logits (B,L,V)  条件: cond_mem (B,L,H)  时间: t (B,)
    输出: 速度 v(x_t, cond, t) (B,L,V)
    """
    def __init__(self, vocab_size:int, hidden:int=640, n_heads:int=10, n_layers:int=8, t_dim:int=128, p:float=0.1):
        super().__init__()
        self.in_proj  = nn.Linear(vocab_size, hidden)
        self.t_proj   = nn.Linear(t_dim, hidden)
        self.blocks   = nn.ModuleList([CrossAttnBlock(hidden, n_heads, p=p) for _ in range(n_layers)])
        self.ln_in    = nn.LayerNorm(hidden)
        self.ln_out   = nn.LayerNorm(hidden)
        self.out_proj = nn.Linear(hidden, vocab_size)

    def forward(self, x_logits, cond_mem, t, src_key_padding_mask=None, mem_key_padding_mask=None):
        B,L,V = x_logits.shape
        H = self.in_proj.out_features
        h = self.in_proj(x_logits)
        t_emb = timestep_embedding(t, self.t_proj.in_features)
        h = self.ln_in(h + self.t_proj(t_emb).unsqueeze(1).expand(B, L, H))
        x = h
        for blk in self.blocks:
            x = blk(x, cond_mem, src_kpm=src_key_padding_mask, mem_kpm=mem_key_padding_mask)
        return self.out_proj(self.ln_out(x))

# ========================
# Flow Matching
# ========================

def label_smooth_onehot(ids: torch.Tensor, vocab: int, eps: float, pad_id: int):
    """
    返回概率张量 y ∈ R^{B×L×V}；PAD 行清零
    """
    B,L = ids.shape
    y = torch.full((B,L,vocab), eps/max(vocab-1,1), device=ids.device, dtype=torch.float32)
    y.scatter_(2, ids.unsqueeze(-1), 1.0 - eps)
    y.masked_fill_((ids==pad_id).unsqueeze(-1), 0.0)
    return y

# ========================
# Checkpoint
# ========================

def save_ckpt(accel: Accelerator, out_dir: Path, epoch: int, global_step: int, best_metric: float):
    ckpt_dir = out_dir / f"checkpoint/ckpt-epoch-{epoch}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    accel.save_state(ckpt_dir)
    meta = {"epoch": epoch, "global_step": global_step, "best_val": best_metric}
    if accel.is_main_process:
        (out_dir / "checkpoint").mkdir(exist_ok=True)
        torch.save(meta, out_dir / "checkpoint/last.pt")
        accel.print(f"[Checkpoint] Saved: {ckpt_dir} (step={global_step}, best_train={best_metric:.6f})")

def auto_resume(accel: Accelerator, out_dir: Path):
    last = out_dir / "checkpoint/last.pt"
    if last.exists():
        meta = torch.load(last, map_location="cpu")
        epoch0 = int(meta.get("epoch", 0))
        step0  = int(meta.get("global_step", 0))
        best   = float(meta.get("best_val", float("inf")))
        ckpt_dir = out_dir / f"checkpoint/ckpt-epoch-{epoch0}"
        if ckpt_dir.exists():
            accel.print(f"[Auto-Resume] Found checkpoint -> {ckpt_dir}")
            accel.load_state(ckpt_dir)
            accel.print(f"[Auto-Resume] Restored epoch={epoch0}, step={step0}, best_train={best:.6f}")
            return epoch0, step0, best
        else:
            accel.print(f"[Auto-Resume] last.pt found but ckpt dir missing: {ckpt_dir}, start fresh.")
    return 0, 0, float("inf")

# ========================
# training
# ========================

def main(args):

    set_seed(42); torch.manual_seed(42); random.seed(42)
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    accelerator = Accelerator(
        gradient_accumulation_steps=args.grad_accum,
        mixed_precision="bf16" if dtype is torch.bfloat16 else "fp16",
        log_with="wandb" if args.wandb else None
    )
    if args.wandb:
        import wandb; wandb.login(); accelerator.init_trackers(project_name=args.wandb)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Tokenizer & Dataset
    rna_tok = AutoTokenizer.from_pretrained(args.rna_tokenizer, use_fast=True)
    pad_id  = rna_tok.pad_token_id
    vocab_size = len(rna_tok)

    full_ds = ProtRnaDataset(args.dataset, rna_tok, max_len=args.seq_len)


    k = min(10, len(full_ds))
    tr = Subset(full_ds, list(range(k)))
    tr_dl = DataLoader(tr, batch_size=args.batch, shuffle=True, collate_fn=collate_fn, pin_memory=True)

    # model
    enc = ProteinEncoder(args.enc_tokenizer, embed_dim=128, hidden_size=args.hidden)
    dec = FlowDecoder(vocab_size=vocab_size, hidden=args.hidden, n_heads=args.heads, n_layers=args.layers, p=0.1)

    params = list(enc.parameters()) + list(dec.parameters())
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-2)


    enc, dec, opt, tr_dl = accelerator.prepare(enc, dec, opt, tr_dl)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)


    start_epoch, global_step, best_train = auto_resume(accelerator, out_dir)

    def run_epoch(loader):
        nonlocal global_step
        model_loss_sum, n_batches = 0.0, 0

        enc.train(); dec.train()

        for batch in loader:
            ids   = batch["rna_ids"].to(accelerator.device)[:, :args.seq_len]   # (B,L)
            prots = batch["prot"]
            B, L  = ids.shape


            with torch.cuda.amp.autocast(enabled=accelerator.mixed_precision!="no"):
                cond_mem_full = enc(prots)                           # (B,Lp,H)
                cond_mem      = torch.nn.functional.adaptive_avg_pool1d(
                    cond_mem_full.transpose(1,2), L
                ).transpose(1,2)                                     # (B,L,H)

                nonpad = (ids != pad_id)                             # (B,L)
                y  = label_smooth_onehot(ids, vocab_size, args.label_smooth, pad_id)  # (B,L,V)
                t  = torch.rand(B, device=ids.device) * (1-1e-3) + 5e-4
                u  = torch.randn_like(y)
                x_t    = (1.0 - t)[:,None,None]*u + t[:,None,None]*y
                v_star = (y - u)

            with accelerator.accumulate(dec):
                v_pred = dec(x_t, cond_mem, t, src_key_padding_mask=~nonpad)
                diff = (v_pred - v_star)
                m = nonpad.unsqueeze(-1)
                loss = (diff.masked_fill(~m, 0.0).pow(2).sum() / m.sum().clamp(min=1))

                accelerator.backward(loss)
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step(); opt.zero_grad(set_to_none=True)

            if accelerator.sync_gradients:
                global_step += 1

            model_loss_sum += float(loss.detach().float().item())
            n_batches += 1

        return model_loss_sum / max(1, n_batches)


    total_epochs = args.epochs
    if start_epoch >= total_epochs:
        accelerator.print(f"[Info] start_epoch ({start_epoch}) >= total_epochs ({total_epochs}); nothing to do.")
        return

    for epoch in range(start_epoch + 1, total_epochs + 1):
        t0 = time.time()
        train_loss = run_epoch(tr_dl)
        epoch_time = time.time() - t0


        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        if args.wandb:
            accelerator.log({
                "epoch": epoch,
                "train/epoch_loss": train_loss,
                "epoch_time_sec": epoch_time,
                "lr": current_lr
            }, step=epoch)

        accelerator.print(
            f"[Epoch {epoch}/{total_epochs}] train_loss={train_loss:.6f} "
            f"| lr={current_lr:.6e} | step={global_step} | time={epoch_time:.2f}s"
        )

        best_train = min(best_train, train_loss)
        save_ckpt(accelerator, out_dir, epoch, global_step, best_train)

    if accelerator.is_main_process:
        torch.save(accelerator.unwrap_model(dec).state_dict(), out_dir / "flow_decoder.pt")
        torch.save(accelerator.unwrap_model(enc).state_dict(), out_dir / "protein_encoder.pt")
        rna_tok.save_pretrained(out_dir)
    accelerator.end_training()
    accelerator.print("Done.")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str,
                        default="/workspace/huangxiaoniu/RNAtranslator/new_val_set1_10.jsonl")
    parser.add_argument("--enc_tokenizer", default='/workspace/huangxiaoniu/diffu/protein_tokenizer')
    parser.add_argument("--rna_tokenizer", default='/workspace/huangxiaoniu/diffu/rna_tokenizer')
    parser.add_argument("--out", type=str,
                        default="./train_fm_out_10")
    parser.add_argument("--seq_len", type=int, default=1024)
    parser.add_argument("--batch", type=int, default=10)
    parser.add_argument("--grad_accum", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--hidden", type=int, default=640)
    parser.add_argument("--heads", type=int, default=10)
    parser.add_argument("--layers", type=int, default=8)
    parser.add_argument("--label_smooth", type=float, default=0.05)
    parser.add_argument("--wandb", type=str, default='Flowmatching_9.28_10')
    args = parser.parse_args()
    main(args)
