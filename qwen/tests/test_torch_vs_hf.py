"""Check qmk.torch_model against transformers' Qwen3_5 on a truncated real checkpoint.

Same (dequantized) weights in both; validates our reading of the architecture
(prefill + one cached decode step).
Run: .venv/bin/python qwen/tests/test_torch_vs_hf.py [--layers 4]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qmk.cache import Cache  # noqa: E402
from qmk.model import FullLayerWeights, Weights  # noqa: E402
from qmk.torch_model import TorchModel  # noqa: E402

CKPT = Path(__file__).resolve().parents[2] / "models" / "Qwen3.8-27B-FP8"
DEV = torch.device("cuda")


def build_hf(w: Weights):
    from transformers import AutoConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM

    full = AutoConfig.from_pretrained(CKPT)
    tc = full.text_config
    tc.num_hidden_layers = w.cfg.layers
    tc.layer_types = list(w.cfg.layer_types)
    tc.mtp_num_hidden_layers = 0
    tc._attn_implementation = "sdpa"
    tc.dtype = torch.bfloat16
    with torch.device("meta"):
        model = Qwen3_5ForCausalLM(tc)
    sd = {"model.embed_tokens.weight": w.embed, "model.norm.weight": w.final_norm,
          "lm_head.weight": w.lm_head.dense()}
    for i, lw in enumerate(w.layers):
        p = f"model.layers.{i}."
        sd[p + "input_layernorm.weight"] = lw.ln1
        sd[p + "post_attention_layernorm.weight"] = lw.ln2
        gu = lw.gate_up.split_dense()
        sd[p + "mlp.gate_proj.weight"], sd[p + "mlp.up_proj.weight"] = gu["gate"], gu["up"]
        sd[p + "mlp.down_proj.weight"] = lw.down.dense()
        if isinstance(lw, FullLayerWeights):
            a = p + "self_attn."
            qkv = lw.qkv.split_dense()
            sd[a + "q_proj.weight"], sd[a + "k_proj.weight"], sd[a + "v_proj.weight"] = qkv["q"], qkv["k"], qkv["v"]
            sd[a + "o_proj.weight"] = lw.out.dense()
            sd[a + "q_norm.weight"], sd[a + "k_norm.weight"] = lw.q_norm, lw.k_norm
        else:
            a = p + "linear_attn."
            qkvz = lw.qkvz.split_dense()
            ab = lw.ab.split_dense()
            sd[a + "in_proj_qkv.weight"], sd[a + "in_proj_z.weight"] = qkvz["qkv"], qkvz["z"]
            sd[a + "in_proj_b.weight"], sd[a + "in_proj_a.weight"] = ab["b"], ab["a"]
            sd[a + "out_proj.weight"] = lw.out.dense()
            sd[a + "conv1d.weight"] = lw.conv_w[:, None, :]
            sd[a + "A_log"] = lw.a_log.to(torch.bfloat16)
            sd[a + "dt_bias"] = lw.dt_bias.to(torch.bfloat16)
            sd[a + "norm.weight"] = lw.norm_w
    sd = {k: v.to(DEV, torch.bfloat16) for k, v in sd.items()}
    model.load_state_dict(sd, strict=True, assign=True)
    model.model.rotary_emb.__init__(tc, device=DEV)
    return model.eval()


def compare(name: str, a: torch.Tensor, b: torch.Tensor) -> None:
    a, b = a.float(), b.float()
    rel = ((a - b).norm() / b.norm()).item()
    top_a, top_b = a.topk(5).indices.tolist(), b.topk(5).indices.tolist()
    print(f"{name}: rel L2 {rel:.2e}  max abs {(a - b).abs().max().item():.3g}  top5 ours {top_a} hf {top_b}")
    assert rel < 2e-2, f"{name}: mismatch"
    assert top_a[0] == top_b[0], f"{name}: argmax differs"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, default=4)
    args = ap.parse_args()
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(CKPT)
    w = Weights.load(CKPT, DEV, layers=args.layers)
    text = tok.apply_chat_template([{"role": "user", "content": "Write a haiku about GPUs and memory bandwidth."}],
                                   add_generation_prompt=True, tokenize=False)
    ids = tok(text).input_ids
    print(f"{len(ids)} prompt tokens, {args.layers} layers")

    hf = build_hf(w)
    with torch.no_grad():
        out = hf(torch.tensor([ids], device=DEV), use_cache=True)
        hf_prefill = out.logits[0, -1]
        nxt = int(hf_prefill.argmax())
        hf_step = hf(torch.tensor([[nxt]], device=DEV), past_key_values=out.past_key_values, use_cache=True).logits[0, -1]

    cache = Cache(w.cfg, slots=2, max_ctx=1024, device=DEV)
    tm = TorchModel(w, cache)
    ours_prefill = tm.prefill(ids, slot=1)
    compare("prefill", ours_prefill, hf_prefill)
    # chunked prefill must match one-shot prefill
    cache2 = Cache(w.cfg, slots=2, max_ctx=1024, device=DEV)
    chunked = TorchModel(w, cache2).prefill(ids, slot=0, chunk=7)
    compare("chunked prefill", chunked, ours_prefill)
    ours_step = TorchModel(w, cache, precise=True).decode_step([nxt], [1])[0]
    compare("decode step", ours_step, hf_step)
    print("OK")


if __name__ == "__main__":
    main()
