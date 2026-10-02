"""How far apart are two *correct* implementations after 64 layers? (fp32-linear vs bf16-linear torch reference)."""
from __future__ import annotations
import sys
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qmk.engine import Engine
from qmk.model import Weights
from qmk.torch_model import TorchModel
from test_decode import CKPT, DEV, PROMPTS, rel

def main():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(CKPT)
    w = Weights.load(CKPT, DEV, log=None)
    eng = Engine(w, slots=3, max_ctx=512)
    slots, tokens = [1, 2], []
    for b, s in enumerate(slots):
        text = tok.apply_chat_template([{"role": "user", "content": PROMPTS[b]}], add_generation_prompt=True, tokenize=False)
        tokens.append(int(eng.prefill(tok(text).input_ids, s).argmax()))
    base = eng.cache.clone()
    ref32 = TorchModel(w, base.clone(), precise=True)
    ref16 = TorchModel(w, base.clone(), precise=False)
    for step in range(3):
        a = ref32.decode_step(tokens, slots)
        b = ref16.decode_step(tokens, slots)
        c = eng.decode_step(tokens, slots).float()
        print(f"step {step}: fp32-ref vs bf16-ref {rel(b, a):.2e} | kernel vs fp32-ref {rel(c, a):.2e} | kernel vs bf16-ref {rel(c, b):.2e}")
        tokens = a.argmax(-1).tolist()

if __name__ == "__main__":
    main()
