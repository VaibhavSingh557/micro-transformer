"""A decoder-only (causal) Transformer in NumPy with hand-written backpropagation.

Architecture (one block, kept deliberately small and readable):

    X      = E[tokens] + P[positions]
    H1     = X + MultiHeadCausalAttention(X) @ Wo
    H2     = H1 + relu(H1 @ W1 + b1) @ W2 + b2
    logits = H2 @ Wout + bout

Simplifications versus production models: no LayerNorm, no dropout,
learned positional embeddings instead of RoPE, a single layer.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from .tokenizer import BPETokenizer


@dataclass(frozen=True)
class Config:
    d_model: int = 32
    n_heads: int = 4
    d_ff: int = 64
    max_len: int = 48
    seed: int = 42

    def __post_init__(self) -> None:
        if self.d_model % self.n_heads:
            raise ValueError("d_model must be divisible by n_heads")


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


def _split_heads(M: np.ndarray, n_heads: int) -> np.ndarray:
    T, d = M.shape
    return M.reshape(T, n_heads, d // n_heads).transpose(1, 0, 2)  # (H, T, dk)


def _merge_heads(M: np.ndarray) -> np.ndarray:
    H, T, dk = M.shape
    return M.transpose(1, 0, 2).reshape(T, H * dk)  # (T, H*dk)


class MicroTransformer:
    def __init__(self, sentences: list[str], cfg: Config = Config(), num_merges: int = 100) -> None:
        self.cfg = cfg
        self.tokenizer = BPETokenizer(num_merges=num_merges)
        self.tokenizer.train(" ".join(sentences))

        V = len(self.tokenizer.vocab)
        d, f, L = cfg.d_model, cfg.d_ff, cfg.max_len
        rng = np.random.default_rng(cfg.seed)

        def mat(rows: int, cols: int, scale: float) -> np.ndarray:
            return rng.normal(0.0, scale, size=(rows, cols))

        s_d, s_f = 1.0 / np.sqrt(d), 1.0 / np.sqrt(f)
        self.params: dict[str, np.ndarray] = {
            "E": mat(V, d, 0.1),
            "P": mat(L, d, 0.1),
            "Wq": mat(d, d, s_d),
            "Wk": mat(d, d, s_d),
            "Wv": mat(d, d, s_d),
            "Wo": mat(d, d, s_d),
            "W1": mat(d, f, s_d),
            "b1": np.zeros(f),
            "W2": mat(f, d, s_f),
            "b2": np.zeros(d),
            "Wout": mat(d, V, s_d),
            "bout": np.zeros(V),
        }
        self._m = {k: np.zeros_like(v) for k, v in self.params.items()}
        self._v = {k: np.zeros_like(v) for k, v in self.params.items()}
        self._t = 0
        self.loss_history: list[float] = []

    # ------------------------------------------------------------------ forward
    def _forward(self, ids):
        p, cfg = self.params, self.cfg
        ids = np.asarray(ids, dtype=int)
        T = len(ids)
        H, dk = cfg.n_heads, cfg.d_model // cfg.n_heads

        X = p["E"][ids] + p["P"][:T]
        Q, K, V = X @ p["Wq"], X @ p["Wk"], X @ p["Wv"]
        Qh, Kh, Vh = _split_heads(Q, H), _split_heads(K, H), _split_heads(V, H)

        scores = Qh @ Kh.transpose(0, 2, 1) / np.sqrt(dk)  # (H, T, T)
        causal = np.triu(np.ones((T, T), dtype=bool), k=1)
        scores = np.where(causal, -np.inf, scores)
        A = _softmax(scores)

        Cc = _merge_heads(A @ Vh)
        H1 = X + Cc @ p["Wo"]

        a = H1 @ p["W1"] + p["b1"]
        Hf = np.maximum(a, 0.0)
        H2 = H1 + Hf @ p["W2"] + p["b2"]

        logits = H2 @ p["Wout"] + p["bout"]
        cache = {"ids": ids, "X": X, "Qh": Qh, "Kh": Kh, "Vh": Vh,
                 "A": A, "Cc": Cc, "H1": H1, "a": a, "Hf": Hf, "H2": H2}
        return logits, cache

    # ----------------------------------------------------------------- backward
    def _loss_grads(self, inp, tgt):
        """Mean next-token cross-entropy and gradients for every parameter."""
        p, cfg = self.params, self.cfg
        logits, c = self._forward(inp)
        tgt = np.asarray(tgt, dtype=int)
        T = len(tgt)
        H, dk = cfg.n_heads, cfg.d_model // cfg.n_heads
        rows = np.arange(T)

        probs = _softmax(logits)
        loss = float(-np.log(probs[rows, tgt]).mean())

        dlogits = probs.copy()
        dlogits[rows, tgt] -= 1.0
        dlogits /= T

        g = {}
        # Output projection
        g["Wout"] = c["H2"].T @ dlogits
        g["bout"] = dlogits.sum(axis=0)
        dH2 = dlogits @ p["Wout"].T

        # Feed-forward block (residual: H2 = H1 + FFN(H1))
        g["W2"] = c["Hf"].T @ dH2
        g["b2"] = dH2.sum(axis=0)
        da = (dH2 @ p["W2"].T) * (c["a"] > 0)
        g["W1"] = c["H1"].T @ da
        g["b1"] = da.sum(axis=0)
        dH1 = dH2 + da @ p["W1"].T

        # Attention output projection (residual: H1 = X + Cc @ Wo)
        g["Wo"] = c["Cc"].T @ dH1
        dCtx = _split_heads(dH1 @ p["Wo"].T, H)

        # Attention core
        A, Vh, Qh, Kh = c["A"], c["Vh"], c["Qh"], c["Kh"]
        dA = dCtx @ Vh.transpose(0, 2, 1)
        dVh = A.transpose(0, 2, 1) @ dCtx
        # Softmax Jacobian; masked entries have A == 0 so their gradient is 0.
        dScores = A * (dA - np.sum(dA * A, axis=-1, keepdims=True)) / np.sqrt(dk)
        dQh = dScores @ Kh
        dKh = dScores.transpose(0, 2, 1) @ Qh

        dQ, dK, dV = _merge_heads(dQh), _merge_heads(dKh), _merge_heads(dVh)
        X = c["X"]
        g["Wq"] = X.T @ dQ
        g["Wk"] = X.T @ dK
        g["Wv"] = X.T @ dV
        dX = dH1 + dQ @ p["Wq"].T + dK @ p["Wk"].T + dV @ p["Wv"].T

        # Embedding and positional gradients
        gE = np.zeros_like(p["E"])
        np.add.at(gE, c["ids"], dX)
        gP = np.zeros_like(p["P"])
        gP[:T] = dX
        g["E"], g["P"] = gE, gP
        return loss, g

    def _adam_step(self, grads, lr, beta1=0.9, beta2=0.999, eps=1e-8) -> None:
        self._t += 1
        for name, grad in grads.items():
            m = beta1 * self._m[name] + (1 - beta1) * grad
            v = beta2 * self._v[name] + (1 - beta2) * grad ** 2
            self._m[name], self._v[name] = m, v
            m_hat = m / (1 - beta1 ** self._t)
            v_hat = v / (1 - beta2 ** self._t)
            self.params[name] -= lr * m_hat / (np.sqrt(v_hat) + eps)

    # ----------------------------------------------------------------- public
    def train(self, sentences: list[str], epochs: int = 600, lr: float = 0.01) -> list[float]:
        seqs = []
        for s in sentences:
            ids, _ = self.tokenizer.encode(s)
            ids = ids[: self.cfg.max_len]
            if len(ids) >= 2:
                seqs.append(ids)
        if not seqs:
            raise ValueError("No sentence has at least two tokens to train on.")

        for _ in range(epochs):
            total = 0.0
            for ids in seqs:
                loss, grads = self._loss_grads(ids[:-1], ids[1:])
                self._adam_step(grads, lr)
                total += loss
            self.loss_history.append(total / len(seqs))
        return self.loss_history

    def num_params(self) -> int:
        return int(sum(v.size for v in self.params.values()))

    def generate(self, prompt: str, max_new_tokens: int = 6, temperature: float = 0.0, seed: int = 0) -> dict:
        t0 = time.perf_counter()
        cfg = self.cfg
        max_new_tokens = max(1, min(int(max_new_tokens), cfg.max_len - 1))

        ids, _ = self.tokenizer.encode(prompt)
        if not ids:
            raise ValueError("Prompt contains no words.")
        ctx = ids[-(cfg.max_len - max_new_tokens):]
        context_tokens = list(ctx)

        rng = np.random.default_rng(seed)
        generated: list[int] = []
        first_attn = None
        top_preds: list[dict] = []

        for step in range(max_new_tokens):
            logits, cache = self._forward(ctx)
            if first_attn is None:
                first_attn = cache["A"]

            last = logits[-1].copy()
            last[self.tokenizer.UNK_ID] = -np.inf  # never generate <unk>

            if step == 0:
                probs0 = _softmax(last)
                for idx in np.argsort(probs0)[::-1][:5]:
                    top_preds.append({"token": self.tokenizer.piece(int(idx)),
                                      "prob": round(float(probs0[idx]) * 100, 2)})

            if temperature <= 0:
                nxt = int(np.argmax(last))
            else:
                probs = _softmax(last / temperature)
                nxt = int(rng.choice(len(probs), p=probs))
            ctx.append(nxt)
            generated.append(nxt)

        return {
            "prompt": prompt,
            "tokens": [self.tokenizer.piece(i) for i in context_tokens],
            "input_ids": context_tokens,
            "attention": first_attn.tolist(),  # (heads, T, T), first layer
            "completion": self.tokenizer.decode(generated),
            "top_predictions": top_preds,
            "telemetry": {
                "num_params": self.num_params(),
                "vocab_size": len(self.tokenizer.vocab),
                "num_heads": cfg.n_heads,
                "train_loss_start": round(self.loss_history[0], 4) if self.loss_history else None,
                "train_loss_final": round(self.loss_history[-1], 4) if self.loss_history else None,
                "latency_ms": round((time.perf_counter() - t0) * 1000, 2),
            },
        }
