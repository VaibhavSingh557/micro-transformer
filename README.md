# Micro-Transformer

A decoder-only Transformer written from scratch in NumPy, including a BPE tokenizer,
hand-written backpropagation, an Adam optimizer, and an interactive attention visualizer.

Built to show how attention, causal masking, and gradient-based training work at the
level of individual matrix operations, with no deep-learning framework.

## Features

- **Byte-Pair Encoding tokenizer** with deterministic training and round-trip decoding
- **Causal multi-head self-attention** with a visible mask
- **Manual backpropagation** through attention, feed-forward, and output layers
- **Gradient-checked:** the test suite compares analytic gradients to finite differences
- **Interactive dashboard:** attention heatmap, subword tokens, and next-token probabilities
- **JSON API** at `POST /api/forward` for programmatic access

## Quick start

Requires Python 3.9+.

```bash
git clone https://github.com/<your-username>/micro-transformer.git
cd micro-transformer
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m unittest -v          # run the test suite
python main.py                 # train, then open http://127.0.0.1:5095
