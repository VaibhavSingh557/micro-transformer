import unittest

import numpy as np

from micro_transformer import Config, MicroTransformer
from micro_transformer.tokenizer import BPETokenizer

SENTENCES = [
    "neural attention layers transform tokens .",
    "causal masking blocks future tokens .",
]


class TokenizerTests(unittest.TestCase):
    def setUp(self):
        self.tok = BPETokenizer(num_merges=50)
        self.tok.train(" ".join(SENTENCES))

    def test_round_trip_recovers_normalized_text(self):
        text = " ".join(SENTENCES)
        ids, _ = self.tok.encode(text)
        self.assertEqual(self.tok.decode(ids), " ".join(text.lower().split()))

    def test_unknown_characters_map_to_unk(self):
        ids, _ = self.tok.encode("zebra")  # 'z' never appears in the corpus
        self.assertIn(BPETokenizer.UNK_ID, ids)


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = Config(d_model=16, n_heads=2, d_ff=16, max_len=32, seed=5)
        cls.model = MicroTransformer(SENTENCES, cls.cfg, num_merges=50)
        cls.losses = cls.model.train(SENTENCES, epochs=60, lr=0.02)

    def test_attention_is_causal_and_normalized(self):
        ids, _ = self.model.tokenizer.encode(SENTENCES[0])
        _, cache = self.model._forward(ids)
        for head in cache["A"]:
            np.testing.assert_allclose(head.sum(axis=-1), 1.0, atol=1e-12)
            self.assertTrue(np.all(np.triu(head, k=1) == 0.0))

    def test_analytic_gradients_match_finite_differences(self):
        m = MicroTransformer(["neural attention layers ."],
                             Config(d_model=8, n_heads=2, d_ff=8, max_len=16, seed=1),
                             num_merges=10)
        ids, _ = m.tokenizer.encode("neural attention layers .")
        inp, tgt = ids[:-1], ids[1:]
        _, grads = m._loss_grads(inp, tgt)

        rng = np.random.default_rng(0)
        eps = 1e-6
        for name, param in m.params.items():
            for _ in range(3):
                idx = tuple(int(rng.integers(0, s)) for s in param.shape)
                original = param[idx]
                param[idx] = original + eps
                loss_plus, _ = m._loss_grads(inp, tgt)
                param[idx] = original - eps
                loss_minus, _ = m._loss_grads(inp, tgt)
                param[idx] = original
                numeric = (loss_plus - loss_minus) / (2 * eps)
                self.assertAlmostEqual(grads[name][idx], numeric, delta=1e-5,
                                       msg=f"gradient mismatch in {name}{idx}")

    def test_training_reduces_loss(self):
        self.assertLess(self.losses[-1], self.losses[0] * 0.5)

    def test_greedy_generation_is_deterministic(self):
        a = self.model.generate("neural attention", max_new_tokens=4)
        b = self.model.generate("neural attention", max_new_tokens=4)
        self.assertEqual(a["completion"], b["completion"])
        self.assertEqual(len(a["attention"]), self.cfg.n_heads)

    def test_empty_prompt_is_rejected(self):
        with self.assertRaises(ValueError):
            self.model.generate("   ")


if __name__ == "__main__":
    unittest.main()
