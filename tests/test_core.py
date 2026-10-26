import tempfile
import unittest
from pathlib import Path
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM
from abliteration.core import edit_weights, intervention, project, unit_direction, mean_kl, refused


class ProjectionTests(unittest.TestCase):
    def test_projection_removes_only_parallel_component(self):
        x = torch.tensor([[2., 3., 4.], [-2., 1., 5.]])
        r = torch.tensor([1., 2., 3.]); r /= r.norm()
        y = project(x, r)
        self.assertTrue(torch.allclose(y @ r, torch.zeros(2), atol=1e-6))
        self.assertTrue(torch.allclose(project(y, r), y, atol=1e-6))
        self.assertTrue(torch.allclose(x - y, (x @ r)[:, None] * r, atol=1e-6))

    def test_degenerate_direction_is_rejected(self):
        with self.assertRaises(ValueError):
            unit_direction(torch.zeros(2, 3, 4), torch.zeros(2, 3, 4))

    def test_kl_identity_and_detector(self):
        logits = torch.randn(5, 7)
        self.assertAlmostEqual(mean_kl(logits, logits), 0.)
        self.assertTrue(refused("I cannot help with that request."))
        self.assertFalse(refused("The answer is 42."))

    def test_weight_edit_matches_hooks_and_survives_reload(self):
        torch.manual_seed(9)
        cfg = Qwen2Config(vocab_size=32, hidden_size=16, intermediate_size=32,
                          num_hidden_layers=2, num_attention_heads=4,
                          num_key_value_heads=2, tie_word_embeddings=True)
        model = Qwen2ForCausalLM(cfg).eval()
        ids = torch.tensor([[1, 3, 8, 2]])
        head = model.lm_head.weight.detach().clone()
        original = model(ids).logits.detach()
        r = torch.randn(16); r /= r.norm()
        with intervention(model, r):
            hooked = model(ids).logits.detach()
        self.assertTrue(torch.allclose(model(ids).logits, original))
        records = edit_weights(model, r)
        self.assertEqual(len(records), 5)
        self.assertTrue(torch.equal(model.lm_head.weight, head))
        self.assertNotEqual(model.lm_head.weight.data_ptr(), model.model.embed_tokens.weight.data_ptr())
        self.assertTrue(torch.allclose(model(ids).logits, hooked, atol=1e-5))
        with tempfile.TemporaryDirectory() as folder:
            model.save_pretrained(folder)
            loaded = Qwen2ForCausalLM.from_pretrained(folder)
            self.assertTrue(torch.allclose(loaded(ids).logits, hooked, atol=1e-5))
            self.assertFalse(loaded.config.tie_word_embeddings)


if __name__ == "__main__":
    unittest.main()
