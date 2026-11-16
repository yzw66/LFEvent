"""Equation, gradient and inference checks without downloading a language model."""

import math
import unittest

import torch
from torch import nn
from torch.nn import functional as F

from lfevent.model import (AugmentedTextFusion, CMAP, LFEvent, LFEventConfig,
                               diagonal_contrastive_loss, lfevent_loss)
from lfevent.utils import batches, gradient_evidence, head_state, load_heads


class TinyEncoder(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.embedding = nn.Embedding(64, dim)
        self.dropout = nn.Dropout(.5)

    def forward(self, input_ids, attention_mask):
        tokens = self.dropout(self.embedding(input_ids)) * attention_mask.unsqueeze(-1)
        return F.normalize(tokens.sum(1) / attention_mask.sum(1, keepdim=True), dim=-1)


class LFEventTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(230)
        self.config = LFEventConfig(text_dim=12, image_dim=20, attention_heads=3, fusion_hidden=16)

    def model(self):
        return LFEvent(self.config, text_encoder=TinyEncoder(12))

    def data(self, count=8):
        return F.normalize(torch.randn(count, 4, 12), dim=-1), torch.randn(count, 20)

    def test_losses_match_explicit_paper_sums(self):
        # Independent logsumexp expression for both Eqs.9 and10, in float64.
        outputs = {key: torch.randn(7, 9, dtype=torch.float64, requires_grad=True)
                   for key in ["prediction", "target", "image_alignment", "text_alignment"]}
        losses = lfevent_loss(outputs, .07, .3)

        def reference(left, right):
            normalized_left = left / left.norm(dim=1, keepdim=True)
            normalized_right = right / right.norm(dim=1, keepdim=True)
            similarity = normalized_left @ normalized_right.T / .07
            return torch.stack([torch.logsumexp(row, dim=0) - row[i]
                                for i, row in enumerate(similarity)]).mean()

        expected_l1 = reference(outputs["prediction"], outputs["target"])
        expected_l2 = reference(outputs["image_alignment"], outputs["text_alignment"])
        torch.testing.assert_close(losses["contrastive"], expected_l1)
        torch.testing.assert_close(losses["modal"], expected_l2)
        torch.testing.assert_close(losses["loss"], expected_l1 + .3 * expected_l2)
        losses["loss"].backward()
        for value in outputs.values():
            self.assertGreater(float(value.grad.norm()), 0)

    def test_negatives_and_pairing_affect_loss(self):
        features = torch.eye(6)
        aligned = diagonal_contrastive_loss(features, features, .07)
        shuffled = diagonal_contrastive_loss(features, features.roll(1, 0), .07)
        self.assertLess(float(aligned), .001)
        self.assertGreater(float(shuffled), 10.)
        with self.assertRaises(ValueError):
            diagonal_contrastive_loss(features[:1], features[:1], .07)

    def test_cmap_matches_equations_7_and_8(self):
        cmap = CMAP(12, 20).double()
        text, image = torch.randn(5, 12, dtype=torch.float64), torch.randn(5, 20, dtype=torch.float64)
        query, key, value = text @ cmap.q.weight.T, image @ cmap.k.weight.T, image @ cmap.v.weight.T
        scores = torch.exp(query @ key.T / math.sqrt(12))
        expected = (scores / scores.sum(1, keepdim=True)) @ value @ cmap.o.weight.T
        torch.testing.assert_close(cmap(text, image), expected)
        changed = text.clone()
        changed[0] *= -5
        self.assertGreater(float((cmap(changed, image)[0] - expected[0]).detach().norm()), 1e-5)

    def test_patch_cmap_is_sample_isolated_and_rejects_single_key(self):
        cmap = CMAP(12, 20)
        text, images = torch.randn(5, 12), torch.randn(5, 6, 20)
        result = cmap(text, images)
        images[1:] += 100
        torch.testing.assert_close(result[0], cmap(text, images)[0])
        with self.assertRaisesRegex(ValueError, "single image token"):
            cmap(text, images[:, :1])

    def test_four_source_attention_gates_and_sample_isolation(self):
        fusion = AugmentedTextFusion(12, 3, 16).eval()
        sources, _ = self.data()
        result, weights = fusion(sources, return_weights=True)
        self.assertEqual(result.shape, (8, 12))
        torch.testing.assert_close(weights.sum(1), torch.ones(8))
        for source in range(4):
            changed = sources.clone()
            changed[0, source] += torch.randn(12) * 3
            output = fusion(changed)
            self.assertGreater(float((output[0] - result[0]).detach().norm()), 1e-5)
            torch.testing.assert_close(output[1:], result[1:])

    def test_every_new_module_receives_gradients_and_updates(self):
        model = self.model().train()
        sources, images = self.data()
        before = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
        outputs = model.forward_features(sources, images)
        self.assertTrue(outputs["target"].requires_grad)
        losses = model.loss(outputs)
        losses["loss"].backward()
        evidence = gradient_evidence(model)
        self.assertTrue(all(value > 0 and math.isfinite(value) for value in evidence.values()))
        torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=.005).step()
        for prefix in ["augmented_fusion.attention", "augmented_fusion.fusion", "cmap", "predictor"] + [
                f"augmented_fusion.gates.{i}" for i in range(4)]:
            self.assertTrue(any(not torch.equal(before[name], p) for name, p in model.named_parameters()
                                if name.startswith(prefix + ".")), prefix)
        for name, parameter in model.named_parameters():
            if name.startswith("text_encoder."):
                torch.testing.assert_close(parameter, before[name], rtol=0, atol=0)
                self.assertIsNone(parameter.grad)

    def test_modal_loss_alone_trains_shared_cmap_projections(self):
        # L2 must influence inference-time features, not just standalone discarded heads.
        model = self.model().train()
        sources, images = self.data()
        losses = model.loss(model.forward_features(sources, images))
        losses["modal"].backward()
        for projection in [model.cmap.q, model.cmap.k]:
            self.assertIsNotNone(projection.weight.grad)
            self.assertGreater(float(projection.weight.grad.norm()), 0)
        self.assertTrue(all(p.grad is None for p in model.text_encoder.parameters()))

    def test_online_cached_forward_equivalence_and_frozen_eval_policy(self):
        model = self.model()
        sources = [{"input_ids": torch.randint(0, 64, (8, 7)), "attention_mask": torch.ones(8, 7)}
                   for _ in range(4)]
        images = torch.randn(8, 20)
        model.train()
        self.assertFalse(model.text_encoder.training)
        self.assertFalse(model.text_encoder.dropout.training)
        encoded1, encoded2 = model.encode_sources(sources), model.encode_sources(sources)
        torch.testing.assert_close(encoded1, encoded2, rtol=0, atol=0)
        model.eval()
        cached = model.forward_features(encoded1, images)
        online = model(sources, images)
        for key in cached:
            torch.testing.assert_close(cached[key], online[key], rtol=0, atol=0)
        # Detection needs only original input; no enhanced annotations at inference.
        torch.testing.assert_close(model.cluster_feature(sources[0], images), online["prediction"], rtol=0, atol=0)

    def test_compact_checkpoint_restores_inference(self):
        model = self.model().eval()
        sources, images = self.data()
        expected = model.cluster_features(sources[:, 0], images)
        restored = self.model().eval()
        load_heads(restored, head_state(model))
        torch.testing.assert_close(restored.cluster_features(sources[:, 0], images), expected, rtol=0, atol=0)

    def test_batches_preserve_every_row_without_singletons(self):
        for count in [2, 63, 64, 65, 128, 129, 257]:
            groups = batches(count, 64)
            self.assertTrue(all(len(group) >= 2 for group in groups))
            self.assertEqual([i for group in groups for i in group], list(range(count)))

    def test_invalid_config_is_rejected(self):
        for options in [{"temperature": 0}, {"temperature": float("nan")}, {"modal_weight": -1},
                        {"attention_heads": 5}, {"text_dim": 0}]:
            with self.assertRaises(ValueError):
                LFEventConfig(**options)


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main()
