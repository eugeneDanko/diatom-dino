from __future__ import annotations

import unittest

try:
    import torch
except (ImportError, OSError):  # local maintenance environment may not have a usable PyTorch build
    torch = None


@unittest.skipIf(torch is None, "PyTorch is unavailable")
class ContrastiveLossTest(unittest.TestCase):
    def test_loss_is_finite_and_differentiable(self) -> None:
        from classifier.contrastive_loss import HierarchicalContrastiveLoss

        genus = torch.randn(8, 16, requires_grad=True)
        species = torch.randn(8, 16, requires_grad=True)
        genus_labels = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
        species_labels = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
        losses = HierarchicalContrastiveLoss(hierarchy_weight=0.2)(
            genus, species, genus_labels, species_labels
        )
        self.assertTrue(torch.isfinite(losses["total"]))
        losses["total"].backward()
        self.assertIsNotNone(genus.grad)
        self.assertIsNotNone(species.grad)

    def test_no_positive_pairs_returns_differentiable_zero(self) -> None:
        from classifier.contrastive_loss import SupervisedContrastiveLoss

        features = torch.randn(4, 8, requires_grad=True)
        labels = torch.arange(4)
        loss = SupervisedContrastiveLoss()(features, labels)
        self.assertEqual(float(loss), 0.0)
        loss.backward()
        self.assertIsNotNone(features.grad)

    def test_view_consistency_is_zero_for_identical_views(self) -> None:
        from classifier.contrastive_loss import HierarchicalContrastiveLoss

        physical = torch.randn(4, 8)
        embeddings = torch.cat([physical, physical], dim=0)
        loss = HierarchicalContrastiveLoss._view_consistency(embeddings, 2)
        self.assertAlmostEqual(float(loss), 0.0, places=6)

    def test_view_consistency_validates_view_layout(self) -> None:
        from classifier.contrastive_loss import HierarchicalContrastiveLoss

        with self.assertRaises(ValueError):
            HierarchicalContrastiveLoss._view_consistency(torch.randn(5, 8), 2)


if __name__ == "__main__":
    unittest.main()
