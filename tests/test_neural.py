"""Optional CPU forward checks; no optimization or recorded data."""
import importlib.util
import unittest
import numpy as np


@unittest.skipUnless(importlib.util.find_spec("torch"), "optional torch dependency")
class NeuralForwardTests(unittest.TestCase):
    def test_reference_shapes_ranges_and_window_independence(self):
        import torch
        from quadfdi.neural import build_network
        from quadfdi.common import seed_neural
        torch.set_num_threads(1)
        x = torch.from_numpy(np.random.default_rng(41).normal(size=(3, 16, 10)).astype(np.float32))
        for method in ("mlp", "fcn", "lstm", "tcn", "transformer"):
            with self.subTest(method=method):
                seed_neural(20261009)
                network = build_network(method, 16, {"hidden_size": 64}).eval()
                with torch.no_grad():
                    together = network(x)
                    split = torch.cat([network(x[i:i + 1]) for i in range(3)])
                self.assertEqual(tuple(together.shape), (3, 4))
                self.assertTrue(torch.isfinite(together).all())
                self.assertTrue(((together >= 0) & (together <= 1)).all())
                torch.testing.assert_close(together, split, atol=1e-6, rtol=1e-5)

    def test_control_is_a_distinct_five_class_network(self):
        import torch
        from quadfdi.neural import build_control_network
        network = build_control_network()
        self.assertEqual(sum(p.numel() for p in network.parameters()), 34437)
        with torch.no_grad():
            logits = network(torch.zeros(2, 201))
        self.assertEqual(tuple(logits.shape), (2, 5))


if __name__ == "__main__":
    unittest.main()
