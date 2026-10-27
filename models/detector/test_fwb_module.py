"""Independent FWB shape, finiteness, gradient, and parameter tests."""

from __future__ import annotations

import unittest
from copy import deepcopy
from pathlib import Path

import torch
import ultralytics
import yaml
from ultralytics.nn import tasks

from models.detector.fwb_module import FWB
from models.detector.fwb_registration import register_fwb


class FWBTests(unittest.TestCase):
    def test_preserves_shape_and_backpropagates_across_feature_sizes(self) -> None:
        for shape in ((2, 8, 16, 16), (1, 256, 20, 20), (1, 512, 10, 10)):
            with self.subTest(shape=shape):
                module = FWB(shape[1])
                features = torch.randn(*shape, requires_grad=True)
                output = module(features)
                self.assertEqual(output.shape, features.shape)
                self.assertTrue(torch.isfinite(output).all().item())
                output.square().mean().backward()
                self.assertIsNotNone(features.grad)
                self.assertTrue(torch.isfinite(features.grad).all().item())
                self.assertTrue(
                    all(
                        parameter.grad is not None
                        and torch.isfinite(parameter.grad).all().item()
                        for parameter in module.parameters()
                    )
                )

    def test_expected_parameter_count(self) -> None:
        module = FWB(256)
        self.assertEqual(sum(parameter.numel() for parameter in module.parameters()), 256 * 256 + 256)

    def test_rejects_channel_changes(self) -> None:
        with self.assertRaisesRegex(ValueError, "must preserve channels"):
            FWB(32, 16)

    def test_rejects_wrong_runtime_channel_count(self) -> None:
        module = FWB(8)
        with self.assertRaisesRegex(ValueError, "expected a BCHW tensor"):
            module(torch.randn(1, 4, 8, 8))

    def test_registered_parser_wires_fwb_as_channel_preserving_layer(self) -> None:
        register_fwb()
        config = {
            "nc": 15,
            "scale": "n",
            "scales": {"n": [1.0, 0.25, 1024]},
            "backbone": [
                [-1, 1, "Conv", [1024, 1, 1]],
                [-1, 1, "FWB", [1024]],
            ],
            "head": [[-1, 1, "Classify", [15]]],
        }
        network, _ = tasks.parse_model(config, ch=3, verbose=False)
        fwb_layer = network[1]
        self.assertIsInstance(fwb_layer, FWB)
        self.assertEqual(fwb_layer.c1, 256)
        seen: list[tuple[torch.Size, torch.Size]] = []
        fwb_layer.register_forward_hook(
            lambda _module, inputs, output: seen.append((inputs[0].shape, output.shape))
        )
        with torch.inference_mode():
            logits = network(torch.randn(2, 3, 16, 16))
        self.assertEqual(tuple(logits.shape), (2, 15))
        self.assertEqual(seen[0][0][1], 256)
        self.assertEqual(seen[0][0], seen[0][1])

    def test_custom_yaml_preserves_stock_graph_and_shifts_absolute_references(self) -> None:
        package_root = Path(ultralytics.__file__).resolve().parent
        stock_path = package_root / "cfg" / "models" / "v8" / "yolov8.yaml"
        custom_path = Path(__file__).with_name("yolov8n_fwb.yaml")
        stock = yaml.safe_load(stock_path.read_text(encoding="utf-8"))
        custom = yaml.safe_load(custom_path.read_text(encoding="utf-8"))
        stock_layers = stock["backbone"] + stock["head"]
        custom_layers = custom["backbone"] + custom["head"]

        self.assertEqual(len(custom_layers), len(stock_layers) + 1)
        self.assertEqual(custom["nc"], 15)
        self.assertEqual(custom["scale"], "n")
        self.assertEqual(custom_layers[10], [-1, 1, "FWB", [1024]])
        self.assertEqual(custom_layers[:10], stock_layers[:10])

        def shifted_reference(reference, original_layer_index):
            if isinstance(reference, int):
                if original_layer_index == 20 and reference == 9:
                    return 10
                return reference + 1 if reference >= 10 else reference
            return [
                10 if original_layer_index == 20 and item == 9
                else item + 1 if item >= 10
                else item
                for item in reference
            ]

        for original_index, original_row in enumerate(stock_layers[10:], start=10):
            expected = deepcopy(original_row)
            expected[0] = shifted_reference(expected[0], original_index)
            self.assertEqual(custom_layers[original_index + 1], expected)


if __name__ == "__main__":
    unittest.main()
