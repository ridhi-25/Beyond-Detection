"""Safe unit tests for the pretrained checkpoint layer-index remapping."""

from __future__ import annotations

import unittest

from training.train_fwb import _shift_layer_index


class FWBTransferMappingTests(unittest.TestCase):
    def test_backbone_indices_are_unchanged(self) -> None:
        self.assertEqual(_shift_layer_index("model.0.conv.weight"), "model.0.conv.weight")
        self.assertEqual(_shift_layer_index("model.9.cv2.conv.weight"), "model.9.cv2.conv.weight")

    def test_neck_and_detect_indices_shift_after_inserted_fwb(self) -> None:
        self.assertEqual(_shift_layer_index("model.10.cv1.conv.weight"), "model.11.cv1.conv.weight")
        self.assertEqual(_shift_layer_index("model.12.m.0.cv1.conv.weight"), "model.13.m.0.cv1.conv.weight")
        self.assertEqual(_shift_layer_index("model.22.cv2.0.conv.weight"), "model.23.cv2.0.conv.weight")

    def test_non_layer_state_keys_are_unchanged(self) -> None:
        self.assertEqual(_shift_layer_index("stride"), "stride")


if __name__ == "__main__":
    unittest.main()
