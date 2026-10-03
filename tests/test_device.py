import torch

from chessslm.device import get_device


def test_get_device_returns_supported_device() -> None:
    device = get_device()

    assert isinstance(device, torch.device)
    assert device.type in {
        "cpu",
        "cuda",
        "mps",
    }
