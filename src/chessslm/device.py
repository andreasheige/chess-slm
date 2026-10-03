import torch


def get_device() -> torch.device:
    """Select the best available device for running the model.

    PyTorch operations run on a device. Both the model parameters and
    the tensors passed to the model must live on the same device.

    Device priority:

    1. CUDA - NVIDIA GPU
    2. MPS  - Apple Silicon GPU
    3. CPU  - always available fallback

    Keeping device selection in one place means the rest of the
    training code does not need hardware-specific logic.
    """

    if torch.cuda.is_available():
        return torch.device("cuda")

    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")
