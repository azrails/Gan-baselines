from abc import abstractmethod

import torch


class BaseMetric:
    """
    Base class for all metrics
    """

    def __init__(self, device, name=None, *args, **kwargs):
        """
        Args:
            name (str | None): metric name to use in logger and writer.
        """
        if device == "auto":
            device = (
                "cuda"
                if torch.cuda.is_available()
                else ("mps" if torch.backends.mps.is_available() else "cpu")
            )
        else:
            device = device
        self.device = device
        self.name = name if name is not None else type(self).__name__

    @abstractmethod
    def __call__(self, **batch):
        """
        Defines metric calculation logic for a given batch.
        Can use external functions (like TorchMetrics) or custom ones.
        """
        raise NotImplementedError()
